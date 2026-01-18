import os
import json
import re
import sqlite3
import threading
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from financial_analyzer_core import AppConfig, AnalysisResult, ProgressCallback, get_base_dir
from fa_platform.jsonx import sanitize_json
from fa_platform.pipeline import build_artifacts as _build_artifacts_common

_TOOL_ID = os.path.basename(os.path.dirname(__file__))


def _get_logger(name: str = "validation_report") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def _tool_params(cfg: AppConfig) -> Dict[str, Any]:
    tp = getattr(cfg, "tool_params", None)
    if not isinstance(tp, dict):
        return {}
    tid = str(getattr(cfg, "tool_id", "") or "").strip()
    if tid:
        bucket = tp.get(tid)
        if isinstance(bucket, dict):
            return bucket
    bucket = tp.get(_TOOL_ID)
    return bucket if isinstance(bucket, dict) else {}


def _get_param(cfg: AppConfig, key: str, default: Any) -> Any:
    params = _tool_params(cfg)
    if key in params:
        return params.get(key)
    if hasattr(cfg, key):
        v = getattr(cfg, key)
        return default if v is None else v
    return default


def _rules_path(source_tool_id: str) -> str:
    base = os.path.abspath(get_base_dir())
    tid = str(source_tool_id or "").strip()
    if tid:
        p = os.path.join(base, "tools", tid, "rules.json")
        if os.path.exists(p):
            return p
    return ""


def _load_rules(source_tool_id: str) -> Dict[str, Any]:
    p = _rules_path(source_tool_id)
    if not p or not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _normalize_subject_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s,，]", "", s)
    return s


def _expand_keywords(source_tool_id: str, keywords: List[Any]) -> List[str]:
    rules = _load_rules(source_tool_id)
    aliases: Dict[str, set] = {}
    sa = rules.get("subject_aliases")
    if isinstance(sa, dict):
        for canon, syns in sa.items():
            canon_s = str(canon or "").strip()
            if not canon_s:
                continue
            items = [canon_s]
            if isinstance(syns, list):
                items.extend([str(x or "").strip() for x in syns if str(x or "").strip()])
            variants = list(dict.fromkeys(items))
            for v in variants:
                aliases.setdefault(_normalize_subject_text(v), set()).update(variants)

    out: List[str] = []
    seen = set()
    for kw in (keywords or []):
        s = str(kw or "").strip()
        if not s:
            continue
        norm = _normalize_subject_text(s)
        variants = aliases.get(norm)
        if variants:
            for v in variants:
                if v not in seen:
                    out.append(v)
                    seen.add(v)
        else:
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out


def _df_preview_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    df2 = df.head(int(limit)).copy()
    try:
        df2 = df2.astype(object)
    except Exception:
        pass
    for c in df2.columns:
        try:
            df2[c] = pd.Series([sanitize_json(x) for x in df2[c].tolist()], dtype="object")
        except Exception:
            pass
    return df2.to_dict(orient="records")


def _open_source_run(source_tool_id: str, source_run_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    tid = str(source_tool_id or "").strip()
    rid = str(source_run_id or "").strip()
    if not tid:
        return None, "source_tool_id 不能为空"
    try:
        from fa_platform.run_index import get_latest_run, get_run

        info = get_run(tid, rid) if rid else get_latest_run(tid)
    except Exception as e:
        return None, str(e)
    if not info:
        return None, "未找到可用的来源 run（请先运行来源工具）"
    return info, None


def _run_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _warehouse_path() -> str:
    base = os.path.abspath(get_base_dir())
    return os.path.abspath(os.path.join(base, "data", "warehouse.sqlite"))


def _read_warehouse_cleaned_df(source_run_id: str) -> Tuple[pd.DataFrame, Optional[str]]:
    bid = str(source_run_id or "").strip()
    if not bid:
        return pd.DataFrame(), "source_run_id 不能为空（用于从累计库读取）"
    sp = _warehouse_path()
    if not sp or not os.path.exists(sp):
        return pd.DataFrame(), "累计库不存在，请先运行来源工具清洗落库"
    try:
        with sqlite3.connect(sp) as conn:
            df = pd.read_sql_query("SELECT * FROM warehouse_cleaned WHERE batch_id = ?", conn, params=[bid])
        if df is None or df.empty:
            return pd.DataFrame(), f"累计库中未找到 batch_id={bid} 的清洗数据"
        try:
            df = df.drop(columns=[c for c in ["batch_id", "row_hash"] if c in df.columns])
        except Exception:
            pass
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def _ensure_warehouse_validation_schema(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_validation'")
        if cur.fetchone():
            cur2 = conn.execute("PRAGMA table_info(warehouse_validation)")
            cols = {str(r[1]) for r in (cur2.fetchall() or []) if r and len(r) > 1}
            if "期间" not in cols:
                legacy = f"warehouse_validation_legacy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                conn.execute(f'ALTER TABLE warehouse_validation RENAME TO "{legacy}"')
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warehouse_validation (
          tool_id TEXT NOT NULL,
          run_id TEXT NOT NULL,
          source_tool_id TEXT,
          source_run_id TEXT,
          源文件 TEXT,
          来源Sheet TEXT,
          期间 TEXT,
          验证项目 TEXT,
          时间属性 TEXT,
          差额 REAL,
          是否平衡 TEXT,
          验证结果 TEXT,
          PRIMARY KEY (tool_id, run_id, 源文件, 期间, 验证项目, 时间属性)
        )
        """
    )
    conn.execute('CREATE INDEX IF NOT EXISTS ix_wv_tool_run ON warehouse_validation(tool_id, run_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_wv_unbalanced ON warehouse_validation(是否平衡)')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_wv_source_file ON warehouse_validation(源文件)')
    conn.execute('CREATE INDEX IF NOT EXISTS ix_wv_period ON warehouse_validation(期间)')


def _warehouse_user_flags(conn: sqlite3.Connection) -> int:
    try:
        cur = conn.execute("PRAGMA user_version")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def _set_warehouse_user_flags(conn: sqlite3.Connection, v: int) -> None:
    try:
        conn.execute(f"PRAGMA user_version = {int(v)}")
    except Exception:
        pass


def _distinct_valid_periods(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    if "期间" not in df.columns:
        return []
    try:
        ser = df["期间"].dropna().astype(str).map(lambda x: str(x or "").strip())
        vals = ser.tolist()
    except Exception:
        vals = []
    seen: set = set()
    out: List[str] = []
    for p in vals:
        pp = str(p or "").strip()
        if not pp:
            continue
        if not re.match(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])$", pp):
            continue
        if pp not in seen:
            seen.add(pp)
            out.append(pp)
    out.sort()
    return out


def _delete_other_runs_for_validation_period(conn: sqlite3.Connection, tool_id: str, source_tool_id: str, period: str, keep_run_id: str) -> None:
    tid = str(tool_id or "").strip()
    stid = str(source_tool_id or "").strip()
    per = str(period or "").strip()
    rid = str(keep_run_id or "").strip()
    if not tid or not per or not rid:
        return
    if stid:
        conn.execute(
            "DELETE FROM warehouse_validation WHERE tool_id = ? AND source_tool_id = ? AND 期间 = ? AND run_id <> ?",
            (tid, stid, per, rid),
        )
    else:
        conn.execute(
            "DELETE FROM warehouse_validation WHERE tool_id = ? AND (source_tool_id IS NULL OR TRIM(source_tool_id) = '') AND 期间 = ? AND run_id <> ?",
            (tid, per, rid),
        )


def _dedup_existing_validation_by_period(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_validation'")
        if not cur.fetchone():
            return
    except Exception:
        return
    try:
        cur = conn.execute(
            """
            SELECT tool_id, COALESCE(source_tool_id, '') as source_tool_id, 期间, MAX(run_id) as keep_run_id
            FROM warehouse_validation
            WHERE TRIM(COALESCE(期间, '')) != ''
            GROUP BY tool_id, COALESCE(source_tool_id, ''), 期间
            """
        )
        groups = cur.fetchall() or []
    except Exception:
        groups = []
    for r in groups:
        try:
            tid = str(r[0] or "").strip()
            stid = str(r[1] or "").strip()
            per = str(r[2] or "").strip()
            keep = str(r[3] or "").strip()
        except Exception:
            continue
        if not tid or not per or not keep:
            continue
        try:
            _delete_other_runs_for_validation_period(conn, tid, stid, per, keep)
        except Exception:
            pass


def _replace_validation_periods(conn: sqlite3.Connection, tool_id: str, source_tool_id: str, periods: List[str]) -> None:
    tid = str(tool_id or "").strip()
    stid = str(source_tool_id or "").strip()
    if not tid:
        return
    for p in periods or []:
        per = str(p or "").strip()
        if not per:
            continue
        try:
            if stid:
                conn.execute(
                    "DELETE FROM warehouse_validation WHERE tool_id = ? AND source_tool_id = ? AND 期间 = ?",
                    (tid, stid, per),
                )
            else:
                conn.execute(
                    "DELETE FROM warehouse_validation WHERE tool_id = ? AND (source_tool_id IS NULL OR TRIM(source_tool_id) = '') AND 期间 = ?",
                    (tid, per),
                )
        except Exception:
            pass


def _write_validation_to_warehouse(
    tool_id: str,
    run_id: str,
    source_tool_id: str,
    source_run_id: str,
    df_validation: pd.DataFrame,
) -> Optional[str]:
    sp = _warehouse_path()
    if not sp:
        return "累计库路径为空"
    try:
        os.makedirs(os.path.dirname(sp) or os.getcwd(), exist_ok=True)
    except Exception:
        pass
    try:
        with sqlite3.connect(sp) as conn:
            _ensure_warehouse_validation_schema(conn)
            flags = _warehouse_user_flags(conn)
            if not (int(flags) & 4):
                _dedup_existing_validation_by_period(conn)
                _set_warehouse_user_flags(conn, int(flags) | 4)
            cols = [str(c) for c in (df_validation.columns.tolist() if df_validation is not None else [])]
            required = ["源文件", "来源Sheet", "期间", "验证项目", "时间属性", "差额", "是否平衡", "验证结果"]
            for c in required:
                if c not in cols:
                    df_validation[c] = ""
            _replace_validation_periods(conn, str(tool_id or ""), str(source_tool_id or ""), _distinct_valid_periods(df_validation))
            rows: List[Tuple[Any, ...]] = []
            for r in df_validation.itertuples(index=False):
                rec = r._asdict() if hasattr(r, "_asdict") else dict(zip(df_validation.columns, list(r)))
                rows.append(
                    (
                        str(tool_id),
                        str(run_id),
                        str(source_tool_id),
                        str(source_run_id),
                        rec.get("源文件"),
                        rec.get("来源Sheet"),
                        rec.get("期间"),
                        rec.get("验证项目"),
                        rec.get("时间属性"),
                        rec.get("差额"),
                        rec.get("是否平衡"),
                        rec.get("验证结果"),
                    )
                )
            conn.executemany(
                """
                INSERT OR REPLACE INTO warehouse_validation(
                  tool_id, run_id, source_tool_id, source_run_id,
                  源文件, 来源Sheet, 期间, 验证项目, 时间属性, 差额, 是否平衡, 验证结果
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return None
    except Exception as e:
        return str(e)



def extract_amount_info(df, source_tool_id: str, keywords, sheet_type=None, time_attr=None, category=None) -> Tuple[float, bool, str]:
    filtered_df = df.copy()
    if sheet_type:
        filtered_df = filtered_df[filtered_df["报表类型"] == sheet_type]
    if time_attr:
        filtered_df = filtered_df[filtered_df["时间属性"] == time_attr]
    if category:
        filtered_df = filtered_df[filtered_df["大类"] == category]
    if filtered_df.empty or "科目" not in filtered_df.columns or "金额" not in filtered_df.columns:
        return 0.0, False, ""
    subjects_norm = filtered_df["科目"].astype(str).map(_normalize_subject_text)
    expanded = _expand_keywords(source_tool_id, list(keywords or []))
    for keyword in expanded:
        kw_norm = _normalize_subject_text(keyword)
        if not kw_norm:
            continue
        exact_mask = subjects_norm == kw_norm
        if exact_mask.any():
            row = filtered_df.loc[exact_mask].iloc[0]
            return float(row["金额"]), True, str(row["科目"])
    for keyword in expanded:
        kw_norm = _normalize_subject_text(keyword)
        if not kw_norm:
            continue
        contain_mask = subjects_norm.str.contains(re.escape(kw_norm), na=False)
        if contain_mask.any():
            candidates = filtered_df.loc[contain_mask, ["科目", "金额"]].copy()
            candidates["__len"] = candidates["科目"].astype(str).map(lambda x: len(_normalize_subject_text(x)))
            candidates = candidates.sort_values("__len", ascending=True)
            row = candidates.iloc[0]
            return float(row["金额"]), True, str(row["科目"])
    return 0.0, False, ""


def validate_balance_sheet(df_group, source_tool_id: str, tolerance: float = 0.01, time_attr: str = "期末余额", assets_keywords=None, liabilities_keywords=None, equity_keywords=None):
    assets, af, _ = extract_amount_info(df_group, source_tool_id, assets_keywords or ["资产总计", "资产总额", "资产合计"], sheet_type="资产负债表", time_attr=time_attr, category="资产")
    liabilities, lf, _ = extract_amount_info(df_group, source_tool_id, liabilities_keywords or ["负债合计", "负债总计", "负债总额"], sheet_type="资产负债表", time_attr=time_attr, category="负债及权益")
    equity, ef, _ = extract_amount_info(df_group, source_tool_id, equity_keywords or ["所有者权益合计", "股东权益合计", "权益合计"], sheet_type="资产负债表", time_attr=time_attr, category="负债及权益")
    if not (af and lf and ef):
        return {"验证项目": "BS表资产=负债+权益验证", "时间属性": time_attr, "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs(assets - (liabilities + equity))
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表资产=负债+权益验证", "时间属性": time_attr, "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def validate_bs_pl_balance(df_group, source_tool_id: str, tolerance: float = 0.01) -> Dict[str, Any]:
    re_begin, rb_f, _ = extract_amount_info(df_group, source_tool_id, ["未分配利润"], sheet_type="资产负债表", time_attr="年初余额", category="负债及权益")
    re_end, re_f, _ = extract_amount_info(df_group, source_tool_id, ["未分配利润"], sheet_type="资产负债表", time_attr="期末余额", category="负债及权益")
    np, np_f, _ = extract_amount_info(df_group, source_tool_id, ["归属于母公司所有者的净利润", "归属于母公司股东的净利润"], sheet_type="利润表", time_attr="本年累计金额")
    if not (rb_f and re_f and np_f):
        return {"验证项目": "BS表与PL表平衡验证", "时间属性": "年初/期末 vs 本年累计", "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs((re_end - re_begin) - np)
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表与PL表平衡验证", "时间属性": "年初/期末 vs 本年累计", "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def validate_bs_cf_balance(df_group, source_tool_id: str, tolerance: float = 0.01) -> Dict[str, Any]:
    bank_end, bf, _ = extract_amount_info(df_group, source_tool_id, ["银行存款"], sheet_type="资产负债表", time_attr="期末余额", category="资产")
    cf_end, cf_f, _ = extract_amount_info(df_group, source_tool_id, ["期末现金及现金等价物余额", "期末现金及现金等价物余额(附注)"], sheet_type="现金流量表", time_attr="本年累计金额")
    if not (bf and cf_f):
        return {"验证项目": "BS表与CF表平衡验证", "时间属性": "期末余额 vs 本年累计", "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs(bank_end - cf_end)
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表与CF表平衡验证", "时间属性": "期末余额 vs 本年累计", "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def run_analysis(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    if logger is None:
        logger = _get_logger()

    tool_id = str(getattr(cfg, "tool_id", "") or _TOOL_ID).strip() or _TOOL_ID
    result.tool_id = tool_id
    result.run_id = _run_timestamp()

    source_tool_id = str(_get_param(cfg, "source_tool_id", "report_ingestor") or "report_ingestor").strip()
    source_run_id = str(_get_param(cfg, "source_run_id", "") or "").strip()
    tolerance = float(_get_param(cfg, "validation_tolerance", 0.01) or 0.01)

    info, err = _open_source_run(source_tool_id, source_run_id)
    if err:
        result.errors.append(err)
        return result

    cleaned_path = str(info.get("cleaned_path") or "").strip()
    result.cleaned_path = cleaned_path
    source_run_id = str(info.get("run_id") or source_run_id or "").strip()

    df: pd.DataFrame
    derr: Optional[str]
    df, derr = _read_warehouse_cleaned_df(source_run_id)
    if derr:
        result.errors.append(derr)
        return result
    if df is None or df.empty:
        return result

    if cancel_event and cancel_event.is_set():
        result.cancelled = True
        return result

    validation_results: List[Dict[str, Any]] = []
    if "源文件" not in df.columns or "期间" not in df.columns:
        result.errors.append("cleaned 表缺少分组列：源文件/期间")
        return result

    for group_keys, df_group in df.groupby(["源文件", "期间"], dropna=False):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            return result
        group_info = dict(zip(["源文件", "期间"], group_keys if isinstance(group_keys, tuple) else [group_keys]))
        has_bs = "资产负债表" in df_group.get("报表类型", pd.Series([], dtype="object")).values
        has_pl = "利润表" in df_group.get("报表类型", pd.Series([], dtype="object")).values
        has_cf = "现金流量表" in df_group.get("报表类型", pd.Series([], dtype="object")).values

        if has_bs:
            v = validate_balance_sheet(df_group, source_tool_id, tolerance=tolerance)
            v.update(group_info)
            validation_results.append(v)
        if has_bs and has_pl:
            v = validate_bs_pl_balance(df_group, source_tool_id, tolerance=tolerance)
            v.update(group_info)
            validation_results.append(v)
        if has_bs and has_cf:
            v = validate_bs_cf_balance(df_group, source_tool_id, tolerance=tolerance)
            v.update(group_info)
            validation_results.append(v)

    df_validation = pd.DataFrame(validation_results) if validation_results else pd.DataFrame()
    if df_validation.empty:
        return result

    validation_path = cleaned_path.replace(".xlsx", "_验证报告.xlsx") if cleaned_path.lower().endswith(".xlsx") else ""
    if validation_path:
        try:
            df_validation.to_excel(validation_path, index=False)
            result.validation_path = validation_path
        except Exception as e:
            result.errors.append(str(e))

    result.validation_groups = int(len(df_validation))
    result.validation_preview = _df_preview_records(df_validation, limit=2000)
    if "是否平衡" in df_validation.columns:
        unbalanced = df_validation[df_validation["是否平衡"] == "否"]
        result.unbalanced_count = int(len(unbalanced))
        if not unbalanced.empty:
            result.unbalanced_preview = _df_preview_records(unbalanced, limit=200)

    werr = _write_validation_to_warehouse(
        tool_id=str(result.tool_id or tool_id),
        run_id=str(result.run_id or ""),
        source_tool_id=source_tool_id,
        source_run_id=source_run_id,
        df_validation=df_validation,
    )
    if werr:
        result.errors.append(werr)

    result.artifacts = _build_artifacts_common(
        cleaned_path=str(result.cleaned_path or ""),
        cleaned_sqlite_path="",
        validation_path=str(result.validation_path or ""),
        metrics_path="",
    )
    return result
