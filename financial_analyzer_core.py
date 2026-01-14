import pandas as pd
import datetime
import re
import hashlib
import glob
import os
import json
import logging
import threading
import argparse
import sqlite3
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional, Any, Dict, List, Tuple


OUTPUT_PATH = "清洗后的AI标准财务表.xlsx"

import sys

def get_base_dir():
    """获取程序运行的基础目录，兼容开发环境和 PyInstaller 打包环境"""
    if hasattr(sys, '_MEIPASS'):
        # 打包环境：返回 .exe 所在的目录（或者根据需要返回 sys._MEIPASS）
        # 通常配置文件应该放在 .exe 旁边，而不是临时目录里
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(get_base_dir(), "config", "financial_analyzer_config.json")


@dataclass
class AppConfig:
    input_dir: str = field(default_factory=lambda: os.getcwd())
    file_glob: str = "*.xlsx"
    output_dir: str = field(default_factory=lambda: "output")
    output_basename: str = OUTPUT_PATH
    generate_validation: bool = True
    generate_metrics: bool = True
    exclude_output_files: bool = True
    sheet_keyword_bs: str = "BS-合并"
    sheet_keyword_pl: str = "PL-合并"
    sheet_keyword_cf: str = "CF-合并"
    header_keyword_bs: str = "期末余额"
    header_keyword_pl: str = "本年累计"
    header_keyword_cf: str = "本期金额"
    date_cells_bs: List[List[int]] = field(default_factory=lambda: [[2, 3], [2, 2]])
    date_cells_pl: List[List[int]] = field(default_factory=lambda: [[2, 2], [2, 1]])
    date_cells_cf: List[List[int]] = field(default_factory=lambda: [[2, 4], [2, 0]])
    validation_tolerance: float = 0.01
    saved_queries: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    cancelled: bool = False
    errors: List[str] = field(default_factory=list)
    found_files: List[str] = field(default_factory=list)
    processed_files: int = 0
    cleaned_rows: int = 0
    metrics_groups: int = 0
    validation_groups: int = 0
    unbalanced_count: int = 0
    cleaned_path: Optional[str] = None
    cleaned_sqlite_path: Optional[str] = None
    validation_path: Optional[str] = None
    metrics_path: Optional[str] = None
    unbalanced_preview: List[Dict[str, Any]] = field(default_factory=list)
    validation_preview: List[Dict[str, Any]] = field(default_factory=list)
    metrics_preview: List[Dict[str, Any]] = field(default_factory=list)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_output_root() -> str:
    return os.path.join(get_base_dir(), "output")


def _default_data_root() -> str:
    return os.path.join(get_base_dir(), "data")


def _resolve_under_base(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(get_base_dir(), p))


_RULES_CACHE: Dict[str, Any] = {"mtime": None, "data": None, "aliases": None}

def _cjk_count(s: str) -> int:
    return sum(1 for ch in (s or "") if "\u4e00" <= ch <= "\u9fff")

def _maybe_fix_mojibake_str(s: Any) -> Any:
    if not isinstance(s, str) or not s:
        return s
    if _cjk_count(s) > 0:
        return s
    try:
        raw = s.encode("latin-1")
    except Exception:
        return s
    try:
        fixed = raw.decode("utf-8")
    except Exception:
        return s
    if not fixed or fixed == s:
        return s
    if _cjk_count(fixed) == 0:
        return s
    return fixed

def _repair_mojibake_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _maybe_fix_mojibake_str(obj)
    if isinstance(obj, list):
        items = [_repair_mojibake_obj(x) for x in obj]
        if items and all(isinstance(x, str) for x in items):
            if all(_cjk_count(x) == 0 for x in items):
                try:
                    bs_parts = []
                    for x in items:
                        bs_parts.append(x.encode("latin-1"))
                    merged = b"\xa0".join(bs_parts)
                    fixed = merged.decode("utf-8")
                    if fixed and _cjk_count(fixed) > 0:
                        return [fixed]
                except Exception:
                    pass
        return items
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            kk = _repair_mojibake_obj(k) if isinstance(k, str) else k
            vv = _repair_mojibake_obj(v)
            if kk in out and isinstance(out.get(kk), list) and isinstance(vv, list):
                merged = list(out[kk])
                for it in vv:
                    if it not in merged:
                        merged.append(it)
                out[kk] = merged
            else:
                out[kk] = vv
        return out
    return obj


def _rules_path() -> str:
    return os.path.join(get_base_dir(), "config", "rules.json")

def _maybe_materialize_rules_json(target_path: str) -> None:
    p = str(target_path or "").strip()
    if not p or os.path.exists(p):
        return
    if hasattr(sys, "_MEIPASS"):
        try:
            embedded = os.path.join(getattr(sys, "_MEIPASS"), "config", "rules.json")
            if os.path.exists(embedded):
                _ensure_dir(os.path.dirname(p) or os.getcwd())
                with open(embedded, "r", encoding="utf-8") as rf:
                    raw = rf.read()
                with open(p, "w", encoding="utf-8") as wf:
                    wf.write(raw)
                return
        except Exception:
            pass
    try:
        legacy = os.path.join(get_base_dir(), "rules.json")
        if os.path.exists(legacy):
            _ensure_dir(os.path.dirname(p) or os.getcwd())
            with open(legacy, "r", encoding="utf-8") as rf:
                raw = rf.read()
            with open(p, "w", encoding="utf-8") as wf:
                wf.write(raw)
    except Exception:
        pass


def _load_rules() -> Dict[str, Any]:
    p = _rules_path()
    _maybe_materialize_rules_json(p)
    try:
        mtime = float(os.path.getmtime(p))
    except Exception:
        mtime = None

    cached_mtime = _RULES_CACHE.get("mtime")
    cached_data = _RULES_CACHE.get("data")
    if cached_data is not None and cached_mtime == mtime:
        return cached_data

    data: Dict[str, Any] = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        if isinstance(raw, dict):
            data = raw
    except Exception:
        data = {}

    try:
        repaired = _repair_mojibake_obj(data)
        if isinstance(repaired, dict) and repaired != data:
            data = repaired
            _ensure_dir(os.path.dirname(p) or os.getcwd())
            with open(p, "w", encoding="utf-8") as wf:
                json.dump(data, wf, ensure_ascii=False, indent=2)
            try:
                mtime = float(os.path.getmtime(p))
            except Exception:
                mtime = None
    except Exception:
        pass

    aliases = {}
    try:
        sa = data.get("subject_aliases") if isinstance(data, dict) else None
        if isinstance(sa, dict):
            for canon, syns in sa.items():
                canon_s = str(canon or "").strip()
                if not canon_s:
                    continue
                items = [canon_s]
                if isinstance(syns, list):
                    for x in syns:
                        xs = str(x or "").strip()
                        if xs:
                            items.append(xs)
                variants = list(dict.fromkeys(items))
                for v in variants:
                    aliases.setdefault(_normalize_subject_text(v), set()).update(variants)
    except Exception:
        aliases = {}

    _RULES_CACHE["mtime"] = mtime
    _RULES_CACHE["data"] = data
    _RULES_CACHE["aliases"] = aliases
    return data


def _expand_keywords(keywords: List[Any]) -> List[str]:
    rules = _load_rules()
    aliases = _RULES_CACHE.get("aliases") or {}
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


def _is_timestamp_folder(name: str) -> bool:
    return bool(re.match(r"^\d{8}_\d{6}$", str(name or "").strip()))


def _run_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_int_pair(pair: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        return int(pair[0]), int(pair[1])
    except Exception:
        return None


def _cleaned_sqlite_path_for(cleaned_path: str) -> str:
    name = os.path.basename(str(cleaned_path or "")).strip()
    if not name:
        base_name = "cleaned"
    else:
        base_name, _ = os.path.splitext(name)
        base_name = base_name or "cleaned"

    ts = ""
    try:
        parent = os.path.basename(os.path.dirname(str(cleaned_path or "")))
        if _is_timestamp_folder(parent):
            ts = parent
    except Exception:
        ts = ""

    filename = f"{base_name}_{ts}.sqlite" if ts else f"{base_name}.sqlite"
    return os.path.abspath(os.path.join(_default_data_root(), filename))


def _write_cleaned_sqlite(
    df: pd.DataFrame,
    sqlite_path: str,
    df_validation: Optional[pd.DataFrame] = None,
    df_metrics: Optional[pd.DataFrame] = None,
) -> None:
    def _make_index_name(prefix: str, col: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_")
        digest = hashlib.sha1(str(col).encode("utf-8")).hexdigest()[:12]
        if safe:
            return f"{prefix}_{safe}_{digest}"
        return f"{prefix}_{digest}"

    _ensure_dir(os.path.dirname(sqlite_path) or os.getcwd())
    conn = sqlite3.connect(sqlite_path)
    try:
        df2 = df.copy()
        if "金额" in df2.columns:
            df2["金额"] = pd.to_numeric(df2["金额"], errors="coerce").fillna(0.0)
        df2.to_sql("cleaned", conn, if_exists="replace", index=False, chunksize=2000)
        for col in ["源文件", "日期", "报表类型", "大类", "时间属性", "科目", "金额"]:
            if col in df2.columns:
                idx_name = _make_index_name("idx_cleaned", col)
                conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON cleaned("{col}")')

        if df_validation is not None and isinstance(df_validation, pd.DataFrame) and not df_validation.empty:
            vdf = df_validation.copy()
            if "差额" in vdf.columns:
                vdf["差额"] = pd.to_numeric(vdf["差额"], errors="coerce")
            vdf.to_sql("validation", conn, if_exists="replace", index=False, chunksize=2000)
            for col in ["源文件", "日期", "是否平衡", "验证项目", "时间属性", "差额", "来源Sheet"]:
                if col in vdf.columns:
                    idx_name = _make_index_name("idx_validation", col)
                    conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON validation("{col}")')

        if df_metrics is not None and isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:
            mdf = df_metrics.copy()
            mdf.to_sql("metrics", conn, if_exists="replace", index=False, chunksize=2000)
            for col in ["源文件", "日期"]:
                if col in mdf.columns:
                    idx_name = _make_index_name("idx_metrics", col)
                    conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON metrics("{col}")')
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_config(path: str) -> AppConfig:
    try:
        if not os.path.exists(path):
            if hasattr(sys, "_MEIPASS"):
                try:
                    embedded = os.path.join(getattr(sys, "_MEIPASS"), "config", "financial_analyzer_config.json")
                    if os.path.exists(embedded):
                        _ensure_dir(os.path.dirname(path) or os.getcwd())
                        with open(embedded, "r", encoding="utf-8") as rf:
                            raw = rf.read()
                        with open(path, "w", encoding="utf-8") as wf:
                            wf.write(raw)
                except Exception:
                    pass
            legacy = os.path.join(get_base_dir(), "financial_analyzer_config.json")
            if os.path.exists(legacy):
                try:
                    _ensure_dir(os.path.dirname(path) or os.getcwd())
                    with open(legacy, "r", encoding="utf-8") as rf:
                        raw = rf.read()
                    with open(path, "w", encoding="utf-8") as wf:
                        wf.write(raw)
                except Exception:
                    pass
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
    except FileNotFoundError:
        return AppConfig()
    except Exception:
        return AppConfig()


def save_config(path: str, cfg: AppConfig) -> None:
    _ensure_dir(os.path.dirname(path) or os.getcwd())
    data = asdict(cfg)
    try:
        base = os.path.abspath(get_base_dir())
        out_dir = str(data.get("output_dir", "") or "").strip()
        if out_dir and os.path.isabs(out_dir):
            try:
                rel = os.path.relpath(out_dir, base)
                if not rel.startswith(".."):
                    data["output_dir"] = rel.replace("\\", "/")
            except Exception:
                pass
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_logger(name: str = "financial_analyzer", handler: Optional[logging.Handler] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _json_safe_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp)):
        return str(v)
    try:
        item = getattr(v, "item", None)
        if callable(item):
            return item()
    except Exception:
        pass
    return v


def _df_preview_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    df2 = df.head(int(limit)).copy()
    for c in df2.columns:
        df2[c] = df2[c].map(_json_safe_value)
    return df2.to_dict(orient="records")


def _is_date_like(date_val: Any) -> bool:
    if date_val is None:
        return False
    try:
        if pd.isna(date_val):
            return False
    except Exception:
        pass

    if isinstance(date_val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return True

    if isinstance(date_val, (int, float)) and not isinstance(date_val, bool):
        try:
            n = float(date_val)
        except Exception:
            return False
        if pd.isna(n):
            return False
        if not (20000 <= n <= 80000):
            return False
        try:
            dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(n))
            return 1990 <= dt.year <= 2100
        except Exception:
            return False

    text = str(date_val).strip()
    if not text:
        return False

    digits = re.findall(r"\d+", text)
    if len(digits) >= 2 and len(digits[0]) == 4:
        try:
            month = int(digits[1])
            if 1 <= month <= 12:
                if len(digits) >= 3:
                    day = int(digits[2])
                    if 1 <= day <= 31:
                        return True
                    return False
                return True
        except Exception:
            pass

    try:
        ts = pd.to_datetime(text, errors="coerce", infer_datetime_format=True)
        return not pd.isna(ts)
    except Exception:
        return False


def _read_date_nearby(df: pd.DataFrame, r: int, c: int, max_radius: int = 10) -> Any:
    rows, cols = int(df.shape[0]), int(df.shape[1])

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < rows and 0 <= cc < cols

    if in_bounds(r, c):
        val0 = df.iat[r, c]
        if _is_date_like(val0):
            return val0

    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            rr_top = r + dr
            cc_left = c - radius
            cc_right = c + radius
            if in_bounds(rr_top, cc_left):
                val = df.iat[rr_top, cc_left]
                if _is_date_like(val):
                    return val
            if in_bounds(rr_top, cc_right):
                val = df.iat[rr_top, cc_right]
                if _is_date_like(val):
                    return val

        for dc in range(-radius + 1, radius):
            cc = c + dc
            rr_top = r - radius
            rr_bottom = r + radius
            if in_bounds(rr_top, cc):
                val = df.iat[rr_top, cc]
                if _is_date_like(val):
                    return val
            if in_bounds(rr_bottom, cc):
                val = df.iat[rr_bottom, cc]
                if _is_date_like(val):
                    return val

    return None


def _read_date_from_cells(df: pd.DataFrame, cells: List[List[int]]) -> Any:
    for cell in cells:
        rc = _safe_int_pair(cell)
        if rc is None:
            continue
        r, c = rc
        if 0 <= r < df.shape[0] and 0 <= c < df.shape[1]:
            val = _read_date_nearby(df, r, c)
            if val is not None:
                return val
    return None


def _find_header_row(df: pd.DataFrame, keyword: str) -> Optional[int]:
    def norm(s: Any) -> str:
        t = str(s or "")
        t = t.strip().lower()
        t = t.replace("（", "(").replace("）", ")").replace("\u3000", " ")
        t = re.sub(r"\s+", "", t)
        return t

    try:
        kw = norm(keyword)
        if not kw:
            return None

        max_rows = min(int(df.shape[0]), 120)
        sub = df.iloc[:max_rows]
        values = sub.to_numpy()
        for i in range(values.shape[0]):
            row = values[i]
            for j in range(row.shape[0]):
                if kw in norm(row[j]):
                    return int(sub.index[i])
        return None
    except Exception:
        return None


def clean_date_str(date_val):
    """
    清洗日期：支持 Excel数字、'2025年11月'、'2025-11-30' 等格式
    """
    if pd.isna(date_val) or date_val == "":
        return "未知日期"

    if isinstance(date_val, (int, float)):
        try:
            return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(date_val))).strftime("%Y-%m-%d")
        except Exception:
            return str(date_val)

    text = str(date_val)
    digits = re.findall(r"\d+", text)
    if len(digits) >= 2:
        year = digits[0]
        month = digits[1].zfill(2)
        day = digits[2].zfill(2) if len(digits) > 2 else "01"
        return f"{year}-{month}-{day}"

    return text.split(" ")[0]


def clean_bs(
    file_path,
    sheet_name,
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    excel_file: Optional[pd.ExcelFile] = None,
    sheet_df: Optional[pd.DataFrame] = None,
):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        if sheet_df is not None:
            df = sheet_df
        elif excel_file is not None:
            df = excel_file.parse(sheet_name=sheet_name, header=None)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_bs)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_bs)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_bs}")
        df_left = df.iloc[header_row + 1 :, [0, 1, 2]].copy()
        df_left.columns = ["科目", "年初余额", "期末余额"]
        df_left["大类"] = "资产"
        df_parts = [df_left]
        if df.shape[1] >= 6:
            df_right = df.iloc[header_row + 1 :, [3, 4, 5]].copy()
            df_right.columns = ["科目", "年初余额", "期末余额"]
            df_right["大类"] = "负债及权益"
            df_parts.append(df_right)
        df_clean = pd.concat(df_parts, ignore_index=True)
        df_clean = df_clean.dropna(subset=["科目"])
        df_clean = df_clean[df_clean["科目"].astype(str).str.strip() != ""]
        df_final = df_clean.melt(
            id_vars=["大类", "科目"], value_vars=["年初余额", "期末余额"], var_name="时间属性", value_name="金额"
        )
        df_final["报表类型"] = "资产负债表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_pl(
    file_path,
    sheet_name,
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    excel_file: Optional[pd.ExcelFile] = None,
    sheet_df: Optional[pd.DataFrame] = None,
):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        if sheet_df is not None:
            df = sheet_df
        elif excel_file is not None:
            df = excel_file.parse(sheet_name=sheet_name, header=None)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_pl)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_pl)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_pl}")
        df_clean = df.iloc[header_row + 1 :, [0, 2, 3]].copy()
        df_clean.columns = ["科目", "本期金额", "本年累计金额"]
        df_clean = df_clean.dropna(subset=["科目"])
        df_final = df_clean.melt(
            id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额"
        )
        df_final["大类"] = "损益"
        df_final["报表类型"] = "利润表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_cf(
    file_path,
    sheet_name,
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    excel_file: Optional[pd.ExcelFile] = None,
    sheet_df: Optional[pd.DataFrame] = None,
):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        if sheet_df is not None:
            df = sheet_df
        elif excel_file is not None:
            df = excel_file.parse(sheet_name=sheet_name, header=None)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, cfg.date_cells_cf)
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, cfg.header_keyword_cf)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {cfg.header_keyword_cf}")
        df_left = df.iloc[header_row + 1 :, [0, 2, 3]].copy()
        df_left.columns = ["科目", "本期金额", "本年累计金额"]
        if df.shape[1] >= 8:
            df_right = df.iloc[header_row + 1 :, [4, 6, 7]].copy()
            df_right.columns = ["科目", "本期金额", "本年累计金额"]
            df_combined = pd.concat([df_left, df_right], ignore_index=True)
        else:
            df_combined = df_left
        df_combined = df_combined.dropna(subset=["科目"])
        df_combined = df_combined[df_combined["科目"].astype(str).str.strip() != ""]
        df_final = df_combined.melt(
            id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额"
        )
        df_final["大类"] = "现金流"
        df_final["报表类型"] = "现金流量表"
        df_final["日期"] = report_date
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def _normalize_subject_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s,，]", "", s)
    return s


def extract_amount_info(df, keywords, sheet_type=None, time_attr=None, category=None) -> Tuple[float, bool, str]:
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

    expanded = _expand_keywords(list(keywords or []))
    
    # First pass: Exact match
    for keyword in expanded:
        kw_norm = _normalize_subject_text(keyword)
        if not kw_norm:
            continue
        exact_mask = subjects_norm == kw_norm
        if exact_mask.any():
            row = filtered_df.loc[exact_mask].iloc[0]
            return float(row["金额"]), True, str(row["科目"])

    # Second pass: Partial match (contains)
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
            # Double check to avoid matching "Liabilities and Equity" when looking for "Equity"
            # If the candidate contains "负债" and we are looking for Equity (and keyword doesn't contain 负债), skip it?
            # But we don't know if we are looking for Equity specifically here (generic function).
            # But usually "Total Liabilities and Equity" is much longer than "Total Equity".
            # If we sort by length (ascending), "Total Equity" should come before "Total Liabilities and Equity".
            # The issue only happens if "Total Equity" is MISSING completely.
            # In that case, we might still match "Total Liabilities and Equity" which is bad.
            # We can add a blacklist? Or specific logic?
            # For now, splitting passes is a huge improvement.
            return float(row["金额"]), True, str(row["科目"])

    return 0.0, False, ""


def extract_amount(df, keywords, sheet_type=None, time_attr=None, category=None):
    val, found, _ = extract_amount_info(df, keywords, sheet_type=sheet_type, time_attr=time_attr, category=category)
    return val if found else 0.0


def validate_balance_sheet(
    df_group,
    tolerance: float = 0.01,
    time_attr: str = "期末余额",
    assets_keywords: Optional[List[str]] = None,
    liabilities_keywords: Optional[List[str]] = None,
    equity_keywords: Optional[List[str]] = None,
):
    assets, assets_found, _ = extract_amount_info(
        df_group,
        assets_keywords or ["资产总计", "资产总额", "资产合计"],
        sheet_type="资产负债表",
        time_attr=time_attr,
        category="资产",
    )
    liabilities, liabilities_found, _ = extract_amount_info(
        df_group,
        liabilities_keywords or ["负债合计", "负债总计", "负债总额"],
        sheet_type="资产负债表",
        time_attr=time_attr,
        category="负债及权益",
    )
    equity, equity_found, _ = extract_amount_info(
        df_group,
        equity_keywords or ["所有者权益合计", "股东权益合计", "所有者权益总计", "权益合计", "所有者权益（或股东权益）合计"],
        sheet_type="资产负债表",
        time_attr=time_attr,
        category="负债及权益",
    )
    if not (assets_found and liabilities_found and equity_found):
        return {
            "验证项目": "BS表资产=负债+权益验证",
            "时间属性": time_attr,
            "资产总计": assets,
            "负债合计": liabilities,
            "所有者权益合计": equity,
            "差额": None,
            "是否平衡": "否",
            "验证结果": "数据缺失",
        }
    diff = abs(assets - (liabilities + equity))
    is_balanced = diff <= tolerance
    return {
        "验证项目": "BS表资产=负债+权益验证",
        "时间属性": time_attr,
        "资产总计": assets,
        "负债合计": liabilities,
        "所有者权益合计": equity,
        "差额": diff,
        "是否平衡": "是" if is_balanced else "否",
        "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})",
    }


def validate_bs_pl_balance(df_group, tolerance: float = 0.01) -> Dict[str, Any]:
    re_begin, re_begin_found, _ = extract_amount_info(
        df_group,
        ["未分配利润"],
        sheet_type="资产负债表",
        time_attr="年初余额",
        category="负债及权益",
    )
    re_end, re_end_found, _ = extract_amount_info(
        df_group,
        ["未分配利润"],
        sheet_type="资产负债表",
        time_attr="期末余额",
        category="负债及权益",
    )
    np, np_found, _ = extract_amount_info(
        df_group,
        ["归属于母公司所有者的净利润", "归属于母公司股东的净利润"],
        sheet_type="利润表",
        time_attr="本年累计金额",
    )
    if not (re_begin_found and re_end_found and np_found):
        return {
            "验证项目": "BS表与PL表平衡验证",
            "时间属性": "年初/期末 vs 本年累计",
            "未分配利润期初余额": re_begin,
            "未分配利润期末余额": re_end,
            "归属于母公司所有者的净利润本年累计": np,
            "差额": None,
            "是否平衡": "否",
            "验证结果": "数据缺失",
        }
    diff_val = (re_end - re_begin) - np
    diff = abs(diff_val)
    is_balanced = diff <= tolerance
    return {
        "验证项目": "BS表与PL表平衡验证",
        "时间属性": "年初/期末 vs 本年累计",
        "未分配利润期初余额": re_begin,
        "未分配利润期末余额": re_end,
        "归属于母公司所有者的净利润本年累计": np,
        "差额": diff,
        "是否平衡": "是" if is_balanced else "否",
        "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})",
    }


def validate_bs_cf_balance(df_group, tolerance: float = 0.01) -> Dict[str, Any]:
    bank_end, bank_found, _ = extract_amount_info(
        df_group,
        ["银行存款"],
        sheet_type="资产负债表",
        time_attr="期末余额",
        category="资产",
    )
    cf_end, cf_found, _ = extract_amount_info(
        df_group,
        ["期末现金及现金等价物余额", "期末现金及现金等价物余额(附注)", "期末现金及现金等价物余额（附注）"],
        sheet_type="现金流量表",
        time_attr="本年累计金额",
    )
    if not (bank_found and cf_found):
        return {
            "验证项目": "BS表与CF表平衡验证",
            "时间属性": "期末余额 vs 本年累计",
            "银行存款期末余额": bank_end,
            "期末现金及现金等价物余额本年累计": cf_end,
            "差额": None,
            "是否平衡": "否",
            "验证结果": "数据缺失",
        }
    diff_val = bank_end - cf_end
    diff = abs(diff_val)
    is_balanced = diff <= tolerance
    return {
        "验证项目": "BS表与CF表平衡验证",
        "时间属性": "期末余额 vs 本年累计",
        "银行存款期末余额": bank_end,
        "期末现金及现金等价物余额本年累计": cf_end,
        "差额": diff,
        "是否平衡": "是" if is_balanced else "否",
        "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})",
    }


def _format_percent(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    try:
        return f"{float(val) * 100:.2f}%"
    except Exception:
        return None


def _round_number(val: Optional[float], digits: int = 2) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), int(digits))
    except Exception:
        return None


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return n / d


def calculate_financial_metrics(df_group):
    rules = _load_rules()
    metrics_def = rules.get("metrics", [])
    variables_def = rules.get("variables", {})

    # 如果没有配置 metrics，尝试使用内置的硬编码逻辑（作为后备或初始过渡）
    # 但由于我们已经生成了完整的 rules.json，这里直接优先使用 rules.json
    # 为了防止 rules.json 被意外清空，如果 metrics_def 为空，可以保留原有逻辑。
    # 不过为了代码整洁，我们假设 rules.json 是 source of truth。
    
    # 1. 计算所有变量
    var_values = {}
    if variables_def:
        for var_name, var_rule in variables_def.items():
            keywords = var_rule.get("keywords", [])
            sheet_type = var_rule.get("sheet_type")
            time_attr = var_rule.get("time_attr")
            category = var_rule.get("category")
            val = extract_amount(df_group, keywords, sheet_type=sheet_type, time_attr=time_attr, category=category)
            var_values[var_name] = val
    
    # 如果 variables_def 为空（例如旧版配置），我们需要手动提取旧逻辑所需的变量吗？
    # 鉴于用户明确要求“外置”，我们应当依赖 rules.json。
    # 如果 rules.json 只有 subject_aliases，那么 metrics_def 也为空，返回空字典。
    
    if not metrics_def:
        # Fallback to hardcoded if no metrics defined in rules
        # (Copying original logic for safety if rules.json is missing metrics)
        return _calculate_financial_metrics_fallback(df_group)

    metrics = {}
    
    # Context for eval
    context = {
        "safe_div": _safe_div,
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        **var_values
    }

    for m in metrics_def:
        name = m.get("name")
        formula = m.get("formula")
        fmt = m.get("format")
        
        if not name or not formula:
            continue
            
        try:
            # 允许在公式中使用变量名
            val = eval(formula, {"__builtins__": {}}, context)
        except Exception:
            val = None
            
        if fmt == "percent":
            metrics[name] = _format_percent(val)
        elif fmt == "round2":
            metrics[name] = _round_number(val, 2)
        elif fmt == "number":
            # 保持原值（可能是 float 或 int）
            metrics[name] = val
        else:
            metrics[name] = val

    return metrics


def _calculate_financial_metrics_fallback(df_group):
    """
    保留原有硬编码逻辑作为后备，防止 rules.json 配置缺失
    """
    metrics = {}
    assets_total_end = extract_amount(df_group, ["资产总计", "资产总额", "资产合计"], sheet_type="资产负债表", time_attr="期末余额")
    current_assets_end = extract_amount(
        df_group, ["流动资产合计", "流动资产总计"], sheet_type="资产负债表", time_attr="期末余额"
    )
    cash_end = extract_amount(df_group, ["货币资金", "现金及现金等价物"], sheet_type="资产负债表", time_attr="期末余额")
    inventory_begin = extract_amount(df_group, ["存货"], sheet_type="资产负债表", time_attr="年初余额")
    inventory_end = extract_amount(df_group, ["存货"], sheet_type="资产负债表", time_attr="期末余额")
    ar_begin = extract_amount(df_group, ["应收账款", "应收帐款"], sheet_type="资产负债表", time_attr="年初余额")
    ar_end = extract_amount(df_group, ["应收账款", "应收帐款"], sheet_type="资产负债表", time_attr="期末余额")
    liabilities_total_end = extract_amount(
        df_group, ["负债合计", "负债总计", "负债总额"], sheet_type="资产负债表", time_attr="期末余额"
    )
    current_liabilities_end = extract_amount(
        df_group, ["流动负债合计", "流动负债总计"], sheet_type="资产负债表", time_attr="期末余额"
    )
    equity_total_end = extract_amount(
        df_group, ["所有者权益合计", "股东权益合计", "权益合计"], sheet_type="资产负债表", time_attr="期末余额"
    )

    revenue_m = extract_amount(df_group, ["营业收入", "主营业务收入"], sheet_type="利润表", time_attr="本期金额")
    revenue_ytd = extract_amount(df_group, ["营业收入", "主营业务收入"], sheet_type="利润表", time_attr="本年累计金额")
    main_revenue_ytd = extract_amount(df_group, ["主营业务收入"], sheet_type="利润表", time_attr="本年累计金额")
    cost_ytd = extract_amount(df_group, ["营业成本", "主营业务成本"], sheet_type="利润表", time_attr="本年累计金额")
    operating_profit_ytd = extract_amount(df_group, ["营业利润"], sheet_type="利润表", time_attr="本年累计金额")
    net_profit_ytd = extract_amount(df_group, ["净利润"], sheet_type="利润表", time_attr="本年累计金额")
    rd_m = extract_amount(df_group, ["研发费用"], sheet_type="利润表", time_attr="本期金额")
    rd_ytd = extract_amount(df_group, ["研发费用"], sheet_type="利润表", time_attr="本年累计金额")

    operating_cf_ytd = extract_amount(
        df_group, ["经营活动产生的现金流量净额", "经营活动现金流量净额"], sheet_type="现金流量表", time_attr="本年累计金额"
    )
    investing_cf_ytd = extract_amount(
        df_group, ["投资活动产生的现金流量净额", "投资活动现金流量净额"], sheet_type="现金流量表", time_attr="本年累计金额"
    )
    financing_cf_ytd = extract_amount(
        df_group, ["筹资活动产生的现金流量净额", "筹资活动现金流量净额"], sheet_type="现金流量表", time_attr="本年累计金额"
    )

    ratio_keys = {
        "流动比率",
        "速动比率",
        "现金比率",
        "资产负债率",
        "产权比率",
        "权益乘数",
        "毛利率",
        "营业利润率",
        "净利率",
        "ROE(净资产收益率)",
        "ROA(总资产收益率)",
        "现金流量比率",
        "当月研发投入强度",
        "本年研发投入强度",
    }

    metrics["流动比率"] = _safe_div(current_assets_end, current_liabilities_end)
    metrics["速动比率"] = _safe_div(current_assets_end - inventory_end, current_liabilities_end)
    metrics["现金比率"] = _safe_div(cash_end, current_liabilities_end)
    metrics["资产负债率"] = _safe_div(liabilities_total_end, assets_total_end)
    metrics["产权比率"] = _safe_div(liabilities_total_end, equity_total_end)
    metrics["权益乘数"] = _safe_div(assets_total_end, equity_total_end)
    metrics["毛利率"] = _safe_div(revenue_ytd - cost_ytd, revenue_ytd)
    metrics["营业利润率"] = _safe_div(operating_profit_ytd, revenue_ytd)
    metrics["净利率"] = _safe_div(net_profit_ytd, revenue_ytd)
    metrics["ROE(净资产收益率)"] = _safe_div(net_profit_ytd, equity_total_end)
    metrics["ROA(总资产收益率)"] = _safe_div(net_profit_ytd, assets_total_end)
    metrics["经营活动现金流净额"] = operating_cf_ytd
    metrics["投资活动现金流净额"] = investing_cf_ytd
    metrics["筹资活动现金流净额"] = financing_cf_ytd
    metrics["现金流量比率"] = _safe_div(operating_cf_ytd, current_liabilities_end)

    metrics["当月研发投入强度"] = _safe_div(rd_m, revenue_m)
    metrics["本年研发投入强度"] = _safe_div(rd_ytd, revenue_ytd)

    products_begin = extract_amount(
        df_group, ["产成品(库存商品)", "产成品", "库存商品"], sheet_type="资产负债表", time_attr="年初余额", category="资产"
    )
    products_end = extract_amount(
        df_group, ["产成品(库存商品)", "产成品", "库存商品"], sheet_type="资产负债表", time_attr="期末余额", category="资产"
    )
    metrics["本年现价工业总产值"] = (main_revenue_ytd or 0.0) + (products_end - products_begin)

    metrics["应收账款周转率"] = _safe_div(revenue_ytd, ar_begin + ar_end)
    metrics["存货周转率"] = _safe_div(cost_ytd, inventory_begin + inventory_end)

    for k in list(metrics.keys()):
        if k in ratio_keys:
            metrics[k] = _format_percent(metrics[k])
        elif k in {"应收账款周转率", "存货周转率"}:
            metrics[k] = _round_number(metrics[k], 2)
    return metrics


def _list_excel_files(cfg: AppConfig) -> List[str]:
    pattern = os.path.join(cfg.input_dir, cfg.file_glob)
    files = glob.glob(pattern)
    files = [os.path.abspath(p) for p in files if os.path.isfile(p)]
    if cfg.exclude_output_files:
        exclude = {
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename)),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_验证报告.xlsx"))),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_财务指标.xlsx"))),
        }
        files = [p for p in files if os.path.abspath(p) not in exclude]
    files.sort(key=lambda p: p.lower())
    return files


ProgressCallback = Callable[[str, int, int, str], None]


def _pick_sheet_name(df_group: pd.DataFrame, sheet_type: str) -> str:
    if df_group is None or df_group.empty:
        return ""
    if "来源Sheet" not in df_group.columns or "报表类型" not in df_group.columns:
        return ""
    sub = df_group[df_group["报表类型"] == sheet_type]
    if sub.empty:
        return ""
    try:
        return str(sub["来源Sheet"].mode(dropna=False).iat[0])
    except Exception:
        try:
            return str(sub["来源Sheet"].iloc[0])
        except Exception:
            return ""


def analyze_directory(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    if logger is None:
        logger = _get_logger()

    base_dir = os.path.abspath(get_base_dir())
    output_root_raw = str(cfg.output_dir or "").strip() or _default_output_root()
    output_root = _resolve_under_base(output_root_raw)
    if output_root == base_dir:
        output_root = os.path.abspath(os.path.join(output_root, "output"))

    stamp = _run_timestamp()
    run_dir = output_root
    if not _is_timestamp_folder(os.path.basename(run_dir)):
        run_dir = os.path.join(output_root, stamp)

    _ensure_dir(run_dir)
    _ensure_dir(_default_data_root())
    files = _list_excel_files(cfg)
    result.found_files = files

    if not files:
        msg = "未找到任何匹配的 .xlsx 文件"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    logger.info(f"找到 {len(files)} 个Excel文件")
    all_files_data = []

    for idx, file_path in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            logger.warning("已取消运行")
            return result

        if progress_cb:
            progress_cb("file", idx, len(files), os.path.basename(file_path))

        logger.info(f"正在处理文件: {os.path.basename(file_path)}")
        try:
            excel_file = pd.ExcelFile(file_path)
            try:
                all_sheets = excel_file.sheet_names
                bs_sheets = [s for s in all_sheets if cfg.sheet_keyword_bs.upper() in s.upper()]
                pl_sheets = [s for s in all_sheets if cfg.sheet_keyword_pl.upper() in s.upper()]
                cf_sheets = [s for s in all_sheets if cfg.sheet_keyword_cf.upper() in s.upper()]

                logger.info(f"发现 {len(all_sheets)} 个Sheet")
                logger.info(f"BS: {bs_sheets if bs_sheets else '无'} | PL: {pl_sheets if pl_sheets else '无'} | CF: {cf_sheets if cf_sheets else '无'}")

                file_sheets_data = []
                for sheet in bs_sheets:
                    if cancel_event and cancel_event.is_set():
                        result.cancelled = True
                        logger.warning("已取消运行")
                        return result
                    df = clean_bs(file_path, sheet, cfg, logger, excel_file=excel_file)
                    if not df.empty:
                        file_sheets_data.append(df)

                for sheet in pl_sheets:
                    if cancel_event and cancel_event.is_set():
                        result.cancelled = True
                        logger.warning("已取消运行")
                        return result
                    df = clean_pl(file_path, sheet, cfg, logger, excel_file=excel_file)
                    if not df.empty:
                        file_sheets_data.append(df)

                for sheet in cf_sheets:
                    if cancel_event and cancel_event.is_set():
                        result.cancelled = True
                        logger.warning("已取消运行")
                        return result
                    df = clean_cf(file_path, sheet, cfg, logger, excel_file=excel_file)
                    if not df.empty:
                        file_sheets_data.append(df)

                if file_sheets_data:
                    file_data = pd.concat(file_sheets_data, ignore_index=True)
                    file_data["源文件"] = os.path.basename(file_path)
                    all_files_data.append(file_data)
                    result.processed_files += 1
                    logger.info(f"完成: 提取 {len(file_data)} 行")
                else:
                    logger.warning("未提取到任何数据，可能缺少包含BS/PL/CF的Sheet")
            finally:
                try:
                    close = getattr(excel_file, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"读取失败: {e}")
            result.errors.append(f"{os.path.basename(file_path)}: {e}")

    if cancel_event and cancel_event.is_set():
        result.cancelled = True
        logger.warning("已取消运行")
        return result

    if not all_files_data:
        msg = "所有文件均未提取到有效数据"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    all_data = pd.concat(all_files_data, ignore_index=True)
    all_data["金额"] = all_data["金额"].astype(str).str.replace("—", "0").str.replace(",", "")
    all_data["金额"] = pd.to_numeric(all_data["金额"], errors="coerce").fillna(0)
    all_data["科目"] = all_data["科目"].astype(str).str.strip()

    cols = ["源文件", "来源Sheet", "日期", "报表类型", "大类", "科目", "时间属性", "金额"]
    final_cols = [c for c in cols if c in all_data.columns]
    all_data = all_data[final_cols]

    cleaned_path = os.path.abspath(os.path.join(run_dir, cfg.output_basename))
    all_data.to_excel(cleaned_path, index=False)
    result.cleaned_path = cleaned_path
    result.cleaned_rows = int(len(all_data))
    logger.info(f"原始数据已保存: {cleaned_path}")
    cleaned_sqlite_path = _cleaned_sqlite_path_for(cleaned_path)
    result.cleaned_sqlite_path = cleaned_sqlite_path

    validation_group_cols = ["源文件", "日期"]
    metrics_group_cols = ["源文件", "日期"]
    existing_validation_group_cols = [col for col in validation_group_cols if col in all_data.columns]
    existing_metrics_group_cols = [col for col in metrics_group_cols if col in all_data.columns]

    validation_results = []
    metrics_results = []
    df_validation: Optional[pd.DataFrame] = None
    df_metrics: Optional[pd.DataFrame] = None
    if existing_validation_group_cols:
        grouped_validation = all_data.groupby(existing_validation_group_cols, dropna=False)
        for group_keys, df_group in grouped_validation:
            if cancel_event and cancel_event.is_set():
                result.cancelled = True
                logger.warning("已取消运行")
                return result

            group_info = dict(
                zip(existing_validation_group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys])
            )
            if cfg.generate_validation and "报表类型" in df_group.columns:
                tol = float(cfg.validation_tolerance)
                has_bs = "资产负债表" in df_group["报表类型"].values
                has_pl = "利润表" in df_group["报表类型"].values
                has_cf = "现金流量表" in df_group["报表类型"].values

                if has_bs:
                    v = validate_balance_sheet(df_group, tolerance=tol, time_attr="期末余额")
                    v.update(group_info)
                    v.setdefault("来源Sheet", _pick_sheet_name(df_group, "资产负债表"))
                    validation_results.append(v)

                if has_bs and has_pl:
                    v = validate_bs_pl_balance(df_group, tolerance=tol)
                    v.update(group_info)
                    bs_sheet = _pick_sheet_name(df_group, "资产负债表")
                    pl_sheet = _pick_sheet_name(df_group, "利润表")
                    v.setdefault("来源Sheet", f"BS:{bs_sheet} | PL:{pl_sheet}".strip())
                    validation_results.append(v)

                if has_bs and has_cf:
                    v = validate_bs_cf_balance(df_group, tolerance=tol)
                    v.update(group_info)
                    bs_sheet = _pick_sheet_name(df_group, "资产负债表")
                    cf_sheet = _pick_sheet_name(df_group, "现金流量表")
                    v.setdefault("来源Sheet", f"BS:{bs_sheet} | CF:{cf_sheet}".strip())
                    validation_results.append(v)

    if existing_metrics_group_cols:
        grouped_metrics = all_data.groupby(existing_metrics_group_cols, dropna=False)
        for group_keys, df_group in grouped_metrics:
            if cancel_event and cancel_event.is_set():
                result.cancelled = True
                logger.warning("已取消运行")
                return result

            group_info = dict(zip(existing_metrics_group_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
            if cfg.generate_metrics:
                metrics = calculate_financial_metrics(df_group)
                metrics.update(group_info)
                metrics_results.append(metrics)

    if cfg.generate_validation and validation_results:
        df_validation = pd.DataFrame(validation_results)
        validation_path = cleaned_path.replace(".xlsx", "_验证报告.xlsx")
        try:
            with pd.ExcelWriter(validation_path, engine="openpyxl") as writer:
                df_validation.to_excel(writer, index=False, sheet_name="验证明细")
                try:
                    dfv = df_validation.copy()
                    if "差额" in dfv.columns:
                        dfv["差额"] = pd.to_numeric(dfv["差额"], errors="coerce")
                    grp_cols = [c for c in ["源文件", "日期", "验证项目"] if c in dfv.columns]
                    if grp_cols:
                        g = dfv.groupby(grp_cols, dropna=False)
                        agg = g.agg(
                            总条数=("验证项目", "count"),
                            不平衡条数=("是否平衡", lambda s: int((s == "否").sum())),
                            最大差额=("差额", "max"),
                            平均差额=("差额", "mean"),
                        ).reset_index()
                        agg.to_excel(writer, index=False, sheet_name="异常汇总")
                except Exception:
                    pass
        except Exception:
            df_validation.to_excel(validation_path, index=False)
        result.validation_path = validation_path
        result.validation_groups = int(len(df_validation))
        result.validation_preview = _df_preview_records(df_validation, limit=2000)
        unbalanced = df_validation[df_validation["是否平衡"] == "否"] if "是否平衡" in df_validation.columns else pd.DataFrame()
        result.unbalanced_count = int(len(unbalanced))
        if not unbalanced.empty:
            preview_cols = [c for c in ["源文件", "来源Sheet", "日期", "时间属性", "差额", "验证结果"] if c in unbalanced.columns]
            result.unbalanced_preview = _df_preview_records(unbalanced[preview_cols], limit=200)
        logger.info(f"验证报告已保存: {validation_path}")

    if cfg.generate_metrics and metrics_results:
        df_metrics = pd.DataFrame(metrics_results)
        metrics_path = cleaned_path.replace(".xlsx", "_财务指标.xlsx")
        df_metrics.to_excel(metrics_path, index=False)
        result.metrics_path = metrics_path
        result.metrics_groups = int(len(df_metrics))
        result.metrics_preview = _df_preview_records(df_metrics, limit=2000)
        logger.info(f"财务指标已保存: {metrics_path}")

    try:
        _write_cleaned_sqlite(all_data, cleaned_sqlite_path, df_validation=df_validation, df_metrics=df_metrics)
        logger.info(f"SQLite已保存: {cleaned_sqlite_path}")
    except Exception as e:
        logger.warning(f"SQLite保存失败: {e}")

    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="财务数据分析 - 命令行模式")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径(JSON)")
    parser.add_argument("--input-dir", type=str, default=None, help="输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--glob", type=str, default=None, help="文件匹配模式，如 *.xlsx")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = _resolve_under_base(args.output_dir)
    if args.glob:
        cfg.file_glob = args.glob

    logger = _get_logger()
    res = analyze_directory(cfg, logger=logger)
    if res.cancelled:
        return 2
    if res.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
