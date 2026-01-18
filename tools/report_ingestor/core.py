import pandas as pd
import datetime
import re
import hashlib
import glob
import os
import json
import logging
import threading
import sqlite3
from typing import Optional, Any, Dict, List, Tuple, Iterable
from dataclasses import asdict

from financial_analyzer_core import AppConfig, AnalysisResult, ProgressCallback
from financial_analyzer_core import _cleaned_sqlite_path_for as _cleaned_sqlite_path_for_common
from fa_platform.paths import (
    ensure_dir as _ensure_dir_common,
    default_output_root as _default_output_root_common,
    default_data_root as _default_data_root_common,
    resolve_under_base as _resolve_under_base_common,
    get_base_dir as _get_base_dir_common,
)
from fa_platform.jsonx import sanitize_json
from fa_platform.pipeline import build_artifacts as _build_artifacts_common, build_run_dir as _build_run_dir_common, write_sqlite_tables as _write_sqlite_tables_common

_TOOL_ID = os.path.basename(os.path.dirname(__file__))
_ACTIVE_TOOL_ID = ""


def _ensure_dir(path: str) -> None:
    _ensure_dir_common(path)


def _default_output_root() -> str:
    return _default_output_root_common()


def _default_data_root() -> str:
    return _default_data_root_common()


def _resolve_under_base(path: str) -> str:
    return _resolve_under_base_common(path)


def _get_base_dir():
    return _get_base_dir_common()


def _get_logger(name: str = "report_ingestor") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def _json_safe_value(v: Any) -> Any:
    return sanitize_json(v)


_LEGACY_TOP_LEVEL_TOOL_PARAM_KEYS = (
    "sheet_keyword_bs",
    "sheet_keyword_pl",
    "sheet_keyword_cf",
    "header_keyword_bs",
    "header_keyword_pl",
    "header_keyword_cf",
    "date_cells_bs",
    "date_cells_pl",
    "date_cells_cf",
    "validation_tolerance",
)


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


def _rules_path() -> str:
    tid = str(_ACTIVE_TOOL_ID or "").strip() or _TOOL_ID
    try:
        base = os.path.abspath(_get_base_dir())
        p = os.path.join(base, "tools", tid, "rules.json")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), "rules.json")


def _load_rules() -> Dict[str, Any]:
    p = _rules_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def load_config(path: Optional[str] = None) -> AppConfig:
    if not path:
        path = os.path.join(os.path.dirname(__file__), "config.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AppConfig()
        if isinstance(data, dict):
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            tid = str(data.get("tool_id") or _TOOL_ID).strip()
            tp = getattr(cfg, "tool_params", None)
            if not isinstance(tp, dict):
                cfg.tool_params = {}
                tp = cfg.tool_params
            bucket = tp.get(tid)
            if not isinstance(bucket, dict):
                bucket = {}
            merged = dict(bucket)
            for k in _LEGACY_TOP_LEVEL_TOOL_PARAM_KEYS:
                if k in data and k not in merged:
                    merged[k] = data.get(k)
            tp[tid] = merged
            cfg.tool_id = tid
        return cfg
    except Exception:
        return AppConfig()


def save_config(path: Optional[str], cfg: AppConfig) -> None:
    if not path:
        path = os.path.join(os.path.dirname(__file__), "config.json")
    _ensure_dir(os.path.dirname(path))
    data = asdict(cfg)
    try:
        base = os.path.abspath(_get_base_dir())
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


_RULES_CACHE: Dict[str, Any] = {"mtime": None, "data": None, "aliases": None}


def _normalize_subject_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s,，]", "", s)
    return s


def _expand_keywords(keywords: List[Any]) -> List[str]:
    rules = _load_rules()
    aliases = {}
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
    return _cleaned_sqlite_path_for_common(cleaned_path)


def _write_cleaned_sqlite(
    df: pd.DataFrame,
    sqlite_path: str,
    df_validation: Optional[pd.DataFrame] = None,
    df_metrics: Optional[pd.DataFrame] = None,
) -> None:
    _write_sqlite_tables_common(sqlite_path, df, validation=df_validation, metrics=df_metrics)


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
            df2[c] = pd.Series([_json_safe_value(x) for x in df2[c].tolist()], dtype="object")
        except Exception:
            try:
                df2[c] = df2[c].apply(_json_safe_value).astype(object)
            except Exception:
                df2[c] = df2[c].astype(str)
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
            if in_bounds(r + dr, c - radius) and _is_date_like(df.iat[r + dr, c - radius]):
                return df.iat[r + dr, c - radius]
            if in_bounds(r + dr, c + radius) and _is_date_like(df.iat[r + dr, c + radius]):
                return df.iat[r + dr, c + radius]
        for dc in range(-radius + 1, radius):
            if in_bounds(r - radius, c + dc) and _is_date_like(df.iat[r - radius, c + dc]):
                return df.iat[r - radius, c + dc]
            if in_bounds(r + radius, c + dc) and _is_date_like(df.iat[r + radius, c + dc]):
                return df.iat[r + radius, c + dc]
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
        t = str(s or "").strip().lower().replace("（", "(").replace("）", ")").replace("\u3000", " ")
        return re.sub(r"\s+", "", t)

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


def _find_footer_row_in_col(df: pd.DataFrame, start_row: int, col_idx: int, keyword: str) -> Optional[int]:
    def norm(s: Any) -> str:
        t = str(s or "").strip().lower().replace("（", "(").replace("）", ")").replace("\u3000", " ")
        return re.sub(r"\s+", "", t)

    try:
        kw = norm(keyword)
        if not kw:
            return None
        if col_idx < 0 or col_idx >= int(df.shape[1]):
            return None
        start_row = max(int(start_row), 0)
        if start_row >= int(df.shape[0]):
            return None
        series = df.iloc[start_row:, col_idx]
        values = series.to_numpy()
        for i in range(values.shape[0]):
            if kw in norm(values[i]):
                return int(series.index[i])
        return None
    except Exception:
        return None


def clean_date_str(date_val):
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


def _split_tokens(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            s = str(x or "").strip()
            if s:
                out.append(s)
        return out
    s = str(v or "").strip()
    if not s:
        return []
    if "," in s or "，" in s:
        parts = re.split(r"[,\uFF0C]+", s)
        out = [p.strip() for p in parts if p.strip()]
        return out
    return [s]


def _sheet_rules_config(cfg: AppConfig) -> Dict[str, Any]:
    v = _get_param(cfg, "sheet_rules", {}) or {}
    return v if isinstance(v, dict) else {}


def _sheet_rules_match_mode(cfg: AppConfig) -> str:
    mode = str(_sheet_rules_config(cfg).get("match_mode") or "contains").strip().lower()
    return mode if mode in ("exact", "contains", "regex") else "contains"


def _sheet_rules_case_insensitive(cfg: AppConfig) -> bool:
    return bool(_sheet_rules_config(cfg).get("case_insensitive", True))


def _sheet_rules_list(cfg: AppConfig) -> List[Dict[str, Any]]:
    rules = _sheet_rules_config(cfg).get("rules")
    if not isinstance(rules, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rules:
        if isinstance(r, dict):
            out.append(r)
    out.sort(key=lambda x: (int(x.get("priority") or 1000),))
    return out


def _match_sheet_name(sheet_name: str, pattern: str, match_mode: str, case_insensitive: bool) -> bool:
    s = str(sheet_name or "")
    p = str(pattern or "")
    if not p:
        return False
    if case_insensitive:
        s = s.upper()
        p = p.upper()
    if match_mode == "exact":
        return s == p
    if match_mode == "contains":
        return p in s
    if match_mode == "regex":
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            return re.search(pattern, sheet_name or "", flags=flags) is not None
        except Exception:
            return False
    return False


def _pick_sheet_rule(cfg: AppConfig, sheet_name: str) -> Optional[Dict[str, Any]]:
    match_mode = _sheet_rules_match_mode(cfg)
    case_insensitive = _sheet_rules_case_insensitive(cfg)
    for r in _sheet_rules_list(cfg):
        patterns = _split_tokens(r.get("patterns"))
        if any(_match_sheet_name(sheet_name, pat, match_mode, case_insensitive) for pat in patterns):
            return r
    return None


def _derive_period_id(report_date: Any) -> str:
    s = str(report_date or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    digits = re.findall(r"\d+", s)
    if len(digits) >= 2:
        y = digits[0]
        mo = digits[1].zfill(2)
        return f"{y}{mo}"
    return ""


def _extract_period_id_from_filename(filename: str) -> str:
    s = os.path.basename(str(filename or ""))
    m6 = re.search(r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])", s)
    if m6:
        return str(m6.group(0))
    m4 = re.search(r"\d{2}(?:0[1-9]|1[0-2])", s)
    if not m4:
        return ""
    y2 = int(m4.group(0)[:2])
    mm = m4.group(0)[2:4]
    yyyy = 2000 + y2 if y2 <= 79 else 1900 + y2
    return f"{yyyy}{mm}"


def _report_date_from_period_id(period_id: str) -> str:
    pid = str(period_id or "").strip()
    if re.match(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])$", pid):
        return f"{pid[:4]}-{pid[4:6]}-01"
    return ""


def _classify_amount(v: Any) -> Tuple[Optional[float], str, str]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, pd.Timestamp) and pd.isna(v)):
            return None, "", "EMPTY"
    except Exception:
        pass
    s = str(v)
    s2 = s.strip()
    if s2 == "":
        return None, "", "EMPTY"
    if s2 in ("—", "——", "-", "--", "－"):
        return None, s2, "NA"
    try:
        t = s2.replace("，", ",").replace(",", "").replace(" ", "")
        if t.startswith("(") and t.endswith(")"):
            t = "-" + t[1:-1]
        return float(t), s2, "OK"
    except Exception:
        return None, s2, "ERROR"


def _warehouse_path() -> str:
    return os.path.abspath(os.path.join(_default_data_root(), "warehouse.sqlite"))


def _ensure_warehouse_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_batch (
          batch_id TEXT PRIMARY KEY,
          tool_id TEXT,
          run_id TEXT,
          created_at TEXT NOT NULL,
          mode TEXT,
          input_dir TEXT,
          file_glob TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warehouse_cleaned (
          batch_id TEXT NOT NULL,
          row_hash TEXT NOT NULL,
          源文件 TEXT,
          来源Sheet TEXT,
          期间 TEXT,
          年份 TEXT,
          报表口径 TEXT,
          报表类型 TEXT,
          大类 TEXT,
          科目 TEXT,
          科目规范 TEXT,
          时间属性 TEXT,
          金额 REAL,
          PRIMARY KEY (batch_id, row_hash)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_warehouse_period_stmt ON warehouse_cleaned(期间, 报表类型, 时间属性)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_warehouse_period ON warehouse_cleaned(期间)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_warehouse_subject ON warehouse_cleaned(科目规范)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_warehouse_year ON warehouse_cleaned(年份)")


def _row_hash_for(values: Iterable[Any]) -> str:
    parts: List[str] = []
    for v in values:
        if v is None:
            parts.append("")
        else:
            parts.append(str(v))
    raw = "\u0001".join(parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _get_distinct_valid_periods(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    if "期间" not in df.columns:
        return []
    try:
        ser = df["期间"].dropna().astype(str).map(lambda x: str(x or "").strip())
    except Exception:
        return []
    if ser is None:
        return []
    try:
        periods = [p for p in ser.tolist() if p]
    except Exception:
        periods = []
    out: List[str] = []
    seen: set = set()
    for p in periods:
        if not p:
            continue
        if not re.match(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])$", p):
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    out.sort()
    return out


def _warehouse_user_version(conn: sqlite3.Connection) -> int:
    try:
        cur = conn.execute("PRAGMA user_version")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


def _set_warehouse_user_version(conn: sqlite3.Connection, v: int) -> None:
    try:
        conn.execute(f"PRAGMA user_version = {int(v)}")
    except Exception:
        pass


def _dedup_existing_warehouse_by_period(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_cleaned'")
        if not cur.fetchone():
            return
    except Exception:
        return

    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='import_batch'")
        if not cur.fetchone():
            return
    except Exception:
        return

    try:
        cur = conn.execute(
            """
            SELECT wc.期间, wc.batch_id, COALESCE(ib.created_at, '') as created_at
            FROM warehouse_cleaned wc
            LEFT JOIN import_batch ib ON ib.batch_id = wc.batch_id
            WHERE TRIM(COALESCE(wc.期间, '')) != ''
            GROUP BY wc.期间, wc.batch_id
            """
        )
        pairs = cur.fetchall() or []
    except Exception:
        pairs = []

    best: Dict[str, Tuple[str, str]] = {}
    for r in pairs:
        try:
            period = str(r[0] or "").strip()
            bid = str(r[1] or "").strip()
            created_at = str(r[2] or "").strip()
        except Exception:
            continue
        if not period or not bid:
            continue
        cur_best = best.get(period)
        if not cur_best:
            best[period] = (bid, created_at)
            continue
        best_bid, best_created = cur_best
        if created_at > best_created:
            best[period] = (bid, created_at)
        elif created_at == best_created and bid > best_bid:
            best[period] = (bid, created_at)

    for period, (keep_bid, _) in best.items():
        try:
            conn.execute(
                "DELETE FROM warehouse_cleaned WHERE TRIM(COALESCE(期间, '')) = ? AND batch_id <> ?",
                (period, keep_bid),
            )
        except Exception:
            pass


def _replace_warehouse_periods(conn: sqlite3.Connection, periods: List[str]) -> None:
    for p in (periods or []):
        pp = str(p or "").strip()
        if not pp:
            continue
        try:
            conn.execute("DELETE FROM warehouse_cleaned WHERE TRIM(COALESCE(期间, '')) = ?", (pp,))
        except Exception:
            pass


def _write_to_warehouse(df: pd.DataFrame, batch_id: str, cfg: AppConfig, mode: str) -> str:
    wp = _warehouse_path()
    conn = sqlite3.connect(wp)
    try:
        _ensure_warehouse_schema(conn)

        flags = _warehouse_user_version(conn)
        if not (int(flags) & 1):
            _dedup_existing_warehouse_by_period(conn)
            _set_warehouse_user_version(conn, int(flags) | 1)

        _replace_warehouse_periods(conn, _get_distinct_valid_periods(df))

        conn.execute(
            "INSERT OR IGNORE INTO import_batch(batch_id, tool_id, run_id, created_at, mode, input_dir, file_glob) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(batch_id),
                str(getattr(cfg, "tool_id", "") or _TOOL_ID),
                str(batch_id),
                datetime.datetime.now().isoformat(timespec="seconds"),
                str(mode or ""),
                str(getattr(cfg, "input_dir", "") or ""),
                str(getattr(cfg, "file_glob", "") or ""),
            ),
        )
        rows: List[Tuple[Any, ...]] = []
        for r in df.itertuples(index=False):
            rec = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, list(r)))
            period = str(rec.get("期间") or "").strip()
            year = str(rec.get("年份") or "").strip()
            row_hash = _row_hash_for(
                [
                    rec.get("源文件"),
                    rec.get("来源Sheet"),
                    period,
                    rec.get("报表口径"),
                    rec.get("报表类型"),
                    rec.get("大类"),
                    rec.get("科目规范"),
                    rec.get("时间属性"),
                    rec.get("金额"),
                ]
            )
            rows.append(
                (
                    str(batch_id),
                    row_hash,
                    rec.get("源文件"),
                    rec.get("来源Sheet"),
                    period,
                    year,
                    rec.get("报表口径"),
                    rec.get("报表类型"),
                    rec.get("大类"),
                    rec.get("科目"),
                    rec.get("科目规范"),
                    rec.get("时间属性"),
                    rec.get("金额"),
                )
            )
        conn.executemany(
            """
            INSERT OR IGNORE INTO warehouse_cleaned(
              batch_id, row_hash, 源文件, 来源Sheet, 期间, 年份, 报表口径, 报表类型, 大类, 科目, 科目规范, 时间属性, 金额
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return wp
    finally:
        try:
            conn.close()
        except Exception:
            pass


def clean_bs(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None, excel_file: Optional[pd.ExcelFile] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = excel_file.parse(sheet_name=sheet_name, header=None) if excel_file else pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, _get_param(cfg, "date_cells_bs", [[2, 3], [2, 2]]))
        report_date = clean_date_str(date_val)
        header_kw = str(_get_param(cfg, "header_keyword_bs", "期末余额") or "期末余额")
        header_row = _find_header_row(df, header_kw)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {header_kw}")
        data_start = int(header_row) + 1
        footer_kw = str(_get_param(cfg, "footer_keyword_bs", "资产总计") or "资产总计").strip()
        footer_row = _find_footer_row_in_col(df, data_start, 0, footer_kw)
        data_end = int(footer_row) + 1 if footer_row is not None else None

        df_left = df.iloc[data_start:data_end, [0, 1, 2]].copy()
        df_left.columns = ["科目", "年初余额", "期末余额"]
        df_left["大类"] = "资产"
        df_parts = [df_left]
        if df.shape[1] >= 6:
            df_right = df.iloc[data_start:data_end, [3, 4, 5]].copy()
            df_right.columns = ["科目", "年初余额", "期末余额"]
            df_right["大类"] = "负债及权益"
            df_parts.append(df_right)
        df_clean = pd.concat(df_parts, ignore_index=True).dropna(subset=["科目"])
        df_clean = df_clean[df_clean["科目"].astype(str).str.strip() != ""]
        df_final = df_clean.melt(id_vars=["大类", "科目"], value_vars=["年初余额", "期末余额"], var_name="时间属性", value_name="金额")
        df_final["报表类型"] = "资产负债表"
        period_id = _derive_period_id(report_date)
        df_final["期间"] = period_id
        df_final["年份"] = str(period_id or "")[:4] if period_id else ""
        df_final["来源Sheet"] = sheet_name
        if logger:
            logger.info(f"BS-合并 处理完成: {sheet_name}, 提取 {len(df_final)} 行")
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_pl(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None, excel_file: Optional[pd.ExcelFile] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = excel_file.parse(sheet_name=sheet_name, header=None) if excel_file else pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, _get_param(cfg, "date_cells_pl", [[2, 2], [2, 1]]))
        report_date = clean_date_str(date_val)
        header_kw = str(_get_param(cfg, "header_keyword_pl", "本年累计") or "本年累计")
        header_row = _find_header_row(df, header_kw)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {header_kw}")
        df_clean = df.iloc[header_row + 1 :, [0, 2, 3]].copy()
        df_clean.columns = ["科目", "本期金额", "本年累计金额"]
        df_clean = df_clean.dropna(subset=["科目"])
        df_final = df_clean.melt(id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额")
        df_final["大类"] = "损益"
        df_final["报表类型"] = "利润表"
        period_id = _derive_period_id(report_date)
        df_final["期间"] = period_id
        df_final["年份"] = str(period_id or "")[:4] if period_id else ""
        df_final["来源Sheet"] = sheet_name
        if logger:
            logger.info(f"CF-合并 处理完成: {sheet_name}, 提取 {len(df_final)} 行")
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


def clean_cf(file_path, sheet_name, cfg: AppConfig, logger: Optional[logging.Logger] = None, excel_file: Optional[pd.ExcelFile] = None):
    if logger:
        logger.info(f"正在处理: {sheet_name} ...")
    try:
        df = excel_file.parse(sheet_name=sheet_name, header=None) if excel_file else pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        date_val = _read_date_from_cells(df, _get_param(cfg, "date_cells_cf", [[2, 4], [2, 0]]))
        report_date = clean_date_str(date_val)
        header_kw = str(_get_param(cfg, "header_keyword_cf", "本期金额") or "本期金额")
        header_row = _find_header_row(df, header_kw)
        if header_row is None:
            raise ValueError(f"未找到表头关键字: {header_kw}")
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
        df_final = df_combined.melt(id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额")
        df_final["大类"] = "现金流"
        df_final["报表类型"] = "现金流量表"
        period_id = _derive_period_id(report_date)
        df_final["期间"] = period_id
        df_final["年份"] = str(period_id or "")[:4] if period_id else ""
        df_final["来源Sheet"] = sheet_name
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


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


def extract_amount(df, keywords, sheet_type=None, time_attr=None, category=None):
    val, found, _ = extract_amount_info(df, keywords, sheet_type=sheet_type, time_attr=time_attr, category=category)
    return val if found else 0.0


def validate_balance_sheet(df_group, tolerance: float = 0.01, time_attr: str = "期末余额", assets_keywords=None, liabilities_keywords=None, equity_keywords=None):
    assets, af, _ = extract_amount_info(df_group, assets_keywords or ["资产总计", "资产总额", "资产合计"], sheet_type="资产负债表", time_attr=time_attr, category="资产")
    liabilities, lf, _ = extract_amount_info(df_group, liabilities_keywords or ["负债合计", "负债总计", "负债总额"], sheet_type="资产负债表", time_attr=time_attr, category="负债及权益")
    equity, ef, _ = extract_amount_info(df_group, equity_keywords or ["所有者权益合计", "股东权益合计", "权益合计"], sheet_type="资产负债表", time_attr=time_attr, category="负债及权益")
    if not (af and lf and ef):
        return {"验证项目": "BS表资产=负债+权益验证", "时间属性": time_attr, "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs(assets - (liabilities + equity))
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表资产=负债+权益验证", "时间属性": time_attr, "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def validate_bs_pl_balance(df_group, tolerance: float = 0.01) -> Dict[str, Any]:
    re_begin, rb_f, _ = extract_amount_info(df_group, ["未分配利润"], sheet_type="资产负债表", time_attr="年初余额", category="负债及权益")
    re_end, re_f, _ = extract_amount_info(df_group, ["未分配利润"], sheet_type="资产负债表", time_attr="期末余额", category="负债及权益")
    np, np_f, _ = extract_amount_info(df_group, ["归属于母公司所有者的净利润", "归属于母公司股东的净利润"], sheet_type="利润表", time_attr="本年累计金额")
    if not (rb_f and re_f and np_f):
        return {"验证项目": "BS表与PL表平衡验证", "时间属性": "年初/期末 vs 本年累计", "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs((re_end - re_begin) - np)
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表与PL表平衡验证", "时间属性": "年初/期末 vs 本年累计", "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def validate_bs_cf_balance(df_group, tolerance: float = 0.01) -> Dict[str, Any]:
    bank_end, bf, _ = extract_amount_info(df_group, ["银行存款"], sheet_type="资产负债表", time_attr="期末余额", category="资产")
    cf_end, cf_f, _ = extract_amount_info(df_group, ["期末现金及现金等价物余额", "期末现金及现金等价物余额(附注)"], sheet_type="现金流量表", time_attr="本年累计金额")
    if not (bf and cf_f):
        return {"验证项目": "BS表与CF表平衡验证", "时间属性": "期末余额 vs 本年累计", "差额": None, "是否平衡": "否", "验证结果": "数据缺失"}
    diff = abs(bank_end - cf_end)
    is_balanced = diff <= tolerance
    return {"验证项目": "BS表与CF表平衡验证", "时间属性": "期末余额 vs 本年累计", "差额": diff, "是否平衡": "是" if is_balanced else "否", "验证结果": "通过" if is_balanced else f"不平衡(差额:{diff:.2f})"}


def _format_percent(val: Optional[float]) -> Optional[str]:
    try:
        return f"{float(val) * 100:.2f}%" if val is not None else None
    except Exception:
        return None


def _round_number(val: Optional[float], digits: int = 2) -> Optional[float]:
    try:
        return round(float(val), int(digits)) if val is not None else None
    except Exception:
        return None


def _safe_div(n: float, d: float) -> Optional[float]:
    return n / d if d != 0 else None


def calculate_financial_metrics(df_group):
    rules = _load_rules()
    metrics_def = rules.get("metrics", [])
    variables_def = rules.get("variables", {})
    var_values = {}
    if variables_def:
        for var_name, var_rule in variables_def.items():
            val = extract_amount(df_group, var_rule.get("keywords", []), sheet_type=var_rule.get("sheet_type"), time_attr=var_rule.get("time_attr"), category=var_rule.get("category"))
            var_values[var_name] = val

    if not metrics_def:
        return {}

    metrics = {}
    context = {"safe_div": _safe_div, "abs": abs, "round": round, "max": max, "min": min, **var_values}
    for m in metrics_def:
        name, formula, fmt = m.get("name"), m.get("formula"), m.get("format")
        if not name or not formula:
            continue
        try:
            val = eval(formula, {"__builtins__": {}}, context)
        except Exception:
            val = None
        if fmt == "percent":
            metrics[name] = _format_percent(val)
        elif fmt == "round2":
            metrics[name] = _round_number(val, 2)
        else:
            metrics[name] = val
    return metrics


def _list_excel_files(cfg: AppConfig) -> List[str]:
    pattern = os.path.join(cfg.input_dir, cfg.file_glob)
    files = [os.path.abspath(p) for p in glob.glob(pattern) if os.path.isfile(p)]
    if cfg.exclude_output_files:
        exclude = {
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename)),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_验证报告.xlsx"))),
            os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename.replace(".xlsx", "_财务指标.xlsx"))),
        }
        files = [p for p in files if os.path.abspath(p) not in exclude]
    files.sort(key=lambda p: p.lower())
    return files


def _pick_sheet_name(df_group: pd.DataFrame, sheet_type: str) -> str:
    if df_group is None or df_group.empty:
        return ""
    try:
        sub = df_group[df_group["报表类型"] == sheet_type]
        if sub.empty:
            return ""
        return str(sub["来源Sheet"].mode(dropna=False).iat[0])
    except Exception:
        return ""


def run_analysis(
    cfg: AppConfig,
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> AnalysisResult:
    result = AnalysisResult()
    if logger is None:
        logger = _get_logger()

    base_dir = os.path.abspath(_get_base_dir())
    output_root_raw = str(cfg.output_dir or "").strip() or _default_output_root()
    output_root = _resolve_under_base(output_root_raw)
    if output_root == base_dir:
        output_root = os.path.abspath(os.path.join(output_root, "output"))

    stamp = _run_timestamp()
    tool_id = str(getattr(cfg, "tool_id", "")).strip() or os.path.basename(os.path.dirname(__file__))
    global _ACTIVE_TOOL_ID
    _ACTIVE_TOOL_ID = tool_id
    result.run_id = stamp
    tool_dir = os.path.abspath(os.path.join(output_root, tool_id)) if tool_id else os.path.abspath(output_root)
    _ensure_dir(tool_dir)
    monthly_dir = os.path.abspath(os.path.join(tool_dir, "monthly"))
    _ensure_dir(monthly_dir)
    _ensure_dir(_default_data_root())
    files = _list_excel_files(cfg)
    result.found_files = files

    if not files:
        logger.warning("未找到任何匹配的 .xlsx 文件")
        return result

    logger.info(f"找到 {len(files)} 个Excel文件")
    all_files_data = []
    has_sheet_rules = bool(_sheet_rules_list(cfg))

    for idx, file_path in enumerate(files, start=1):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            logger.warning("已取消运行")
            return result
        if progress_cb:
            progress_cb("file", idx, len(files), os.path.basename(file_path))
        base_name = os.path.basename(file_path)
        logger.info(f"正在处理文件: {base_name}")
        file_period_id = _extract_period_id_from_filename(base_name)
        file_report_date = _report_date_from_period_id(file_period_id)

        try:
            excel_file = pd.ExcelFile(file_path)
            try:
                all_sheets = excel_file.sheet_names
                file_sheets_data: List[pd.DataFrame] = []
                if has_sheet_rules:
                    for sheet_name in (all_sheets or []):
                        if cancel_event and cancel_event.is_set():
                            result.cancelled = True
                            return result
                        rule = _pick_sheet_rule(cfg, sheet_name)
                        if not rule:
                            continue
                        st = str(rule.get("statement") or "").strip().upper()
                        df: Optional[pd.DataFrame] = None
                        if st in ("BS", "资产负债表"):
                            df = clean_bs(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                        elif st in ("PL", "利润表"):
                            df = clean_pl(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                        elif st in ("CF", "现金流量表", "现金流"):
                            df = clean_cf(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                        if df is None or df.empty:
                            continue
                        if file_period_id:
                            df["期间"] = file_period_id
                            df["年份"] = str(file_period_id)[:4]

                        scope_raw = str(rule.get("scope") or "").strip()
                        scope = scope_raw
                        if scope_raw.lower() in ("merge",):
                            scope = "合并"
                        elif scope_raw.lower() in ("single",):
                            scope = "单体"

                        entity = str(rule.get("entity") or "").strip()
                        if entity:
                            df["主体"] = entity
                        df["报表版本"] = "月度"
                        if scope:
                            df["报表口径"] = scope
                        file_sheets_data.append(df)
                else:
                    kw_bs = _get_param(cfg, "sheet_keyword_bs", "BS-合并")
                    kw_pl = _get_param(cfg, "sheet_keyword_pl", "PL-合并")
                    kw_cf = _get_param(cfg, "sheet_keyword_cf", "CF-合并")

                    def _match_sheet(n: str, p: Any) -> bool:
                        toks = _split_tokens(p)
                        if not toks:
                            return False
                        up = str(n or "").upper()
                        return any(str(x).upper() in up for x in toks if x)

                    bs_sheets = [s for s in all_sheets if _match_sheet(s, kw_bs)]
                    pl_sheets = [s for s in all_sheets if _match_sheet(s, kw_pl)]
                    cf_sheets = [s for s in all_sheets if _match_sheet(s, kw_cf)]

                    for sheet in bs_sheets:
                        if cancel_event and cancel_event.is_set():
                            result.cancelled = True
                            return result
                        df = clean_bs(file_path, sheet, cfg, logger, excel_file=excel_file)
                        if not df.empty:
                            if file_period_id:
                                df["期间"] = file_period_id
                                df["年份"] = str(file_period_id)[:4]
                            file_sheets_data.append(df)
                    for sheet in pl_sheets:
                        if cancel_event and cancel_event.is_set():
                            result.cancelled = True
                            return result
                        df = clean_pl(file_path, sheet, cfg, logger, excel_file=excel_file)
                        if not df.empty:
                            if file_period_id:
                                df["期间"] = file_period_id
                                df["年份"] = str(file_period_id)[:4]
                            file_sheets_data.append(df)
                    for sheet in cf_sheets:
                        if cancel_event and cancel_event.is_set():
                            result.cancelled = True
                            return result
                        df = clean_cf(file_path, sheet, cfg, logger, excel_file=excel_file)
                        if not df.empty:
                            if file_period_id:
                                df["期间"] = file_period_id
                                df["年份"] = str(file_period_id)[:4]
                            file_sheets_data.append(df)

                if file_sheets_data:
                    file_data = pd.concat(file_sheets_data, ignore_index=True)
                    file_data["源文件"] = os.path.basename(file_path)
                    if "报表版本" not in file_data.columns:
                        file_data["报表版本"] = "月度"
                    if "报表口径" not in file_data.columns:
                        file_data["报表口径"] = "合并"
                    if "主体" not in file_data.columns:
                        file_data["主体"] = ""
                    all_files_data.append(file_data)
                    result.processed_files += 1
            finally:
                excel_file.close()
        except Exception as e:
            logger.error(f"读取失败: {e}")
            result.errors.append(f"{os.path.basename(file_path)}: {e}")

    if cancel_event and cancel_event.is_set():
        result.cancelled = True
        return result
    if not all_files_data:
        return result

    all_data = pd.concat(all_files_data, ignore_index=True)

    if "报表口径" not in all_data.columns:
        all_data["报表口径"] = "合并"
    if "报表版本" not in all_data.columns:
        all_data["报表版本"] = "月度"
    if "主体" not in all_data.columns:
        all_data["主体"] = ""
    if "来源Sheet" not in all_data.columns:
        all_data["来源Sheet"] = ""
    if "源文件" not in all_data.columns:
        all_data["源文件"] = ""

    amt_info = all_data["金额"].apply(_classify_amount)
    all_data["金额"] = amt_info.apply(lambda t: t[0])
    if "期间" not in all_data.columns:
        all_data["期间"] = ""
    if "年份" not in all_data.columns:
        all_data["年份"] = ""
    all_data["科目规范"] = all_data["科目"].apply(_normalize_subject_text)
    try:
        all_data = all_data.drop(columns=[c for c in ["日期", "金额文本", "值状态"] if c in all_data.columns])
    except Exception:
        pass

    base_name = str(cfg.output_basename or "").strip() or "清洗数据.xlsx"
    stem, ext = os.path.splitext(base_name)
    ext = ext or ".xlsx"
    cleaned_path = os.path.abspath(os.path.join(monthly_dir, f"{stem}_{stamp}{ext}"))
    all_data.to_excel(cleaned_path, index=False)
    result.cleaned_path = cleaned_path
    result.cleaned_rows = int(len(all_data))
    try:
        periods: List[str] = []
        try:
            if "期间" in all_data.columns:
                periods = (
                    all_data["期间"]
                    .dropna()
                    .astype(str)
                    .map(lambda x: str(x or "").strip())
                    .tolist()
                )
        except Exception:
            periods = []

        valid_periods = sorted({p for p in periods if re.match(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])$", p)})
        unknown_df: Optional[pd.DataFrame] = None
        if "期间" in all_data.columns:
            unknown_df = all_data[~all_data["期间"].astype(str).str.match(r"^(?:19|20)\d{2}(?:0[1-9]|1[0-2])$")]

        for pid in valid_periods:
            dfp = all_data[all_data["期间"].astype(str) == pid]
            if dfp is None or dfp.empty:
                continue
            out_name = f"清洗数据_{pid}.xlsx"
            out_path = os.path.abspath(os.path.join(monthly_dir, out_name))
            dfp.to_excel(out_path, index=False)

        if unknown_df is not None and not unknown_df.empty:
            out_path = os.path.abspath(os.path.join(monthly_dir, "清洗数据_未知期间.xlsx"))
            unknown_df.to_excel(out_path, index=False)
    except Exception:
        pass
    try:
        _write_to_warehouse(all_data, batch_id=str(stamp), cfg=cfg, mode=_sheet_rules_match_mode(cfg))
    except Exception as e:
        result.errors.append(f"写入累计库失败: {e}")
    result.artifacts = _build_artifacts_common(
        cleaned_path=str(result.cleaned_path or ""),
        cleaned_sqlite_path="",
        validation_path=str(result.validation_path or ""),
        metrics_path=str(result.metrics_path or ""),
    )
    try:
        wp = _warehouse_path()
        if wp and os.path.exists(wp):
            result.artifacts = (result.artifacts or []) + [{"name": "累计库(SQLite)", "path": wp, "kind": "sqlite"}]
    except Exception:
        pass
    return result
