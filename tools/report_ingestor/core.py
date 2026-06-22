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
from fa_platform.paths import (
    ensure_dir as _ensure_dir_common,
    default_output_root as _default_output_root_common,
    default_data_root as _default_data_root_common,
    resolve_under_base as _resolve_under_base_common,
    get_base_dir as _get_base_dir_common,
)
from fa_platform.pipeline import build_artifacts as _build_artifacts_common

_TOOL_ID = os.path.basename(os.path.dirname(__file__))


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


def _normalize_subject_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s,，]", "", s)
    return s


def _run_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_int_pair(pair: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        return int(pair[0]), int(pair[1])
    except Exception:
        return None


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


_SUBJECT_LABELS = ("编制单位", "编报单位", "填报单位", "单位名称", "公司名称", "企业名称", "纳税人名称", "会计主体")
_SUBJECT_SUFFIXES = ("有限公司", "有限责任公司", "股份有限公司", "集团", "公司", "厂", "中心", "合作社", "合伙企业", "事务所")


def _meta_cell_text(v: Any) -> str:
    try:
        if v is None or pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v or "").strip()
    if not s or s.lower() == "nan":
        return ""
    return s.replace("：", ":").replace("　", " ")


def _header_rows(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 8) -> List[List[str]]:
    rows: List[List[str]] = []
    try:
        rmax = min(int(max_rows), int(df.shape[0]))
        cmax = min(int(max_cols), int(df.shape[1]))
        for r in range(rmax):
            row: List[str] = []
            for c in range(cmax):
                row.append(_meta_cell_text(df.iat[r, c]))
            rows.append(row)
    except Exception:
        return []
    return rows


def _clean_subject_candidate(text: Any) -> str:
    s = _meta_cell_text(text)
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    for label in _SUBJECT_LABELS:
        s = re.sub(rf"^{re.escape(label)}[:：]?", "", s)
    s = re.split(r"(?:金额)?单位[:：]?元|会计期间|报表日期|日期[:：]?|统一社会信用代码", s, maxsplit=1)[0]
    s = s.strip(" :-_;,，。")
    if not s:
        return ""
    if any(x in s for x in ("资产负债表", "利润表", "现金流量表", "科目", "项目", "金额")):
        return ""
    if re.fullmatch(r"[\d./年月日 -]+", s):
        return ""
    return s


def _looks_like_subject(text: str) -> bool:
    s = _clean_subject_candidate(text)
    if len(s) < 2:
        return False
    if any(s.endswith(x) or x in s for x in _SUBJECT_SUFFIXES):
        return True
    return len(s) >= 4 and not re.search(r"\d", s)


def _infer_subject_from_header(rows: List[List[str]]) -> str:
    for row in rows:
        for idx, cell in enumerate(row):
            if not cell:
                continue
            if any(label in cell for label in _SUBJECT_LABELS):
                cand = _clean_subject_candidate(cell)
                if _looks_like_subject(cand):
                    return cand
                for nxt in row[idx + 1 : idx + 4]:
                    cand = _clean_subject_candidate(nxt)
                    if _looks_like_subject(cand):
                        return cand
    return ""


def _infer_caliber_from_header(rows: List[List[str]]) -> str:
    text = " ".join(cell for row in rows for cell in row if cell)
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return ""

    single_patterns = (
        r"单体.{0,8}(报表|口径|资产负债表|利润表|现金流量表|财务报表)",
        r"(母公司|本部|个别).{0,8}(报表|口径|资产负债表|利润表|现金流量表|财务报表)",
        r"(报表|口径|资产负债表|利润表|现金流量表|财务报表).{0,8}(单体|母公司|本部|个别)",
    )
    merged_patterns = (
        r"合并.{0,8}(报表|口径|资产负债表|利润表|现金流量表|财务报表)",
        r"(报表|口径|资产负债表|利润表|现金流量表|财务报表).{0,8}合并",
    )
    single_score = sum(1 for p in single_patterns if re.search(p, compact))
    merged_score = sum(1 for p in merged_patterns if re.search(p, compact))
    if single_score > merged_score:
        return "单体"
    if merged_score > single_score:
        return "合并"
    return ""


def _infer_header_metadata(df: pd.DataFrame) -> Dict[str, str]:
    rows = _header_rows(df)
    return {
        "主体": _infer_subject_from_header(rows),
        "报表口径": _infer_caliber_from_header(rows),
    }


# Sheet 名称 -> 报表类型的固定映射
_SHEET_TYPE_MAP = {"资产负债表": "BS", "利润表": "PL", "现金流量表": "CF"}


def _normalize_sheet_name(sheet_name: Any) -> str:
    s = str(sheet_name or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")").replace("\u3000", " ")
    return re.sub(r"\s+", "", s)


def _classify_sheet_type(sheet_name: Any) -> str:
    raw = str(sheet_name or "").strip()
    if raw in _SHEET_TYPE_MAP:
        return _SHEET_TYPE_MAP[raw]

    s = _normalize_sheet_name(raw)
    if not s:
        return ""
    if "资产负债" in s:
        return "BS"
    if "现金流" in s:
        return "CF"
    if "利润表" in s or "损益表" in s:
        return "PL"
    return ""


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
        meta = _infer_header_metadata(df)
        date_val = _read_date_from_cells(df, [[1, 0], [1, 3]])
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, "期末余额")
        if header_row is None:
            raise ValueError("未找到表头关键字: 期末余额")
        data_start = int(header_row) + 1
        footer_row = _find_footer_row_in_col(df, data_start, 0, "资产总计")
        data_end = int(footer_row) + 1 if footer_row is not None else None

        df_left = df.iloc[data_start:data_end, [0, 1, 2]].copy()
        # 新报表格式：col1=期末余额, col2=年初余额（原旧格式为年初/期末，现已对调）
        df_left.columns = ["科目", "期末余额", "年初余额"]
        df_left["大类"] = "资产"
        df_parts = [df_left]
        if df.shape[1] >= 6:
            df_right = df.iloc[data_start:data_end, [3, 4, 5]].copy()
            df_right.columns = ["科目", "期末余额", "年初余额"]
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
        df_final["报表口径"] = meta.get("报表口径") or "未知"
        df_final["主体"] = meta.get("主体") or ""
        if logger:
            logger.info(f"BS 处理完成: {sheet_name}, 提取 {len(df_final)} 行")
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
        meta = _infer_header_metadata(df)
        date_val = _read_date_from_cells(df, [[1, 0], [1, 3]])
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, "本期金额")
        if header_row is None:
            raise ValueError("未找到表头关键字: 本期金额")
        # 新报表格式：col0=科目, col1=本期金额, col2=本年金额（即本年累计）
        df_clean = df.iloc[header_row + 1 :, [0, 1, 2]].copy()
        df_clean.columns = ["科目", "本期金额", "本年累计金额"]
        df_clean = df_clean.dropna(subset=["科目"])
        df_final = df_clean.melt(id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额")
        df_final["大类"] = "损益"
        df_final["报表类型"] = "利润表"
        period_id = _derive_period_id(report_date)
        df_final["期间"] = period_id
        df_final["年份"] = str(period_id or "")[:4] if period_id else ""
        df_final["来源Sheet"] = sheet_name
        df_final["报表口径"] = meta.get("报表口径") or "未知"
        df_final["主体"] = meta.get("主体") or ""
        if logger:
            logger.info(f"PL 处理完成: {sheet_name}, 提取 {len(df_final)} 行")
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
        meta = _infer_header_metadata(df)
        date_val = _read_date_from_cells(df, [[1, 0], [1, 3]])
        report_date = clean_date_str(date_val)
        header_row = _find_header_row(df, "本期金额")
        if header_row is None:
            raise ValueError("未找到表头关键字: 本期金额")
        # 新报表格式：简单4列，col0=科目, col1=本期金额, col2=本年金额（即本年累计）
        # 旧格式为多列双栏，新格式已去掉右侧双栏
        df_combined = df.iloc[header_row + 1 :, [0, 1, 2]].copy()
        df_combined.columns = ["科目", "本期金额", "本年累计金额"]
        df_combined = df_combined.dropna(subset=["科目"])
        df_combined = df_combined[df_combined["科目"].astype(str).str.strip() != ""]
        df_final = df_combined.melt(id_vars=["科目"], value_vars=["本期金额", "本年累计金额"], var_name="时间属性", value_name="金额")
        df_final["大类"] = "现金流"
        df_final["报表类型"] = "现金流量表"
        period_id = _derive_period_id(report_date)
        df_final["期间"] = period_id
        df_final["年份"] = str(period_id or "")[:4] if period_id else ""
        df_final["来源Sheet"] = sheet_name
        df_final["报表口径"] = meta.get("报表口径") or "未知"
        df_final["主体"] = meta.get("主体") or ""
        if logger:
            logger.info(f"CF 处理完成: {sheet_name}, 提取 {len(df_final)} 行")
        return df_final
    except Exception as e:
        if logger:
            logger.error(f"{sheet_name} 处理失败: {e}")
        return pd.DataFrame()


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

        try:
            excel_file = pd.ExcelFile(file_path)
            try:
                all_sheets = excel_file.sheet_names
                file_sheets_data: List[pd.DataFrame] = []
                recognized_sheets: List[str] = []
                empty_sheets: List[str] = []
                for sheet_name in (all_sheets or []):
                    if cancel_event and cancel_event.is_set():
                        result.cancelled = True
                        return result
                    st = _classify_sheet_type(sheet_name)
                    if not st:
                        continue
                    recognized_sheets.append(str(sheet_name))
                    df: Optional[pd.DataFrame] = None
                    if st == "BS":
                        df = clean_bs(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                    elif st == "PL":
                        df = clean_pl(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                    elif st == "CF":
                        df = clean_cf(file_path, sheet_name, cfg, logger, excel_file=excel_file)
                    if df is None or df.empty:
                        empty_sheets.append(str(sheet_name))
                        continue
                    if file_period_id:
                        df["期间"] = file_period_id
                        df["年份"] = str(file_period_id)[:4]
                    file_sheets_data.append(df)

                if not recognized_sheets and logger:
                    preview = ", ".join([str(x) for x in (all_sheets or [])[:8]])
                    logger.warning(f"{base_name} 未识别到三大报表Sheet；前几个Sheet: {preview}")
                elif empty_sheets and logger:
                    logger.warning(f"{base_name} 以下报表Sheet未提取到有效行: {', '.join(empty_sheets)}")

                if file_sheets_data:
                    file_data = pd.concat(file_sheets_data, ignore_index=True)
                    file_data["源文件"] = os.path.basename(file_path)
                    if "报表版本" not in file_data.columns:
                        file_data["报表版本"] = "月度"
                    if "报表口径" not in file_data.columns:
                        file_data["报表口径"] = "未知"
                    if "主体" not in file_data.columns:
                        file_data["主体"] = ""
                    all_files_data.append(file_data)
                    result.processed_files += 1
                elif recognized_sheets and logger:
                    logger.warning(f"{base_name} 已识别报表Sheet但未生成清洗数据")
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
        all_data["报表口径"] = "未知"
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
        _write_to_warehouse(all_data, batch_id=str(stamp), cfg=cfg, mode="fixed")
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
