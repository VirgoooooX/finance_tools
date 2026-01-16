import datetime as _dt
import hashlib as _hashlib
import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from fa_platform.paths import default_data_root, default_output_root, ensure_dir, get_base_dir, resolve_under_base


def run_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_root(output_dir: str) -> str:
    base_dir = os.path.abspath(get_base_dir())
    raw = str(output_dir or "").strip() or str(default_output_root() or "").strip() or "output"
    out = resolve_under_base(raw)
    if os.path.abspath(out) == base_dir:
        out = os.path.abspath(os.path.join(out, "output"))
    return out


def build_run_dir(output_dir: str, tool_id: str, stamp: Optional[str] = None) -> Tuple[str, str]:
    tid = str(tool_id or "").strip()
    ts = str(stamp or "").strip() or run_timestamp()
    out_root = resolve_output_root(output_dir)
    run_dir = os.path.join(out_root, tid, ts) if tid else os.path.join(out_root, ts)
    ensure_dir(run_dir)
    ensure_dir(default_data_root())
    return ts, run_dir


def write_sqlite_tables(
    sqlite_path: str,
    cleaned: pd.DataFrame,
    validation: Optional[pd.DataFrame] = None,
    metrics: Optional[pd.DataFrame] = None,
) -> None:
    def _make_index_name(prefix: str, col: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_")
        digest = _hashlib.sha1(str(col).encode("utf-8")).hexdigest()[:12]
        if safe:
            return f"{prefix}_{safe}_{digest}"
        return f"{prefix}_{digest}"

    def _to_sql(df: pd.DataFrame, table: str, index_cols: Iterable[str]) -> None:
        df2 = df.copy()
        if table == "cleaned" and "金额" in df2.columns:
            df2["金额"] = pd.to_numeric(df2["金额"], errors="coerce").fillna(0.0)
        if table == "validation" and "差额" in df2.columns:
            df2["差额"] = pd.to_numeric(df2["差额"], errors="coerce")
        df2.to_sql(table, conn, if_exists="replace", index=False, chunksize=2000)
        for col in index_cols:
            if col in df2.columns:
                idx_name = _make_index_name(f"idx_{table}", col)
                conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}"("{col}")')

    ensure_dir(os.path.dirname(str(sqlite_path or "")) or os.getcwd())
    conn = sqlite3.connect(sqlite_path)
    try:
        _to_sql(
            cleaned,
            "cleaned",
            ["源文件", "日期", "报表类型", "大类", "时间属性", "科目", "金额", "来源Sheet"],
        )
        if validation is not None and isinstance(validation, pd.DataFrame) and not validation.empty:
            _to_sql(validation, "validation", ["源文件", "日期", "是否平衡", "验证项目", "时间属性", "差额", "来源Sheet"])
        if metrics is not None and isinstance(metrics, pd.DataFrame) and not metrics.empty:
            _to_sql(metrics, "metrics", ["源文件", "日期"])
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def build_artifacts(
    cleaned_path: str = "",
    cleaned_sqlite_path: str = "",
    validation_path: str = "",
    metrics_path: str = "",
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if cleaned_path:
        out.append({"name": "清洗表", "path": cleaned_path, "kind": "xlsx"})
    if validation_path:
        out.append({"name": "验证报告", "path": validation_path, "kind": "xlsx"})
    if metrics_path:
        out.append({"name": "财务指标", "path": metrics_path, "kind": "xlsx"})
    if cleaned_sqlite_path:
        out.append({"name": "清洗SQLite", "path": cleaned_sqlite_path, "kind": "sqlite"})
    return out

