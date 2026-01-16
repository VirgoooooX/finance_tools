import datetime as _dt
import json as _json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from fa_platform.paths import default_data_root, ensure_dir
from fa_platform.jsonx import sanitize_json


def _run_index_path() -> str:
    ensure_dir(default_data_root())
    return os.path.abspath(os.path.join(default_data_root(), "run_index.sqlite"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_run_index_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            tool_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            status TEXT,
            cleaned_path TEXT,
            cleaned_sqlite_path TEXT,
            cleaned_rows INTEGER,
            processed_files INTEGER,
            found_files_json TEXT,
            errors_json TEXT,
            meta_json TEXT,
            PRIMARY KEY (tool_id, run_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_tool_started ON runs(tool_id, started_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_tool_finished ON runs(tool_id, finished_at DESC)")
    conn.commit()


_TS_RE = re.compile(r"^\d{8}_\d{6}$")
_SQLITE_TS_RE = re.compile(r"_([0-9]{8}_[0-9]{6})\.sqlite$", re.IGNORECASE)


def _is_timestamp_folder(name: str) -> bool:
    return bool(_TS_RE.match(str(name or "").strip()))


def _derive_run_id_from_paths(cleaned_path: str, cleaned_sqlite_path: str) -> str:
    try:
        parent = os.path.basename(os.path.dirname(str(cleaned_path or "")))
        if _is_timestamp_folder(parent):
            return parent
    except Exception:
        pass

    try:
        m = _SQLITE_TS_RE.search(os.path.basename(str(cleaned_sqlite_path or "")))
        if m:
            ts = str(m.group(1) or "").strip()
            if _is_timestamp_folder(ts):
                return ts
    except Exception:
        pass

    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def upsert_run_from_result(tool_id: str, cfg: Any, res: Any) -> Tuple[str, str]:
    tid = str(tool_id or "").strip()
    cleaned_path = str(getattr(res, "cleaned_path", "") or "").strip()
    cleaned_sqlite_path = str(getattr(res, "cleaned_sqlite_path", "") or "").strip()

    rid = str(getattr(res, "run_id", "") or "").strip()
    if not rid:
        rid = _derive_run_id_from_paths(cleaned_path, cleaned_sqlite_path)
        try:
            setattr(res, "run_id", rid)
        except Exception:
            pass

    started_at = str(getattr(res, "run_started_at", "") or "").strip()
    finished_at = str(getattr(res, "run_finished_at", "") or "").strip()

    cancelled = bool(getattr(res, "cancelled", False))
    errors = list(getattr(res, "errors", []) or [])
    status = "cancelled" if cancelled else ("error" if errors else "ok")

    cleaned_rows = int(getattr(res, "cleaned_rows", 0) or 0)
    processed_files = int(getattr(res, "processed_files", 0) or 0)
    found_files = list(getattr(res, "found_files", []) or [])

    meta: Dict[str, Any] = {}
    try:
        meta["output_dir"] = str(getattr(cfg, "output_dir", "") or "").strip()
        meta["output_basename"] = str(getattr(cfg, "output_basename", "") or "").strip()
        meta["input_dir"] = str(getattr(cfg, "input_dir", "") or "").strip()
        meta["file_glob"] = str(getattr(cfg, "file_glob", "") or "").strip()
    except Exception:
        pass
    
    try:
        tp = getattr(cfg, "tool_params", None)
        if isinstance(tp, dict):
            bucket = tp.get(tid)
            meta["tool_params"] = sanitize_json(bucket if isinstance(bucket, dict) else {})
    except Exception:
        pass

    try:
        with _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO runs(
                    tool_id, run_id, started_at, finished_at, status,
                    cleaned_path, cleaned_sqlite_path,
                    cleaned_rows, processed_files,
                    found_files_json, errors_json, meta_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tool_id, run_id) DO UPDATE SET
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    status=excluded.status,
                    cleaned_path=excluded.cleaned_path,
                    cleaned_sqlite_path=excluded.cleaned_sqlite_path,
                    cleaned_rows=excluded.cleaned_rows,
                    processed_files=excluded.processed_files,
                    found_files_json=excluded.found_files_json,
                    errors_json=excluded.errors_json,
                    meta_json=excluded.meta_json
                """,
                (
                    tid,
                    rid,
                    started_at,
                    finished_at,
                    status,
                    cleaned_path,
                    cleaned_sqlite_path,
                    cleaned_rows,
                    processed_files,
                    _json.dumps(found_files, ensure_ascii=False),
                    _json.dumps(errors, ensure_ascii=False),
                    _json.dumps(meta, ensure_ascii=False),
                ),
            )
            conn.commit()
    except Exception:
        pass

    return tid, rid


def list_runs(tool_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    tid = str(tool_id or "").strip()
    lim = int(max(1, min(int(limit or 50), 500)))
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT tool_id, run_id, started_at, finished_at, status,
                       cleaned_path, cleaned_sqlite_path,
                       cleaned_rows, processed_files,
                       errors_json, meta_json
                FROM runs
                WHERE tool_id = ?
                ORDER BY COALESCE(finished_at, started_at) DESC, run_id DESC
                LIMIT ?
                """,
                (tid, lim),
            )
            rows = cur.fetchall()
    except Exception:
        rows = []

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            errors = _json.loads(r[9] or "[]")
        except Exception:
            errors = []
        try:
            meta = _json.loads(r[10] or "{}")
        except Exception:
            meta = {}
        out.append(
            {
                "tool_id": r[0],
                "run_id": r[1],
                "started_at": r[2],
                "finished_at": r[3],
                "status": r[4],
                "cleaned_path": r[5],
                "cleaned_sqlite_path": r[6],
                "cleaned_rows": r[7],
                "processed_files": r[8],
                "errors": errors,
                "meta": meta,
            }
        )
    return out


def get_run(tool_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    tid = str(tool_id or "").strip()
    rid = str(run_id or "").strip()
    if not tid or not rid:
        return None
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT tool_id, run_id, started_at, finished_at, status,
                       cleaned_path, cleaned_sqlite_path,
                       cleaned_rows, processed_files,
                       found_files_json, errors_json, meta_json
                FROM runs
                WHERE tool_id = ? AND run_id = ?
                LIMIT 1
                """,
                (tid, rid),
            )
            row = cur.fetchone()
    except Exception:
        row = None

    if not row:
        return None
    try:
        found_files = _json.loads(row[9] or "[]")
    except Exception:
        found_files = []
    try:
        errors = _json.loads(row[10] or "[]")
    except Exception:
        errors = []
    try:
        meta = _json.loads(row[11] or "{}")
    except Exception:
        meta = {}
    return {
        "tool_id": row[0],
        "run_id": row[1],
        "started_at": row[2],
        "finished_at": row[3],
        "status": row[4],
        "cleaned_path": row[5],
        "cleaned_sqlite_path": row[6],
        "cleaned_rows": row[7],
        "processed_files": row[8],
        "found_files": found_files,
        "errors": errors,
        "meta": meta,
    }


def get_latest_run(tool_id: str) -> Optional[Dict[str, Any]]:
    items = list_runs(tool_id, limit=1)
    return items[0] if items else None
