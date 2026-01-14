import os
import sys
import re
import json
import queue
import threading
import logging
import socket
from dataclasses import asdict
from typing import Optional, Any, Dict, List
from pathlib import Path
import sqlite3

import pandas as pd

def get_resource_path(relative_path: str) -> Path:
    """获取资源文件的绝对路径，兼容开发环境和 PyInstaller 打包环境"""
    try:
        # PyInstaller 打包后的临时目录
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).resolve().parent
    return base_path / relative_path

from financial_analyzer_core import (
    OUTPUT_PATH,
    AppConfig,
    AnalysisResult,
    get_base_dir,
    load_config,
    save_config,
    analyze_directory,
    _cleaned_sqlite_path_for,
    _get_logger,
    _list_excel_files,
)


class _WebEventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: List["queue.Queue[Dict[str, Any]]"] = []

    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def publish(self, item: Dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except Exception:
                pass


class _WebQueueLogHandler(logging.Handler):
    def __init__(self, hub: _WebEventHub):
        super().__init__()
        self.hub = hub
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.hub.publish({"type": "log", "level": record.levelname, "message": msg})
        except Exception:
            pass


class _WebRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.hub = _WebEventHub()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.last_result: Optional[AnalysisResult] = None
        self.running = False
        self.current_progress: Dict[str, Any] = {"current": 0, "total": 1, "detail": ""}
        self._lock = threading.Lock()
        self._cleaned_cache: Dict[str, Any] = {"path": None, "mtime": None, "df": None}

    def set_config(self, data: Dict[str, Any]) -> AppConfig:
        cfg = AppConfig()
        for k, v in (data or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if not getattr(cfg, "input_dir", None):
            cfg.input_dir = os.getcwd()
        if not getattr(cfg, "file_glob", None):
            cfg.file_glob = "*.xlsx"
        out_dir = str(getattr(cfg, "output_dir", "") or "").strip()
        if not out_dir:
            cfg.output_dir = "output"
        else:
            cfg.output_dir = out_dir
        if not getattr(cfg, "output_basename", None):
            cfg.output_basename = OUTPUT_PATH
        try:
            cfg.validation_tolerance = float(getattr(cfg, "validation_tolerance", AppConfig().validation_tolerance))
        except Exception:
            cfg.validation_tolerance = AppConfig().validation_tolerance

        with self._lock:
            self.cfg = cfg
        save_config(self.config_path, cfg)
        self.hub.publish({"type": "status", "message": "已保存配置"})
        return cfg

    def get_config(self) -> AppConfig:
        with self._lock:
            return self.cfg

    def scan(self) -> List[str]:
        cfg = self.get_config()
        files = _list_excel_files(cfg)
        return files

    def start(self) -> bool:
        with self._lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False
            self.cancel_event.clear()
            self.running = True
            self.last_result = None
            self.current_progress = {"current": 0, "total": 1, "detail": ""}

        handler = _WebQueueLogHandler(self.hub)
        logger = _get_logger(handler=handler)

        def progress_cb(stage: str, current: int, total: int, detail: str) -> None:
            with self._lock:
                self.current_progress = {"stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)}
            self.hub.publish({"type": "progress", "stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)})

        def worker() -> None:
            try:
                cfg = self.get_config()
                res = analyze_directory(cfg, logger=logger, progress_cb=progress_cb, cancel_event=self.cancel_event)
            except Exception as e:
                res = AnalysisResult(errors=[str(e)])
            with self._lock:
                self.last_result = res
                self.running = False
            self.hub.publish({"type": "done", "result": asdict(res)})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self.hub.publish({"type": "status", "message": "已开始运行"})
        return True

    def stop(self) -> None:
        self.cancel_event.set()
        self.hub.publish({"type": "status", "message": "正在停止..."})

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            res = asdict(self.last_result) if self.last_result is not None else None
            return {
                "running": bool(self.running),
                "progress": dict(self.current_progress),
                "result": res,
            }

    def _get_cleaned_path(self) -> str:
        with self._lock:
            res = self.last_result
            cfg = self.cfg
        if res and getattr(res, "cleaned_path", None):
            return str(res.cleaned_path or "")

        base_dir = os.path.abspath(get_base_dir())
        out_dir = str(cfg.output_dir or "").strip() or "output"
        output_root = os.path.abspath(out_dir) if os.path.isabs(out_dir) else os.path.abspath(os.path.join(base_dir, out_dir))
        if output_root == base_dir:
            output_root = os.path.abspath(os.path.join(output_root, "output"))

        direct = os.path.abspath(os.path.join(output_root, cfg.output_basename))
        if os.path.exists(direct):
            return direct

        try:
            entries = [n for n in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, n))]
        except Exception:
            entries = []
        stamp_dirs = [n for n in entries if re.match(r"^\\d{8}_\\d{6}$", str(n or "").strip())]
        stamp_dirs.sort(reverse=True)
        for d in stamp_dirs:
            p = os.path.abspath(os.path.join(output_root, d, cfg.output_basename))
            if os.path.exists(p):
                return p

        return direct

    def _get_cleaned_sqlite_path(self) -> str:
        with self._lock:
            res = self.last_result
            cfg = self.cfg
        if res and getattr(res, "cleaned_sqlite_path", None):
            return str(getattr(res, "cleaned_sqlite_path") or "")

        xlsx_path = self._get_cleaned_path()
        guess = _cleaned_sqlite_path_for(xlsx_path) if xlsx_path else ""
        if guess and os.path.exists(guess):
            return guess

        base_dir = os.path.abspath(get_base_dir())
        data_root = os.path.abspath(os.path.join(base_dir, "data"))
        stem = os.path.splitext(str(cfg.output_basename or OUTPUT_PATH))[0] or "cleaned"
        candidates: List[str] = []
        try:
            for name in os.listdir(data_root):
                if not str(name).lower().endswith(".sqlite"):
                    continue
                if name == f"{stem}.sqlite" or name.startswith(f"{stem}_"):
                    candidates.append(os.path.join(data_root, name))
        except Exception:
            candidates = []

        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0, reverse=True)
            return os.path.abspath(candidates[0])

        return guess or ""

    def _sqlite_is_available(self) -> bool:
        sp = self._get_cleaned_sqlite_path()
        if not sp or not os.path.exists(sp):
            return False
        xp = self._get_cleaned_path()
        if not xp or not os.path.exists(xp):
            return True
        try:
            return float(os.path.getmtime(sp)) >= float(os.path.getmtime(xp))
        except Exception:
            return True

    def load_cleaned_df(self, force_reload: bool = False) -> pd.DataFrame:
        path = self._get_cleaned_path()
        if not path or not os.path.exists(path):
            raise FileNotFoundError("清洗后的AI标准财务表不存在，请先运行生成")

        mtime = None
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            mtime = None

        with self._lock:
            cache_path = self._cleaned_cache.get("path")
            cache_mtime = self._cleaned_cache.get("mtime")
            cache_df = self._cleaned_cache.get("df")

        if not force_reload and cache_df is not None and cache_path == path and cache_mtime == mtime:
            return cache_df

        df = pd.read_excel(path)
        with self._lock:
            self._cleaned_cache = {"path": path, "mtime": mtime, "df": df}
        return df


def _web_index_html() -> str:
    try:
        p = get_resource_path("web/index.html")
        return p.read_text(encoding="utf-8")
    except Exception:
        return "<!doctype html><html><head><meta charset='utf-8'><title>Web</title></head><body>缺少 web/index.html</body></html>"

def _choose_web_port(host: str, requested_port: int) -> int:
    def try_bind(h: str, p: int) -> Optional[int]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((h, int(p)))
            return int(s.getsockname()[1])
        except Exception:
            return None
        finally:
            try:
                s.close()
            except Exception:
                pass

    rp = int(requested_port or 0)
    if rp <= 0:
        picked = try_bind(host, 0) or try_bind("127.0.0.1", 0)
        return int(picked or 87650)

    if try_bind(host, rp) is not None:
        return rp

    for p in range(rp + 1, rp + 200):
        if try_bind(host, p) is not None:
            print(f"端口 {rp} 被占用，已改用 {p}")
            return p

    picked = try_bind(host, 0) or try_bind("127.0.0.1", 0)
    if picked is None:
        return rp
    print(f"端口 {rp} 被占用，已改用 {picked}")
    return int(picked)


def run_web(config_path: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response
        from fastapi.staticfiles import StaticFiles
    except Exception:
        print("缺少依赖：fastapi。请先安装：py -m pip install fastapi uvicorn")
        return 1

    runner = _WebRunner(config_path=config_path)
    app = FastAPI()

    def _sanitize_json(obj: Any) -> Any:
        import math as _math
        import datetime as _dt

        if obj is None:
            return None
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        if isinstance(obj, float):
            try:
                if _math.isnan(obj) or _math.isinf(obj):
                    return None
            except Exception:
                return None
            return obj
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, (_dt.datetime, _dt.date, pd.Timestamp)):
            return str(obj)
        if isinstance(obj, list):
            return [_sanitize_json(x) for x in obj]
        if isinstance(obj, tuple):
            return [_sanitize_json(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _sanitize_json(v) for k, v in obj.items()}
        try:
            item = getattr(obj, "item", None)
            if callable(item):
                return _sanitize_json(item())
        except Exception:
            pass
        return str(obj)
    try:
        web_dir = get_resource_path("web")
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    except Exception:
        pass

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _web_index_html()

    @app.get("/favicon.ico")
    def favicon():
        p = get_resource_path("app_icon.ico")
        if p.exists() and p.is_file():
            return FileResponse(str(p), media_type="image/x-icon")
        return Response(status_code=404)

    @app.get("/api/config")
    def api_get_config() -> Dict[str, Any]:
        return asdict(runner.get_config())

    @app.post("/api/config")
    def api_set_config(data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = runner.set_config(data)
        return asdict(cfg)

    @app.get("/api/config/queries")
    def api_get_saved_queries() -> Dict[str, Any]:
        return getattr(runner.cfg, "saved_queries", {})

    @app.post("/api/config/queries/{name}")
    def api_save_query(name: str, query: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(runner.cfg, "saved_queries", None):
            setattr(runner.cfg, "saved_queries", {})
        runner.cfg.saved_queries[name] = query
        save_config(runner.config_path, runner.cfg)
        return {"status": "ok", "saved_queries": runner.cfg.saved_queries}

    @app.delete("/api/config/queries/{name}")
    def api_delete_query(name: str) -> Dict[str, Any]:
        sq = getattr(runner.cfg, "saved_queries", {})
        if sq and name in sq:
            del sq[name]
            save_config(runner.config_path, runner.cfg)
        return {"status": "ok", "saved_queries": sq}

    @app.get("/api/rules")
    def api_get_rules() -> Dict[str, Any]:
        try:
            import financial_analyzer_core as core

            rules = core._load_rules()
            path = os.path.join(get_base_dir(), "config", "rules.json")
            return _sanitize_json({"ok": True, "path": path, "rules": rules})
        except Exception as e:
            return {"ok": False, "message": str(e), "rules": {}}

    @app.post("/api/rules")
    def api_save_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import financial_analyzer_core as core

            rules_obj: Any = payload.get("rules") if isinstance(payload, dict) and "rules" in payload else payload
            if not isinstance(rules_obj, dict):
                return {"ok": False, "message": "rules 必须是 JSON 对象(dict)"}

            try:
                rules_obj = core._repair_mojibake_obj(rules_obj)
            except Exception:
                pass

            path = os.path.join(get_base_dir(), "config", "rules.json")
            os.makedirs(os.path.dirname(path) or os.getcwd(), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rules_obj, f, ensure_ascii=False, indent=2)

            try:
                core._RULES_CACHE["mtime"] = None
                core._RULES_CACHE["data"] = None
                core._RULES_CACHE["aliases"] = None
            except Exception:
                pass

            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    @app.post("/api/scan")
    def api_scan() -> Dict[str, Any]:
        files = runner.scan()
        return {"files": files}

    @app.post("/api/run/start")
    def api_start() -> Dict[str, Any]:
        ok = runner.start()
        return {"ok": ok, "message": "已启动" if ok else "任务正在运行中"}

    @app.post("/api/run/stop")
    def api_stop() -> Dict[str, Any]:
        runner.stop()
        return {"ok": True}

    @app.get("/api/run/status")
    def api_status() -> Dict[str, Any]:
        return runner.get_status()

    @app.get("/api/cleaned/options")
    def api_cleaned_options(force: int = 0) -> Dict[str, Any]:
        if runner._sqlite_is_available() and not bool(int(force or 0)):
            sp = runner._get_cleaned_sqlite_path()
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    opts: Dict[str, Any] = {"ok": True}
                    for col in ["源文件", "日期", "报表类型", "大类", "时间属性"]:
                        try:
                            cur = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM cleaned WHERE "{col}" IS NOT NULL AND TRIM(CAST("{col}" AS TEXT)) <> \'\' ORDER BY "{col}" LIMIT 500'
                            )
                            vals = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None and str(r[0]).strip() != ""]
                            opts[col] = vals
                        except Exception:
                            pass
                    try:
                        cur = conn.execute("SELECT COUNT(*) FROM cleaned")
                        row = cur.fetchone()
                        opts["rows"] = int(row[0]) if row else 0
                    except Exception:
                        opts["rows"] = 0
                    opts["path"] = runner._get_cleaned_path()
                    opts["sqlite"] = sp
                    return _sanitize_json(opts)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            df = runner.load_cleaned_df(force_reload=bool(int(force or 0)))
        except Exception as e:
            return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(), "rows": 0}

        opts: Dict[str, Any] = {"ok": True}
        for col in ["源文件", "日期", "报表类型", "大类", "时间属性"]:
            if col in df.columns:
                try:
                    vals = df[col].dropna().astype(str).unique().tolist()
                except Exception:
                    vals = []
                vals = [v for v in vals if str(v).strip() != ""]
                vals.sort()
                opts[col] = vals[:500]
        opts["path"] = runner._get_cleaned_path()
        opts["rows"] = int(len(df))
        return _sanitize_json(opts)

    @app.get("/api/validation/query")
    def api_validation_query(
        min_diff: Optional[float] = None,
        max_diff: Optional[float] = None,
        source_file: str = "",
        is_balanced: str = "",  # "是" or "否" or ""
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not runner.last_result or not runner.last_result.cleaned_sqlite_path:
            return {"status": "error", "message": "No database available"}
        
        db_path = runner.last_result.cleaned_sqlite_path
        if not os.path.exists(db_path):
            return {"status": "error", "message": "Database file not found"}

        try:
            with sqlite3.connect(db_path) as conn:
                # Check if validation table exists
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='validation'")
                if not cursor.fetchone():
                    return {"status": "ok", "data": [], "total": 0}

                where_clauses = []
                params = []

                if min_diff is not None:
                    where_clauses.append("CAST(差额 AS FLOAT) >= ?")
                    params.append(min_diff)
                
                if max_diff is not None:
                    where_clauses.append("CAST(差额 AS FLOAT) <= ?")
                    params.append(max_diff)

                if source_file:
                    where_clauses.append("源文件 LIKE ?")
                    params.append(f"%{source_file}%")

                if is_balanced:
                    where_clauses.append("是否平衡 = ?")
                    params.append(is_balanced)
                else:
                    # Default to showing only unbalanced if not specified? 
                    # No, let frontend decide. But typically we query for exceptions.
                    pass

                where_str = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                # Count total
                count_sql = f"SELECT COUNT(*) FROM validation WHERE {where_str}"
                cursor.execute(count_sql, params)
                total = cursor.fetchone()[0]

                # Query data
                # Sort by difference descending by default
                sql = f"SELECT * FROM validation WHERE {where_str} ORDER BY CAST(差额 AS FLOAT) DESC LIMIT ? OFFSET ?"
                df = pd.read_sql_query(sql, conn, params=params + [limit, offset])
                
                records = df.to_dict(orient="records")
                return {"status": "ok", "data": records, "total": total}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    @app.get("/api/validation/summary")
    def api_validation_summary() -> Dict[str, Any]:
        if not runner.last_result or not runner.last_result.cleaned_sqlite_path:
            return {"status": "error", "message": "No database available"}
        
        db_path = runner.last_result.cleaned_sqlite_path
        if not os.path.exists(db_path):
            return {"status": "error", "message": "Database file not found"}

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='validation'")
                if not cursor.fetchone():
                    return {"status": "ok", "data": []}

                # Aggregate query matching the Excel logic
                # Group by 源文件, 日期, 验证项目
                sql = """
                    SELECT 
                        源文件, 
                        日期, 
                        验证项目, 
                        COUNT(*) as 总条数,
                        SUM(CASE WHEN 是否平衡 = '否' THEN 1 ELSE 0 END) as 不平衡条数,
                        MAX(CAST(差额 AS FLOAT)) as 最大差额,
                        AVG(CAST(差额 AS FLOAT)) as 平均差额
                    FROM validation
                    GROUP BY 源文件, 日期, 验证项目
                    ORDER BY 不平衡条数 DESC, 最大差额 DESC
                """
                df = pd.read_sql_query(sql, conn)
                records = df.to_dict(orient="records")
                return {"status": "ok", "data": records}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    @app.get("/api/cleaned/query")
    def api_cleaned_query(
        q: str = "",
        subject: str = "",
        source_file: str = "",
        date: str = "",
        report_type: str = "",
        category: str = "",
        time_attr: str = "",
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        sort_by: str = "金额",
        sort_dir: str = "desc",
        topn: Optional[int] = None,
        limit: int = 200,
        offset: int = 0,
        group_by: str = "",
    ) -> Dict[str, Any]:
        allowed_cols = {"源文件", "日期", "报表类型", "大类", "科目", "时间属性", "金额", "来源Sheet"}
        if runner._sqlite_is_available():
            sp = runner._get_cleaned_sqlite_path()
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    where: List[str] = []
                    params: List[Any] = []

                    def add_eq(col: str, val: str) -> None:
                        if val is None:
                            return
                        v = str(val).strip()
                        if not v:
                            return
                        where.append(f'"{col}" = ?')
                        params.append(v)

                    add_eq("源文件", source_file)
                    add_eq("日期", date)
                    add_eq("报表类型", report_type)
                    add_eq("大类", category)
                    add_eq("时间属性", time_attr)

                    kw_input = (subject or q or "").strip()
                    if kw_input:
                        import re
                        keywords = [k.strip() for k in re.split(r'[\s,，、;/；|/]+', kw_input) if k.strip()]
                        if keywords:
                            or_clauses = []
                            for k in keywords:
                                or_clauses.append('LOWER(CAST("科目" AS TEXT)) LIKE ?')
                                params.append(f"%{k.lower()}%")
                            if or_clauses:
                                where.append(f"({' OR '.join(or_clauses)})")

                    if min_amount is not None:
                        try:
                            where.append('"金额" >= ?')
                            params.append(float(min_amount))
                        except Exception:
                            pass
                    if max_amount is not None:
                        try:
                            where.append('"金额" <= ?')
                            params.append(float(max_amount))
                        except Exception:
                            pass

                    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
                    
                    s_col = str(sort_by or "").strip()
                    s_dir = str(sort_dir or "").strip().lower()
                    
                    g_cols = [c.strip() for c in (group_by or "").split(",") if c.strip() in allowed_cols]
                    
                    if g_cols:
                        g_str = ", ".join([f'"{c}"' for c in g_cols])
                        base_sql = f'SELECT {g_str}, SUM("金额") as "金额", COUNT(*) as "条数" FROM cleaned' + where_sql + f' GROUP BY {g_str}'
                        count_sql = f'SELECT COUNT(*) FROM (SELECT {g_str} FROM cleaned {where_sql} GROUP BY {g_str})'
                        
                        order_sql = ""
                        if s_col == "金额":
                            order_sql = f' ORDER BY SUM("金额") {"ASC" if s_dir == "asc" else "DESC"}'
                        elif s_col == "条数":
                            order_sql = f' ORDER BY COUNT(*) {"ASC" if s_dir == "asc" else "DESC"}'
                        elif s_col in allowed_cols:
                            order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'
                    else:
                        base_sql = "SELECT * FROM cleaned" + where_sql
                        count_sql = "SELECT COUNT(*) FROM cleaned" + where_sql
                        order_sql = ""
                        if s_col in allowed_cols:
                            order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'

                    total = 0
                    try:
                        cur = conn.execute(count_sql, params)
                        row = cur.fetchone()
                        total = int(row[0]) if row else 0
                    except Exception:
                        total = 0

                    lim = int(max(1, min(int(limit or 200), 2000)))
                    if topn is not None:
                        try:
                            lim = int(max(1, min(int(topn), 2000)))
                        except Exception:
                            pass
                        offset = 0
                    off = int(max(0, int(offset or 0)))
                    rows: List[Dict[str, Any]] = []
                    try:
                        final_sql = base_sql + order_sql + " LIMIT ? OFFSET ?"
                        cur = conn.execute(final_sql, params + [lim, off])
                        cols = [d[0] for d in (cur.description or [])]
                        fetched = cur.fetchall()
                        rows = [dict(zip(cols, r)) for r in fetched]
                    except Exception:
                        rows = []

                    return _sanitize_json(
                        {
                            "ok": True,
                            "path": runner._get_cleaned_path(),
                            "sqlite": sp,
                            "total": int(total),
                            "limit": lim,
                            "offset": off,
                            "rows": rows,
                            "sort_by": s_col,
                            "sort_dir": "asc" if s_dir == "asc" else "desc",
                        }
                    )
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except Exception as e:
                return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(), "total": 0, "limit": 0, "offset": 0, "rows": []}
        try:
            df = runner.load_cleaned_df(force_reload=False)
        except Exception as e:
            return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(), "total": 0, "limit": 0, "offset": 0, "rows": []}
        view = df

        def _filt_eq(col: str, val: str) -> None:
            nonlocal view
            if not val:
                return
            if col in view.columns:
                view = view[view[col].astype(str) == str(val)]

        _filt_eq("源文件", source_file)
        _filt_eq("日期", date)
        _filt_eq("报表类型", report_type)
        _filt_eq("大类", category)
        _filt_eq("时间属性", time_attr)

        kw_input = (subject or q or "").strip()
        if kw_input and "科目" in view.columns:
            import re
            keywords = [k.strip() for k in re.split(r'[\s,，、;/；|/]+', kw_input) if k.strip()]
            if keywords:
                # pandas multiple contains logic (OR)
                # escape regex chars just in case, though user likely inputs plain text
                pattern = "|".join([re.escape(k) for k in keywords])
                view = view[view["科目"].astype(str).str.contains(pattern, case=False, na=False)]

        if min_amount is not None and "金额" in view.columns:
            try:
                view = view[pd.to_numeric(view["金额"], errors="coerce").fillna(0) >= float(min_amount)]
            except Exception:
                pass
        if max_amount is not None and "金额" in view.columns:
            try:
                view = view[pd.to_numeric(view["金额"], errors="coerce").fillna(0) <= float(max_amount)]
            except Exception:
                pass
        
        g_cols = [c.strip() for c in (group_by or "").split(",") if c.strip() in view.columns]
        if g_cols:
            try:
                # Group by
                view = view.assign(
                    金额=pd.to_numeric(view["金额"], errors="coerce").fillna(0)
                ).groupby(g_cols).agg(
                    金额=("金额", "sum"),
                    条数=("金额", "count")
                ).reset_index()
            except Exception:
                pass

        s_col = str(sort_by or "").strip()
        s_dir = str(sort_dir or "").strip().lower()
        if s_col and s_col in view.columns:
            ascending = True if s_dir == "asc" else False
            try:
                if s_col == "金额":
                    view = view.assign(__amt=pd.to_numeric(view["金额"], errors="coerce").fillna(0)).sort_values("__amt", ascending=ascending).drop(columns=["__amt"])
                else:
                    view = view.sort_values(s_col, ascending=ascending)
            except Exception:
                pass

        total = int(len(view))
        lim = int(max(1, min(int(limit or 200), 2000)))
        if topn is not None:
            try:
                lim = int(max(1, min(int(topn), 2000)))
            except Exception:
                pass
            offset = 0
        off = int(max(0, int(offset or 0)))
        page = view.iloc[off : off + lim].copy()
        records = page.to_dict(orient="records")
        return _sanitize_json({
            "ok": True,
            "path": runner._get_cleaned_path(),
            "total": total,
            "limit": lim,
            "offset": off,
            "rows": records,
            "sort_by": s_col,
            "sort_dir": "asc" if s_dir == "asc" else "desc",
        })

    @app.get("/api/cleaned/export")
    def api_cleaned_export(
        q: str = "",
        subject: str = "",
        source_file: str = "",
        date: str = "",
        report_type: str = "",
        category: str = "",
        time_attr: str = "",
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        sort_by: str = "金额",
        sort_dir: str = "desc",
        topn: Optional[int] = None,
        group_by: str = "",
        format: str = "csv",
    ):
        import io, csv
        is_xlsx = (format or "").lower() == "xlsx"
        ext = "xlsx" if is_xlsx else "csv"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if is_xlsx else "text/csv"
        filename = f"cleaned_export.{ext}"

        allowed_cols = {"源文件", "日期", "报表类型", "大类", "科目", "时间属性", "金额", "来源Sheet"}

        if runner._sqlite_is_available():
            sp = runner._get_cleaned_sqlite_path()
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    where: List[str] = []
                    params: List[Any] = []
                    def add_eq(col: str, val: str) -> None:
                        if val is None:
                            return
                        v = str(val).strip()
                        if not v:
                            return
                        where.append(f'"{col}" = ?')
                        params.append(v)
                    add_eq("源文件", source_file)
                    add_eq("日期", date)
                    add_eq("报表类型", report_type)
                    add_eq("大类", category)
                    add_eq("时间属性", time_attr)
                    kw_input = (subject or q or "").strip()
                    if kw_input:
                        import re
                        keywords = [k.strip() for k in re.split(r'[\s,，、;/；|/]+', kw_input) if k.strip()]
                        if keywords:
                            or_clauses = []
                            for k in keywords:
                                or_clauses.append('LOWER(CAST("科目" AS TEXT)) LIKE ?')
                                params.append(f"%{k.lower()}%")
                            if or_clauses:
                                where.append(f"({' OR '.join(or_clauses)})")
                    if min_amount is not None:
                        try:
                            where.append('"金额" >= ?')
                            params.append(float(min_amount))
                        except Exception:
                            pass
                    if max_amount is not None:
                        try:
                            where.append('"金额" <= ?')
                            params.append(float(max_amount))
                        except Exception:
                            pass
                    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
                    
                    s_col = str(sort_by or "").strip()
                    s_dir = str(sort_dir or "").strip().lower()

                    g_cols = [c.strip() for c in (group_by or "").split(",") if c.strip() in allowed_cols]
                    
                    if g_cols:
                        g_str = ", ".join([f'"{c}"' for c in g_cols])
                        base_sql = f'SELECT {g_str}, SUM("金额") as "金额", COUNT(*) as "条数" FROM cleaned' + where_sql + f' GROUP BY {g_str}'
                        
                        order_sql = ""
                        if s_col == "金额":
                            order_sql = f' ORDER BY SUM("金额") {"ASC" if s_dir == "asc" else "DESC"}'
                        elif s_col == "条数":
                            order_sql = f' ORDER BY COUNT(*) {"ASC" if s_dir == "asc" else "DESC"}'
                        elif s_col in allowed_cols:
                            order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'
                    else:
                        base_sql = "SELECT * FROM cleaned" + where_sql
                        order_sql = ""
                        if s_col in allowed_cols:
                            order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'

                    cap = 50000
                    if topn is not None:
                        try:
                            cap = int(max(1, min(int(topn), 50000)))
                        except Exception:
                            pass
                    
                    cur = conn.execute(
                        base_sql + order_sql + " LIMIT ?",
                        params + [cap],
                    )
                    cols = [d[0] for d in (cur.description or [])]
                    fetched = cur.fetchall()
                    
                    if is_xlsx:
                        output = io.BytesIO()
                        try:
                            df = pd.DataFrame([list(r) for r in fetched], columns=cols)
                            df.to_excel(output, index=False)
                            content = output.getvalue()
                        except Exception as e:
                            return Response(str(e), media_type="text/plain", status_code=500)
                    else:
                        output = io.StringIO()
                        writer = csv.writer(output)
                        writer.writerow(cols)
                        for r in fetched:
                            writer.writerow(list(r))
                        content = output.getvalue()
                        output.close()
                    
                    return Response(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except Exception as e:
                return Response(str(e), media_type="text/plain", status_code=500)
        try:
            df = runner.load_cleaned_df(force_reload=False)
        except Exception as e:
            return Response(str(e), media_type="text/plain", status_code=500)
        view = df
        def _filt_eq(col: str, val: str) -> None:
            nonlocal view
            if not val:
                return
            if col in view.columns:
                view = view[view[col].astype(str) == str(val)]
        _filt_eq("源文件", source_file)
        _filt_eq("日期", date)
        _filt_eq("报表类型", report_type)
        _filt_eq("大类", category)
        _filt_eq("时间属性", time_attr)
        kw = (subject or q or "").strip()
        if kw and "科目" in view.columns:
            view = view[view["科目"].astype(str).str.contains(kw, case=False, na=False)]
        if min_amount is not None and "金额" in view.columns:
            try:
                view = view[pd.to_numeric(view["金额"], errors="coerce").fillna(0) >= float(min_amount)]
            except Exception:
                pass
        if max_amount is not None and "金额" in view.columns:
            try:
                view = view[pd.to_numeric(view["金额"], errors="coerce").fillna(0) <= float(max_amount)]
            except Exception:
                pass
        
        g_cols = [c.strip() for c in (group_by or "").split(",") if c.strip() in view.columns]
        if g_cols:
            try:
                # Group by
                view = view.assign(
                    金额=pd.to_numeric(view["金额"], errors="coerce").fillna(0)
                ).groupby(g_cols).agg(
                    金额=("金额", "sum"),
                    条数=("金额", "count")
                ).reset_index()
            except Exception:
                pass

        s_col = str(sort_by or "").strip()
        s_dir = str(sort_dir or "").strip().lower()
        if s_col and s_col in view.columns:
            ascending = True if s_dir == "asc" else False
            try:
                if s_col == "金额":
                    view = view.assign(__amt=pd.to_numeric(view["金额"], errors="coerce").fillna(0)).sort_values("__amt", ascending=ascending).drop(columns=["__amt"])
                else:
                    view = view.sort_values(s_col, ascending=ascending)
            except Exception:
                pass
        cap = 50000
        if topn is not None:
            try:
                cap = int(max(1, min(int(topn), 50000)))
            except Exception:
                pass
        
        if is_xlsx:
            output = io.BytesIO()
            try:
                view.head(cap).to_excel(output, index=False)
                content = output.getvalue()
            finally:
                output.close()
        else:
            output = io.StringIO()
            try:
                view.head(cap).to_csv(output, index=False)
                content = output.getvalue()
            finally:
                output.close()
        return Response(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.post("/api/open")
    def api_open(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = str((payload or {}).get("path", "")).strip()
        if not path or not os.path.exists(path):
            return {"ok": False, "message": "路径不存在"}
        try:
            os.startfile(path)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    @app.get("/api/stream")
    def api_stream():
        import json as _json

        q = runner.hub.subscribe()

        def encode(event: str, data_obj: Any) -> str:
            s = _json.dumps(_sanitize_json(data_obj), ensure_ascii=False, allow_nan=False)
            return f"event: {event}\ndata: {s}\n\n"

        def gen():
            try:
                yield encode("status", {"message": "已连接"})
                while True:
                    try:
                        item = q.get(timeout=0.8)
                    except queue.Empty:
                        st = runner.get_status()
                        yield encode("progress", st.get("progress", {}))
                        continue

                    t = str(item.get("type", "message"))
                    if t == "log":
                        yield encode("log", {"message": item.get("message", "")})
                    elif t == "progress":
                        yield encode("progress", item)
                    elif t == "status":
                        yield encode("status", {"message": item.get("message", "")})
                    elif t == "done":
                        yield encode("done", {"result": item.get("result", {})})
                    else:
                        yield encode("status", {"message": str(item)})
            finally:
                runner.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    chosen_port = _choose_web_port(host, int(port or 0))
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(f"http://{host}:{chosen_port}/")
        except Exception:
            pass

    try:
        import uvicorn
    except Exception:
        print("缺少依赖：uvicorn。请先安装：py -m pip install uvicorn")
        return 1

    uvicorn.run(app, host=host, port=int(chosen_port), log_level="info")
    return 0


if __name__ == "__main__":
    from financial_analyzer_core import DEFAULT_CONFIG_PATH

    raise SystemExit(run_web(DEFAULT_CONFIG_PATH))

