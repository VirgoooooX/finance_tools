import os
import sys
import re
import json
import queue
import threading
import logging
from dataclasses import asdict
from typing import Optional, Any, Dict, List
import sqlite3

import pandas as pd

from fa_platform.paths import get_resource_path
from fa_platform.jsonx import sanitize_json
from fa_platform.webx import choose_web_port, read_web_index_html, sse_encode, get_tool_web_entry_url, get_tool_web_manifest

from financial_analyzer_core import (
    OUTPUT_PATH,
    AppConfig,
    AnalysisResult,
    get_base_dir,
    get_default_tool_id,
    load_config,
    save_config,
    run_tool,
    list_tools,
    get_tool_discovery_errors,
    resolve_tool_id,
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
        self.active_tool_id: str = ""
        self.active_run_id_by_tool: Dict[str, str] = {}

    def _normalize_tool_id(self, tool_id: str = "") -> str:
        tid = str(tool_id or "").strip()
        if tid:
            return tid
        with self._lock:
            tid = str(self.active_tool_id or "").strip()
        if tid:
            return tid
        return str(get_default_tool_id() or "").strip()

    def set_config(self, data: Dict[str, Any], tool_id: str = "") -> AppConfig:
        tid = self._normalize_tool_id(tool_id)
        
        # Load specific tool config path
        # Assuming standard structure: tools/{tid}/config.json
        # If tid is default, we might have logic for legacy paths, but let's standardize on tools/
        
        from financial_analyzer_core import get_base_dir
        config_path = os.path.join(get_base_dir(), "tools", tid, "config.json") if tid else self.config_path
        if not os.path.exists(config_path):
            config_path = self.config_path

        cfg = load_config(config_path)
        # Update cfg with data
        for k, v in (data or {}).items():
            if k == "tool_id":
                continue
            if k == "tool_params":
                if isinstance(v, dict):
                    setattr(cfg, "tool_params", v)
                continue
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                continue
            tp = getattr(cfg, "tool_params", None)
            if not isinstance(tp, dict):
                cfg.tool_params = {}
                tp = cfg.tool_params
            bucket = tp.get(tid)
            if not isinstance(bucket, dict):
                bucket = {}
            bucket[k] = v
            tp[tid] = bucket
        
        # Ensure critical defaults
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
            
        # Tool ID persistence
        setattr(cfg, "tool_id", tid)

        with self._lock:
            # Only update self.cfg if it's the current running config or we decide to swap?
            # Actually self.cfg tracks the "active" config for the runner.
            # If user saves config for another tool, we just save to disk.
            # But if user is operating on the current tool, we update memory.
            # For simplicity, if tid matches current runner's target, update memory.
            # But runner doesn't really know "target" until start() is called.
            # Let's just update self.cfg if tid matches default or currently loaded.
            # For now, just save to disk. The runner reloads config on start() anyway?
            # Wait, runner.start() takes cfg or reloads it? 
            # runner.start() calls run_tool(tid, cfg...).
            # So we should return the updated cfg object.
            pass
            
        save_config(config_path, cfg)
        # Only publish status if it's the active tool?
        self.hub.publish({"type": "status", "message": f"已保存配置 ({tid})"})
        return cfg

    def get_config(self, tool_id: str = "") -> AppConfig:
        tid = self._normalize_tool_id(tool_id)
            
        from financial_analyzer_core import get_base_dir
        config_path = os.path.join(get_base_dir(), "tools", tid, "config.json") if tid else self.config_path
        if not os.path.exists(config_path):
            config_path = self.config_path
             
        return load_config(config_path)

    def scan(self) -> List[str]:
        cfg = self.get_config()
        files = _list_excel_files(cfg)
        return files

    def start(self, tool_id: str = "") -> bool:
        requested_tid = str(tool_id or "").strip()
        if not requested_tid:
            requested_tid = self._normalize_tool_id("")
        tid, used_default = resolve_tool_id(requested_tid)
        with self._lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return False
            self.cancel_event.clear()
            self.running = True
            self.last_result = None
            self.current_progress = {"current": 0, "total": 1, "detail": ""}
            self.active_tool_id = tid

        handler = _WebQueueLogHandler(self.hub)
        logger = _get_logger(handler=handler)

        def progress_cb(stage: str, current: int, total: int, detail: str) -> None:
            with self._lock:
                self.current_progress = {"stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)}
            self.hub.publish({"type": "progress", "stage": stage, "current": int(current), "total": int(total) or 1, "detail": str(detail)})

        def worker() -> None:
            try:
                cfg = self.get_config(tool_id=tid)
                res = run_tool(tid, cfg, logger=logger, progress_cb=progress_cb, cancel_event=self.cancel_event)
            except Exception as e:
                res = AnalysisResult(errors=[str(e)])
            with self._lock:
                self.last_result = res
                self.running = False
                try:
                    rid = str(getattr(res, "run_id", "") or "").strip()
                    if rid:
                        self.active_run_id_by_tool[tid] = rid
                except Exception:
                    pass
            self.hub.publish({"type": "done", "tool_id": tid, "result": asdict(res)})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        if used_default and str(requested_tid or "").strip() and requested_tid != tid:
            self.hub.publish({"type": "status", "message": f"未找到工具：{requested_tid}，已回退到默认工具：{tid}"})
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

    def _get_cleaned_path(self, tool_id: str = "", run_id: str = "") -> str:
        with self._lock:
            res = self.last_result
        tid = self._normalize_tool_id(tool_id)
        if res and getattr(res, "cleaned_path", None) and (not tid or str(getattr(res, "tool_id", "") or "").strip() in ("", tid)):
            return str(res.cleaned_path or "")

        rid = str(run_id or "").strip()
        if not rid:
            with self._lock:
                rid = str(self.active_run_id_by_tool.get(tid, "") or "").strip()
        if tid and rid:
            try:
                from fa_platform.run_index import get_run
                info = get_run(tid, rid)
                if info and info.get("cleaned_path"):
                    return str(info.get("cleaned_path") or "")
            except Exception:
                pass
        if tid and not rid:
            try:
                from fa_platform.run_index import get_latest_run
                info = get_latest_run(tid)
                if info and info.get("cleaned_path"):
                    return str(info.get("cleaned_path") or "")
            except Exception:
                pass

        cfg = self.get_config(tool_id=tid)
        base_dir = os.path.abspath(get_base_dir())
        out_dir = str(cfg.output_dir or "").strip() or "output"
        output_root = os.path.abspath(out_dir) if os.path.isabs(out_dir) else os.path.abspath(os.path.join(base_dir, out_dir))
        if output_root == base_dir:
            output_root = os.path.abspath(os.path.join(output_root, "output"))

        direct = os.path.abspath(os.path.join(output_root, cfg.output_basename))
        if os.path.exists(direct):
            return direct
            
        # Try to find tool subdirectory if exists
        # Assuming tool_id is in config or default
        tid = str(getattr(cfg, "tool_id", "") or tid or "").strip()
        
        # Paths to search:
        # 1. output_root/{timestamp} (Legacy)
        # 2. output_root/{tool_id}/{timestamp} (New)
        
        candidates = []
        
        # Scan legacy
        try:
            entries = [os.path.join(output_root, n) for n in os.listdir(output_root)]
            candidates.extend([e for e in entries if os.path.isdir(e) and re.match(r"^\d{8}_\d{6}$", os.path.basename(e))])
        except Exception:
            pass
            
        # Scan new structure
        try:
            tool_dir = os.path.join(output_root, tid)
            if os.path.isdir(tool_dir):
                entries = [os.path.join(tool_dir, n) for n in os.listdir(tool_dir)]
                candidates.extend([e for e in entries if os.path.isdir(e) and re.match(r"^\d{8}_\d{6}$", os.path.basename(e))])
        except Exception:
            pass
            
        candidates.sort(key=lambda p: os.path.basename(p), reverse=True)
        
        for d in candidates:
            p = os.path.abspath(os.path.join(d, cfg.output_basename))
            if os.path.exists(p):
                return p

        return direct

    def _get_cleaned_sqlite_path(self, tool_id: str = "", run_id: str = "") -> str:
        with self._lock:
            res = self.last_result
        tid = self._normalize_tool_id(tool_id)
        if res and getattr(res, "cleaned_sqlite_path", None) and (not tid or str(getattr(res, "tool_id", "") or "").strip() in ("", tid)):
            return str(getattr(res, "cleaned_sqlite_path") or "")

        rid = str(run_id or "").strip()
        if not rid:
            with self._lock:
                rid = str(self.active_run_id_by_tool.get(tid, "") or "").strip()
        if tid and rid:
            try:
                from fa_platform.run_index import get_run
                info = get_run(tid, rid)
                if info:
                    sp = str(info.get("cleaned_sqlite_path") or "").strip()
                    if sp:
                        return sp
            except Exception:
                pass

        xlsx_path = self._get_cleaned_path(tool_id=tid, run_id=rid)
        guess = _cleaned_sqlite_path_for(xlsx_path) if xlsx_path else ""
        if guess and os.path.exists(guess):
            return guess

        cfg = self.get_config(tool_id=tid)
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

    def _sqlite_is_available(self, tool_id: str = "", run_id: str = "") -> bool:
        sp = self._get_cleaned_sqlite_path(tool_id=tool_id, run_id=run_id)
        if not sp or not os.path.exists(sp):
            return False
        xp = self._get_cleaned_path(tool_id=tool_id, run_id=run_id)
        if not xp or not os.path.exists(xp):
            return True
        try:
            return float(os.path.getmtime(sp)) >= float(os.path.getmtime(xp))
        except Exception:
            return True

    def load_cleaned_df(self, tool_id: str = "", run_id: str = "", force_reload: bool = False) -> pd.DataFrame:
        path = self._get_cleaned_path(tool_id=tool_id, run_id=run_id)
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


def run_web(config_path: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, Response, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except Exception:
        print("缺少依赖：fastapi。请先安装：py -m pip install fastapi uvicorn")
        return 1

    runner = _WebRunner(config_path=config_path)
    app = FastAPI()
    try:
        web_dir = get_resource_path("web")
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    except Exception:
        pass

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return read_web_index_html()

    @app.get("/favicon.ico")
    def favicon():
        p = get_resource_path("app_icon.ico")
        if p.exists() and p.is_file():
            return FileResponse(str(p), media_type="image/x-icon")
        return Response(status_code=404)

    @app.get("/api/config")
    def api_get_config(tool_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        cfg = runner.get_config(tool_id=tool_id)
        d = asdict(cfg)
        tp = d.get("tool_params")
        if isinstance(tp, dict):
            bucket = tp.get(tid)
            if isinstance(bucket, dict):
                for k, v in bucket.items():
                    if k not in d:
                        d[k] = v
        return sanitize_json(d)

    @app.post("/api/config")
    def api_set_config(data: Dict[str, Any], tool_id: str = "") -> Dict[str, Any]:
        # tool_id can be in query param or body, let's prefer query param for consistency
        # but fastAPI handles query param automatically if defined in args.
        cfg = runner.set_config(data, tool_id=tool_id)
        tid = runner._normalize_tool_id(tool_id)
        d = asdict(cfg)
        tp = d.get("tool_params")
        if isinstance(tp, dict):
            bucket = tp.get(tid)
            if isinstance(bucket, dict):
                for k, v in bucket.items():
                    if k not in d:
                        d[k] = v
        return sanitize_json(d)

    @app.get("/api/config/queries")
    def api_get_saved_queries(tool_id: str = "") -> Dict[str, Any]:
        cfg = runner.get_config(tool_id=tool_id)
        return getattr(cfg, "saved_queries", {})

    @app.post("/api/config/queries/{name}")
    def api_save_query(name: str, query: Dict[str, Any], tool_id: str = "") -> Dict[str, Any]:
        cfg = runner.get_config(tool_id=tool_id)
        if not getattr(cfg, "saved_queries", None):
            setattr(cfg, "saved_queries", {})
        cfg.saved_queries[name] = query
        
        # Save back
        from financial_analyzer_core import get_base_dir
        tid = runner._normalize_tool_id(tool_id)
        config_path = os.path.join(get_base_dir(), "tools", tid, "config.json")
        if not os.path.exists(config_path):
            config_path = runner.config_path
        
        save_config(config_path, cfg)
        return {"status": "ok", "saved_queries": cfg.saved_queries}

    @app.delete("/api/config/queries/{name}")
    def api_delete_query(name: str, tool_id: str = "") -> Dict[str, Any]:
        cfg = runner.get_config(tool_id=tool_id)
        sq = getattr(cfg, "saved_queries", {})
        if sq and name in sq:
            del sq[name]
            
            from financial_analyzer_core import get_base_dir
            tid = runner._normalize_tool_id(tool_id)
            config_path = os.path.join(get_base_dir(), "tools", tid, "config.json")
            if not os.path.exists(config_path):
                config_path = runner.config_path
                
            save_config(config_path, cfg)
        return {"status": "ok", "saved_queries": sq}

    @app.get("/api/rules")
    def api_get_rules(tool_id: str = "") -> Dict[str, Any]:
        try:
            import financial_analyzer_core as core
            from financial_analyzer_core import get_base_dir
            tid = runner._normalize_tool_id(tool_id)

            # Try to determine the path used by _load_rules, or construct it
            path = os.path.join(get_base_dir(), "tools", tid, "rules.json")
            if not os.path.exists(path):
                path = os.path.join(get_base_dir(), "config", "rules.json")
            
            # Manual load to ensure we get the specific file
            rules = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        rules = json.load(f) or {}
                except Exception:
                    pass
                
            return sanitize_json({"ok": True, "path": path, "rules": rules})
        except Exception as e:
            return {"ok": False, "message": str(e), "rules": {}}

    @app.post("/api/rules")
    def api_save_rules(payload: Dict[str, Any], tool_id: str = "") -> Dict[str, Any]:
        try:
            import financial_analyzer_core as core
            from financial_analyzer_core import get_base_dir
            tid = runner._normalize_tool_id(tool_id)

            rules_obj: Any = payload.get("rules") if isinstance(payload, dict) and "rules" in payload else payload
            if not isinstance(rules_obj, dict):
                return {"ok": False, "message": "rules 必须是 JSON 对象(dict)"}

            try:
                rules_obj = core._repair_mojibake_obj(rules_obj)
            except Exception:
                pass

            path = os.path.join(get_base_dir(), "tools", tid, "rules.json")
            # If default tool and old path exists, maybe update old path? 
            # But we want to migrate to tools folder. So we force tools folder for new saves unless it fails.
            
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

    @app.get("/api/tools")
    def api_tools() -> Dict[str, Any]:
        return {"tools": list_tools(), "default": get_default_tool_id(), "errors": get_tool_discovery_errors()}

    @app.post("/api/tool/select")
    def api_tool_select(tool_id: str = "") -> Dict[str, Any]:
        tid, _ = resolve_tool_id(tool_id)
        with runner._lock:
            runner.active_tool_id = tid
        return {"ok": True, "tool_id": tid}

    @app.get("/api/runs")
    def api_runs(tool_id: str = "", limit: int = 50) -> Dict[str, Any]:
        from fa_platform.run_index import list_runs
        tid = runner._normalize_tool_id(tool_id)
        return sanitize_json({"ok": True, "tool_id": tid, "runs": list_runs(tid, limit=limit)})

    @app.get("/api/run/active")
    def api_run_active(tool_id: str = "") -> Dict[str, Any]:
        from fa_platform.run_index import get_latest_run, get_run
        tid = runner._normalize_tool_id(tool_id)
        with runner._lock:
            rid = str(runner.active_run_id_by_tool.get(tid, "") or "").strip()
        info = get_run(tid, rid) if tid and rid else None
        if info is None and tid:
            info = get_latest_run(tid)
            if info:
                with runner._lock:
                    runner.active_run_id_by_tool[tid] = str(info.get("run_id") or "").strip()
        return sanitize_json({"ok": True, "tool_id": tid, "run": info})

    @app.post("/api/run/select")
    def api_run_select(tool_id: str = "", run_id: str = "") -> Dict[str, Any]:
        from fa_platform.run_index import get_run
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        info = get_run(tid, rid) if tid and rid else None
        if info:
            with runner._lock:
                runner.active_run_id_by_tool[tid] = rid
            return {"ok": True, "tool_id": tid, "run_id": rid}
        return {"ok": False, "tool_id": tid, "run_id": rid, "message": "未找到该运行记录"}

    @app.get("/api/tools/{tool_id}/web")
    def api_tool_web(tool_id: str) -> Dict[str, Any]:
        manifest, m_err = get_tool_web_manifest(tool_id)
        entry_url, e_err = get_tool_web_entry_url(tool_id)
        msg = str(m_err or e_err or "").strip()
        return sanitize_json(
            {
                "ok": bool(entry_url) and not bool(msg),
                "tool_id": str(tool_id or "").strip(),
                "entry_url": entry_url,
                "manifest": manifest or {},
                "message": msg,
            }
        )

    @app.get("/tools/{tool_id}/web")
    @app.get("/tools/{tool_id}/web/")
    def tool_web_root(tool_id: str):
        entry_url, err = get_tool_web_entry_url(tool_id)
        if err:
            return Response(content=str(err), media_type="text/plain; charset=utf-8", status_code=400)
        if not entry_url:
            return Response(status_code=404)
        return RedirectResponse(url=entry_url)

    @app.get("/tools/{tool_id}/web/{resource_path:path}")
    def tool_web_resource(tool_id: str, resource_path: str):
        tid = str(tool_id or "").strip()
        if not tid:
            return Response(status_code=404)

        base = get_resource_path(f"tools/{tid}/web")
        if not base.exists() or not base.is_dir():
            return Response(status_code=404)

        rel = str(resource_path or "").lstrip("/")
        if not rel:
            rel = "index.html"

        try:
            base_resolved = base.resolve()
        except Exception:
            base_resolved = base

        try:
            target = (base_resolved / rel).resolve()
        except Exception:
            target = base_resolved / rel

        try:
            if not target.is_relative_to(base_resolved):
                return Response(status_code=404)
        except Exception:
            return Response(status_code=404)

        if not target.exists() or not target.is_file():
            return Response(status_code=404)
        return FileResponse(str(target))

    @app.post("/api/run/start")
    def api_start(tool_id: str = "") -> Dict[str, Any]:
        ok = runner.start(tool_id=tool_id)
        return {"ok": ok, "message": "已启动" if ok else "任务正在运行中"}

    @app.post("/api/run/stop")
    def api_stop() -> Dict[str, Any]:
        runner.stop()
        return {"ok": True}

    @app.get("/api/run/status")
    def api_status() -> Dict[str, Any]:
        return runner.get_status()

    @app.get("/api/cleaned/options")
    def api_cleaned_options(tool_id: str = "", run_id: str = "", force: int = 0) -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        if runner._sqlite_is_available(tool_id=tid, run_id=rid) and not bool(int(force or 0)):
            sp = runner._get_cleaned_sqlite_path(tool_id=tid, run_id=rid)
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    cols: List[str] = []
                    try:
                        cur = conn.execute("PRAGMA table_info(cleaned)")
                        cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                    except Exception:
                        cols = []

                    opts: Dict[str, Any] = {"ok": True, "tool_id": tid, "run_id": rid, "columns": cols}
                    for col in ["源文件", "日期", "报表类型", "大类", "时间属性"]:
                        if cols and col not in cols:
                            continue
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
                    opts["path"] = runner._get_cleaned_path(tool_id=tid, run_id=rid)
                    opts["sqlite"] = sp
                    return sanitize_json(opts)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            df = runner.load_cleaned_df(tool_id=tid, run_id=rid, force_reload=bool(int(force or 0)))
        except Exception as e:
            return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(tool_id=tid, run_id=rid), "rows": 0, "tool_id": tid, "run_id": rid}

        opts: Dict[str, Any] = {"ok": True, "tool_id": tid, "run_id": rid, "columns": [str(c) for c in df.columns]}
        for col in ["源文件", "日期", "报表类型", "大类", "时间属性"]:
            if col in df.columns:
                try:
                    vals = df[col].dropna().astype(str).unique().tolist()
                except Exception:
                    vals = []
                vals = [v for v in vals if str(v).strip() != ""]
                vals.sort()
                opts[col] = vals[:500]
        opts["path"] = runner._get_cleaned_path(tool_id=tid, run_id=rid)
        opts["rows"] = int(len(df))
        return sanitize_json(opts)

    @app.get("/api/validation/query")
    def api_validation_query(
        tool_id: str = "",
        run_id: str = "",
        min_diff: Optional[float] = None,
        max_diff: Optional[float] = None,
        source_file: str = "",
        is_balanced: str = "",  # "是" or "否" or ""
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        db_path = runner._get_cleaned_sqlite_path(tool_id=tid, run_id=rid)
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
    def api_validation_summary(tool_id: str = "", run_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        db_path = runner._get_cleaned_sqlite_path(tool_id=tid, run_id=rid)
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
        tool_id: str = "",
        run_id: str = "",
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
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        if runner._sqlite_is_available(tool_id=tid, run_id=rid):
            sp = runner._get_cleaned_sqlite_path(tool_id=tid, run_id=rid)
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    cols: List[str] = []
                    try:
                        cur = conn.execute("PRAGMA table_info(cleaned)")
                        cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                    except Exception:
                        cols = []
                    allowed_cols = set(cols) if cols else {"源文件", "日期", "报表类型", "大类", "科目", "时间属性", "金额", "来源Sheet"}
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
                    if kw_input and ("科目" in allowed_cols):
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
                            if "金额" in allowed_cols:
                                where.append('"金额" >= ?')
                                params.append(float(min_amount))
                        except Exception:
                            pass
                    if max_amount is not None:
                        try:
                            if "金额" in allowed_cols:
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

                    return sanitize_json(
                        {
                            "ok": True,
                            "tool_id": tid,
                            "run_id": rid,
                            "path": runner._get_cleaned_path(tool_id=tid, run_id=rid),
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
                return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(tool_id=tid, run_id=rid), "tool_id": tid, "run_id": rid, "total": 0, "limit": 0, "offset": 0, "rows": []}
        try:
            df = runner.load_cleaned_df(tool_id=tid, run_id=rid, force_reload=False)
        except Exception as e:
            return {"ok": False, "message": str(e), "path": runner._get_cleaned_path(tool_id=tid, run_id=rid), "tool_id": tid, "run_id": rid, "total": 0, "limit": 0, "offset": 0, "rows": []}
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
        return sanitize_json({
            "ok": True,
            "tool_id": tid,
            "run_id": rid,
            "path": runner._get_cleaned_path(tool_id=tid, run_id=rid),
            "total": total,
            "limit": lim,
            "offset": off,
            "rows": records,
            "sort_by": s_col,
            "sort_dir": "asc" if s_dir == "asc" else "desc",
        })

    @app.get("/api/cleaned/export")
    def api_cleaned_export(
        tool_id: str = "",
        run_id: str = "",
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

        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()

        if runner._sqlite_is_available(tool_id=tid, run_id=rid):
            sp = runner._get_cleaned_sqlite_path(tool_id=tid, run_id=rid)
            try:
                conn = sqlite3.connect(sp, check_same_thread=False)
                try:
                    cols: List[str] = []
                    try:
                        cur = conn.execute("PRAGMA table_info(cleaned)")
                        cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                    except Exception:
                        cols = []
                    allowed_cols = set(cols) if cols else {"源文件", "日期", "报表类型", "大类", "科目", "时间属性", "金额", "来源Sheet"}
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
                    if kw_input and ("科目" in allowed_cols):
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
                            if "金额" in allowed_cols:
                                where.append('"金额" >= ?')
                                params.append(float(min_amount))
                        except Exception:
                            pass
                    if max_amount is not None:
                        try:
                            if "金额" in allowed_cols:
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
            df = runner.load_cleaned_df(tool_id=tid, run_id=rid, force_reload=False)
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
        q = runner.hub.subscribe()

        def gen():
            try:
                yield sse_encode("status", {"message": "已连接"})
                while True:
                    try:
                        item = q.get(timeout=0.8)
                    except queue.Empty:
                        st = runner.get_status()
                        yield sse_encode("progress", st.get("progress", {}))
                        continue

                    t = str(item.get("type", "message"))
                    if t == "log":
                        yield sse_encode("log", {"message": item.get("message", "")})
                    elif t == "progress":
                        yield sse_encode("progress", item)
                    elif t == "status":
                        yield sse_encode("status", {"message": item.get("message", "")})
                    elif t == "done":
                        yield sse_encode("done", {"tool_id": item.get("tool_id", ""), "result": item.get("result", {})})
                    else:
                        yield sse_encode("status", {"message": str(item)})
            finally:
                runner.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    chosen_port = choose_web_port(host, int(port or 0))
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

