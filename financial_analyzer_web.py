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

from fa_platform.paths import get_resource_path, ensure_dir as _ensure_dir, default_data_root as _default_data_root
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
        tid = runner._normalize_tool_id(tool_id)
        lim = int(max(1, min(int(limit or 50), 500)))
        base_dir = os.path.abspath(get_base_dir())
        sp = os.path.abspath(os.path.join(base_dir, "data", "warehouse.sqlite"))
        if not sp or not os.path.exists(sp):
            return sanitize_json({"ok": True, "tool_id": tid, "runs": []})

        try:
            with sqlite3.connect(sp) as conn:
                if tid == "report_ingestor":
                    try:
                        cur = conn.execute(
                            """
                            SELECT batch_id, created_at, mode
                            FROM import_batch
                            WHERE tool_id = ?
                            ORDER BY created_at DESC, batch_id DESC
                            LIMIT ?
                            """,
                            (tid, lim),
                        )
                        rows = cur.fetchall() or []
                    except Exception:
                        rows = []
                    runs = [
                        {
                            "tool_id": tid,
                            "run_id": str(r[0] or "").strip(),
                            "started_at": str(r[1] or "").strip(),
                            "finished_at": str(r[1] or "").strip(),
                            "status": str(r[2] or "ok").strip() or "ok",
                            "cleaned_rows": 0,
                            "processed_files": 0,
                            "errors": [],
                            "meta": {},
                        }
                        for r in rows
                        if r and str(r[0] or "").strip()
                    ]
                    return sanitize_json({"ok": True, "tool_id": tid, "runs": runs})

                if tid == "validation_report":
                    try:
                        cur = conn.execute(
                            """
                            SELECT DISTINCT source_run_id
                            FROM warehouse_validation
                            WHERE tool_id = ?
                            ORDER BY source_run_id DESC
                            LIMIT ?
                            """,
                            (tid, lim),
                        )
                        rows = cur.fetchall() or []
                    except Exception:
                        rows = []
                    runs = [
                        {
                            "tool_id": tid,
                            "run_id": str(r[0] or "").strip(),
                            "started_at": str(r[0] or "").strip(),
                            "finished_at": str(r[0] or "").strip(),
                            "status": "ok",
                            "cleaned_rows": 0,
                            "processed_files": 0,
                            "errors": [],
                            "meta": {},
                        }
                        for r in rows
                        if r and str(r[0] or "").strip()
                    ]
                    return sanitize_json({"ok": True, "tool_id": tid, "runs": runs})

                if tid == "financial_metrics":
                    try:
                        cur = conn.execute(
                            """
                            SELECT DISTINCT source_run_id
                            FROM warehouse_metrics
                            WHERE tool_id = ?
                            ORDER BY source_run_id DESC
                            LIMIT ?
                            """,
                            (tid, lim),
                        )
                        rows = cur.fetchall() or []
                    except Exception:
                        rows = []
                    runs = [
                        {
                            "tool_id": tid,
                            "run_id": str(r[0] or "").strip(),
                            "started_at": str(r[0] or "").strip(),
                            "finished_at": str(r[0] or "").strip(),
                            "status": "ok",
                            "cleaned_rows": 0,
                            "processed_files": 0,
                            "errors": [],
                            "meta": {},
                        }
                        for r in rows
                        if r and str(r[0] or "").strip()
                    ]
                    return sanitize_json({"ok": True, "tool_id": tid, "runs": runs})
        except Exception:
            pass
        return sanitize_json({"ok": True, "tool_id": tid, "runs": []})

    @app.get("/api/run/active")
    def api_run_active(tool_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        with runner._lock:
            rid = str(runner.active_run_id_by_tool.get(tid, "") or "").strip()
        if rid:
            return sanitize_json({"ok": True, "tool_id": tid, "run": {"tool_id": tid, "run_id": rid}})
        try:
            data = api_runs(tool_id=tid, limit=1)
            runs = data.get("runs") if isinstance(data, dict) else None
            info = runs[0] if isinstance(runs, list) and runs else None
            if info and info.get("run_id"):
                with runner._lock:
                    runner.active_run_id_by_tool[tid] = str(info.get("run_id") or "").strip()
            return sanitize_json({"ok": True, "tool_id": tid, "run": info})
        except Exception:
            return sanitize_json({"ok": True, "tool_id": tid, "run": None})

    @app.post("/api/run/select")
    def api_run_select(tool_id: str = "", run_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        rid = str(run_id or "").strip()
        if rid:
            with runner._lock:
                runner.active_run_id_by_tool[tid] = rid
            return {"ok": True, "tool_id": tid, "run_id": rid}
        return {"ok": False, "tool_id": tid, "run_id": rid, "message": "run_id 不能为空"}

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
        headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}
        return FileResponse(str(target), headers=headers)

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
        out = api_warehouse_options(batch_id="")
        if isinstance(out, dict):
            out["tool_id"] = tid
            out["run_id"] = rid
        return sanitize_json(out)

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
        bid = str(run_id or "").strip()
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"status": "error", "message": "累计库不存在，请先运行清洗落库"}
        if not bid:
            return {"status": "ok", "data": [], "total": 0}

        lim = int(max(1, min(int(limit or 200), 5000)))
        off = int(max(0, int(offset or 0)))

        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_validation'")
                if not cur.fetchone():
                    return {"status": "ok", "data": [], "total": 0}

                try:
                    cur = conn.execute("SELECT MAX(run_id) FROM warehouse_validation WHERE source_run_id = ?", (bid,))
                    row = cur.fetchone()
                    v_run_id = str(row[0] or "").strip() if row else ""
                except Exception:
                    v_run_id = ""
                if not v_run_id:
                    return {"status": "ok", "data": [], "total": 0}

                where: List[str] = ["source_run_id = ?", "run_id = ?"]
                params: List[Any] = [bid, v_run_id]

                if min_diff is not None:
                    where.append("差额 >= ?")
                    params.append(float(min_diff))
                if max_diff is not None:
                    where.append("差额 <= ?")
                    params.append(float(max_diff))
                if source_file:
                    where.append("源文件 LIKE ?")
                    params.append(f"%{str(source_file).strip()}%")
                if is_balanced:
                    where.append("是否平衡 = ?")
                    params.append(str(is_balanced).strip())

                where_str = " AND ".join(where) if where else "1=1"
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM warehouse_validation WHERE {where_str}", params)
                    row = cur.fetchone()
                    total = int(row[0]) if row else 0
                except Exception:
                    total = 0

                sql = f"""
                    SELECT 源文件, 来源Sheet, 期间, 验证项目, 时间属性, 差额, 是否平衡, 验证结果
                    FROM warehouse_validation
                    WHERE {where_str}
                    ORDER BY ABS(COALESCE(差额, 0)) DESC
                    LIMIT ? OFFSET ?
                """
                df = pd.read_sql_query(sql, conn, params=params + [lim, off])
                return sanitize_json(
                    {
                        "status": "ok",
                        "data": df.to_dict(orient="records"),
                        "total": total,
                        "tool_id": tid,
                        "run_id": bid,
                        "batch_id": bid,
                        "validation_run_id": v_run_id,
                    }
                )
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"status": "error", "message": str(e)}


    @app.get("/api/validation/summary")
    def api_validation_summary(tool_id: str = "", run_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        bid = str(run_id or "").strip()
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"status": "error", "message": "累计库不存在，请先运行清洗落库"}
        if not bid:
            return {"status": "ok", "data": []}

        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_validation'")
                if not cur.fetchone():
                    return {"status": "ok", "data": []}

                try:
                    cur = conn.execute("SELECT MAX(run_id) FROM warehouse_validation WHERE source_run_id = ?", (bid,))
                    row = cur.fetchone()
                    v_run_id = str(row[0] or "").strip() if row else ""
                except Exception:
                    v_run_id = ""
                if not v_run_id:
                    return {"status": "ok", "data": []}

                sql = """
                    SELECT
                        源文件,
                        期间,
                        验证项目,
                        COUNT(*) as 总条数,
                        SUM(CASE WHEN 是否平衡 = '否' THEN 1 ELSE 0 END) as 不平衡条数,
                        MAX(ABS(COALESCE(差额, 0))) as 最大差额,
                        AVG(ABS(COALESCE(差额, 0))) as 平均差额
                    FROM warehouse_validation
                    WHERE source_run_id = ? AND run_id = ?
                    GROUP BY 源文件, 期间, 验证项目
                    ORDER BY 不平衡条数 DESC, 最大差额 DESC
                """
                df = pd.read_sql_query(sql, conn, params=[bid, v_run_id])
                return sanitize_json({"status": "ok", "data": df.to_dict(orient="records"), "tool_id": tid, "run_id": bid, "batch_id": bid, "validation_run_id": v_run_id})
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"status": "error", "message": str(e)}


    @app.get("/api/metrics/summary")
    def api_metrics_summary(tool_id: str = "", run_id: str = "") -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        bid = str(run_id or "").strip()
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"ok": False, "message": "累计库不存在，请先运行清洗落库", "tool_id": tid, "run_id": bid, "path": sp}
        if not bid:
            return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "rows": 0, "columns": [], "metric_columns": []}

        try:
            import json as _json
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_metrics'")
                if not cur.fetchone():
                    return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "rows": 0, "columns": [], "metric_columns": []}

                try:
                    cur = conn.execute("SELECT MAX(run_id) FROM warehouse_metrics WHERE source_run_id = ?", (bid,))
                    row = cur.fetchone()
                    m_run_id = str(row[0] or "").strip() if row else ""
                except Exception:
                    m_run_id = ""
                if not m_run_id:
                    return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "rows": 0, "columns": [], "metric_columns": []}

                try:
                    cur = conn.execute("SELECT COUNT(*) FROM warehouse_metrics WHERE source_run_id = ? AND run_id = ?", (bid, m_run_id))
                    row = cur.fetchone()
                    rows = int(row[0]) if row else 0
                except Exception:
                    rows = 0

                metric_keys: set = set()
                try:
                    cur = conn.execute(
                        "SELECT metrics_json FROM warehouse_metrics WHERE source_run_id = ? AND run_id = ? LIMIT 2000",
                        (bid, m_run_id),
                    )
                    samples = cur.fetchall() or []
                except Exception:
                    samples = []
                for r in samples:
                    raw = str(r[0] or "").strip() if r else ""
                    if not raw:
                        continue
                    try:
                        obj = _json.loads(raw)
                    except Exception:
                        obj = None
                    if isinstance(obj, dict):
                        for k in obj.keys():
                            kk = str(k).strip()
                            if kk:
                                metric_keys.add(kk)

                metric_cols = sorted(metric_keys)
                cols = ["源文件", "期间"] + metric_cols
                return sanitize_json({"ok": True, "tool_id": tid, "run_id": bid, "batch_id": bid, "metrics_run_id": m_run_id, "path": sp, "rows": rows, "columns": cols, "metric_columns": metric_cols})
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"ok": False, "message": str(e), "tool_id": tid, "run_id": bid, "path": sp}


    @app.get("/api/metrics/query")
    def api_metrics_query(
        tool_id: str = "",
        run_id: str = "",
        source_file: str = "",
        period: str = "",
        sort_by: str = "",
        sort_dir: str = "desc",
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        tid = runner._normalize_tool_id(tool_id)
        bid = str(run_id or "").strip()
        sp = _warehouse_path()
        lim = int(max(1, min(int(limit or 200), 5000)))
        off = int(max(0, int(offset or 0)))
        s_dir = "asc" if str(sort_dir or "").lower() == "asc" else "desc"

        if not sp or not os.path.exists(sp):
            return {"ok": False, "message": "累计库不存在，请先运行清洗落库", "tool_id": tid, "run_id": bid, "path": sp, "total": 0, "limit": lim, "offset": off, "rows": [], "columns": []}
        if not bid:
            return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "total": 0, "limit": lim, "offset": off, "rows": [], "columns": []}

        try:
            import json as _json
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse_metrics'")
                if not cur.fetchone():
                    return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "total": 0, "limit": lim, "offset": off, "rows": [], "columns": []}

                try:
                    cur = conn.execute("SELECT MAX(run_id) FROM warehouse_metrics WHERE source_run_id = ?", (bid,))
                    row = cur.fetchone()
                    m_run_id = str(row[0] or "").strip() if row else ""
                except Exception:
                    m_run_id = ""
                if not m_run_id:
                    return {"ok": True, "tool_id": tid, "run_id": bid, "path": sp, "total": 0, "limit": lim, "offset": off, "rows": [], "columns": []}

                where: List[str] = ["source_run_id = ?", "run_id = ?"]
                params: List[Any] = [bid, m_run_id]
                if source_file:
                    where.append("源文件 LIKE ?")
                    params.append(f"%{str(source_file).strip()}%")
                if period:
                    where.append("期间 = ?")
                    params.append(str(period).strip())
                where_str = " AND ".join(where) if where else "1=1"

                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM warehouse_metrics WHERE {where_str}", params)
                    row = cur.fetchone()
                    total = int(row[0]) if row else 0
                except Exception:
                    total = 0

                try:
                    cur = conn.execute(f"SELECT 源文件, 期间, metrics_json FROM warehouse_metrics WHERE {where_str}", params)
                    raw_rows = cur.fetchall() or []
                except Exception:
                    raw_rows = []

                expanded: List[Dict[str, Any]] = []
                metric_keys: set = set()
                for r in raw_rows:
                    if not r or len(r) < 3:
                        continue
                    rec: Dict[str, Any] = {"源文件": r[0], "期间": r[1]}
                    mj = str(r[2] or "").strip()
                    if mj:
                        try:
                            obj = _json.loads(mj)
                        except Exception:
                            obj = None
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                kk = str(k).strip()
                                if not kk:
                                    continue
                                metric_keys.add(kk)
                                rec[kk] = v
                    expanded.append(rec)

                cols = ["源文件", "期间"] + sorted(metric_keys)

                def _to_float(v: Any) -> Optional[float]:
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        try:
                            return float(v)
                        except Exception:
                            return None
                    s = str(v).strip()
                    if not s:
                        return None
                    try:
                        return float(s)
                    except Exception:
                        return None

                s_col = str(sort_by or "").strip()
                if s_col and s_col not in set(cols):
                    s_col = ""

                if s_col:
                    if s_col in {"源文件", "期间"}:
                        expanded.sort(key=lambda x: str(x.get(s_col, "") or ""), reverse=(s_dir == "desc"))
                    else:
                        expanded.sort(
                            key=lambda x: (_to_float(x.get(s_col)) is None, _to_float(x.get(s_col)) if _to_float(x.get(s_col)) is not None else 0.0),
                            reverse=(s_dir == "desc"),
                        )
                else:
                    if "期间" in set(cols) and "源文件" in set(cols):
                        expanded.sort(key=lambda x: (str(x.get("期间", "") or ""), str(x.get("源文件", "") or "")), reverse=True)

                page = expanded[off : off + lim]
                return sanitize_json({"ok": True, "tool_id": tid, "run_id": bid, "batch_id": bid, "metrics_run_id": m_run_id, "path": sp, "total": total, "limit": lim, "offset": off, "rows": page, "columns": cols, "sort_by": s_col, "sort_dir": s_dir})
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"ok": False, "message": str(e), "tool_id": tid, "run_id": bid, "path": sp, "total": 0, "limit": lim, "offset": off, "rows": [], "columns": []}


    @app.get("/api/cleaned/query")
    def api_cleaned_query(
        tool_id: str = "",
        run_id: str = "",
        q: str = "",
        subject: str = "",
        source_file: str = "",
        period: str = "",
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
        out = api_warehouse_query(
            batch_id="",
            q=q,
            subject=subject,
            source_file=source_file,
            report_type=report_type,
            category=category,
            time_attr=time_attr,
            period=period,
            min_amount=min_amount,
            max_amount=max_amount,
            sort_by=sort_by,
            sort_dir=sort_dir,
            topn=topn,
            limit=limit,
            offset=offset,
            group_by=group_by,
        )
        if isinstance(out, dict):
            out["tool_id"] = tid
            out["run_id"] = rid
        return sanitize_json(out)

    @app.get("/api/cleaned/export")
    def api_cleaned_export(
        tool_id: str = "",
        run_id: str = "",
        q: str = "",
        subject: str = "",
        source_file: str = "",
        period: str = "",
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
        rid = str(run_id or "").strip()
        return api_warehouse_export(
            batch_id="",
            q=q,
            subject=subject,
            source_file=source_file,
            report_type=report_type,
            category=category,
            time_attr=time_attr,
            period=period,
            min_amount=min_amount,
            max_amount=max_amount,
            sort_by=sort_by,
            sort_dir=sort_dir,
            topn=topn,
            group_by=group_by,
            format=format,
        )

    def _warehouse_path() -> str:
        base_dir = os.path.abspath(get_base_dir())
        return os.path.abspath(os.path.join(base_dir, "data", "warehouse.sqlite"))

    def _normalize_subject_text(text: Any) -> str:
        s = str(text or "").strip().lower()
        s = s.replace("（", "(").replace("）", ")")
        import re as _re
        s = _re.sub(r"[\s,，]", "", s)
        return s

    @app.get("/api/warehouse/options")
    def api_warehouse_options(batch_id: str = "") -> Dict[str, Any]:
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"ok": False, "message": "累计库不存在，请先运行清洗落库", "path": sp, "rows": 0}
        bid = str(batch_id or "").strip()
        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cols: List[str] = []
                try:
                    cur = conn.execute("PRAGMA table_info(warehouse_cleaned)")
                    cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                except Exception:
                    cols = []

                exclude_cols = {"日期", "金额文本", "值状态"}
                cols = [c for c in cols if c not in exclude_cols]
                out: Dict[str, Any] = {"ok": True, "path": sp, "columns": cols}
                try:
                    if bid:
                        cur = conn.execute("SELECT COUNT(*) FROM warehouse_cleaned WHERE batch_id = ?", (bid,))
                        out["batch_id"] = bid
                    else:
                        cur = conn.execute("SELECT COUNT(*) FROM warehouse_cleaned")
                    row = cur.fetchone()
                    out["rows"] = int(row[0]) if row else 0
                except Exception:
                    out["rows"] = 0

                for col in ["源文件", "报表类型", "大类", "时间属性", "期间", "年份", "报表口径"]:
                    if cols and col not in cols:
                        continue
                    try:
                        if bid:
                            cur = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM warehouse_cleaned WHERE batch_id = ? AND "{col}" IS NOT NULL AND TRIM(CAST("{col}" AS TEXT)) <> \'\' ORDER BY "{col}" LIMIT 500',
                                (bid,),
                            )
                        else:
                            cur = conn.execute(
                                f'SELECT DISTINCT "{col}" FROM warehouse_cleaned WHERE "{col}" IS NOT NULL AND TRIM(CAST("{col}" AS TEXT)) <> \'\' ORDER BY "{col}" LIMIT 500'
                            )
                        out[col] = [r[0] for r in (cur.fetchall() or []) if r and r[0] is not None]
                    except Exception:
                        out[col] = []
                return sanitize_json(out)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"ok": False, "message": str(e), "path": sp, "rows": 0}

    @app.get("/api/warehouse/summary")
    def api_warehouse_summary(batch_id: str = "") -> Dict[str, Any]:
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"ok": False, "message": "累计库不存在，请先运行清洗落库", "path": sp}
        bid = str(batch_id or "").strip()
        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                out: Dict[str, Any] = {"ok": True, "path": sp}
                try:
                    if bid:
                        cur = conn.execute("SELECT COUNT(*) FROM warehouse_cleaned WHERE batch_id = ?", (bid,))
                        out["batch_id"] = bid
                    else:
                        cur = conn.execute("SELECT COUNT(*) FROM warehouse_cleaned")
                    row = cur.fetchone()
                    out["rows"] = int(row[0]) if row else 0
                except Exception:
                    out["rows"] = 0

                try:
                    if bid:
                        cur = conn.execute(
                            """
                            SELECT
                              MIN(期间) as min_period,
                              MAX(期间) as max_period
                            FROM warehouse_cleaned
                            WHERE batch_id = ?
                              AND 期间 IS NOT NULL
                              AND TRIM(CAST(期间 AS TEXT)) <> ''
                              AND LENGTH(TRIM(CAST(期间 AS TEXT))) = 6
                            """,
                            (bid,),
                        )
                    else:
                        cur = conn.execute(
                            """
                            SELECT
                              MIN(期间) as min_period,
                              MAX(期间) as max_period
                            FROM warehouse_cleaned
                            WHERE 期间 IS NOT NULL
                              AND TRIM(CAST(期间 AS TEXT)) <> ''
                              AND LENGTH(TRIM(CAST(期间 AS TEXT))) = 6
                            """
                        )
                    row = cur.fetchone()
                    if row:
                        out["earliest_period"] = str(row[0] or "").strip()
                        out["latest_period"] = str(row[1] or "").strip()
                except Exception:
                    pass

                try:
                    if bid:
                        cur = conn.execute("SELECT created_at FROM import_batch WHERE batch_id = ? LIMIT 1", (bid,))
                        row = cur.fetchone()
                        out["latest_imported_at"] = str(row[0] or "").strip() if row else ""
                    else:
                        cur = conn.execute("SELECT MAX(created_at) FROM import_batch")
                        row = cur.fetchone()
                        out["latest_imported_at"] = str(row[0] or "").strip() if row else ""
                except Exception:
                    out["latest_imported_at"] = ""

                return sanitize_json(out)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            return {"ok": False, "message": str(e), "path": sp}

    @app.get("/api/warehouse/query")
    def api_warehouse_query(
        batch_id: str = "",
        q: str = "",
        subject: str = "",
        source_file: str = "",
        report_type: str = "",
        category: str = "",
        time_attr: str = "",
        period: str = "",
        year: str = "",
        caliber: str = "",
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        sort_by: str = "金额",
        sort_dir: str = "desc",
        topn: Optional[int] = None,
        limit: int = 200,
        offset: int = 0,
        group_by: str = "",
    ) -> Dict[str, Any]:
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return {"ok": False, "message": "累计库不存在，请先运行清洗落库", "path": sp, "total": 0, "limit": 0, "offset": 0, "rows": []}
        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cols: List[str] = []
                try:
                    cur = conn.execute("PRAGMA table_info(warehouse_cleaned)")
                    cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                except Exception:
                    cols = []
                cols = [c for c in cols if c not in {"日期", "金额文本", "值状态"}]
                allowed_cols = set(cols) if cols else {
                    "源文件", "来源Sheet", "期间", "年份", "报表口径", "报表类型", "大类", "科目", "科目规范", "时间属性", "金额"
                }

                where: List[str] = []
                params: List[Any] = []

                def add_eq(col: str, val: str) -> None:
                    if val is None:
                        return
                    v = str(val).strip()
                    if not v:
                        return
                    if col not in allowed_cols:
                        return
                    where.append(f'"{col}" = ?')
                    params.append(v)

                add_eq("batch_id", batch_id)
                add_eq("源文件", source_file)
                add_eq("报表类型", report_type)
                add_eq("大类", category)
                add_eq("时间属性", time_attr)
                add_eq("期间", period)
                add_eq("年份", year)
                add_eq("报表口径", caliber)

                kw_input = (subject or q or "").strip()
                if kw_input:
                    import re
                    keywords = [k.strip() for k in re.split(r'[\s,，、;/；|/]+', kw_input) if k.strip()]
                    if keywords:
                        or_clauses: List[str] = []
                        for k in keywords:
                            k_norm = _normalize_subject_text(k)
                            if k_norm and "科目规范" in allowed_cols:
                                or_clauses.append('LOWER(CAST("科目规范" AS TEXT)) LIKE ?')
                                params.append(f"%{k_norm.lower()}%")
                            if "科目" in allowed_cols:
                                or_clauses.append('LOWER(CAST("科目" AS TEXT)) LIKE ?')
                                params.append(f"%{k.lower()}%")
                        if or_clauses:
                            where.append(f"({' OR '.join(or_clauses)})")

                if min_amount is not None and "金额" in allowed_cols:
                    try:
                        where.append('"金额" >= ?')
                        params.append(float(min_amount))
                    except Exception:
                        pass
                if max_amount is not None and "金额" in allowed_cols:
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
                    base_sql = f'SELECT {g_str}, SUM("金额") as "金额", COUNT(*) as "条数" FROM warehouse_cleaned' + where_sql + f' GROUP BY {g_str}'
                    count_sql = f'SELECT COUNT(*) FROM (SELECT {g_str} FROM warehouse_cleaned {where_sql} GROUP BY {g_str})'
                    order_sql = ""
                    if s_col == "金额":
                        order_sql = f' ORDER BY SUM("金额") {"ASC" if s_dir == "asc" else "DESC"}'
                    elif s_col == "条数":
                        order_sql = f' ORDER BY COUNT(*) {"ASC" if s_dir == "asc" else "DESC"}'
                    elif s_col in allowed_cols:
                        order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'
                else:
                    if cols:
                        select_list = ", ".join([f'"{c}"' for c in cols])
                    else:
                        select_list = "*"
                    base_sql = f"SELECT {select_list} FROM warehouse_cleaned" + where_sql
                    count_sql = "SELECT COUNT(*) FROM warehouse_cleaned" + where_sql
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
                    out_cols = [d[0] for d in (cur.description or [])]
                    fetched = cur.fetchall()
                    rows = [dict(zip(out_cols, r)) for r in fetched]
                except Exception:
                    rows = []

                return sanitize_json(
                    {
                        "ok": True,
                        "path": sp,
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
            return {"ok": False, "message": str(e), "path": sp, "total": 0, "limit": 0, "offset": 0, "rows": []}

    @app.get("/api/warehouse/export")
    def api_warehouse_export(
        batch_id: str = "",
        q: str = "",
        subject: str = "",
        source_file: str = "",
        report_type: str = "",
        category: str = "",
        time_attr: str = "",
        period: str = "",
        year: str = "",
        caliber: str = "",
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        sort_by: str = "金额",
        sort_dir: str = "desc",
        topn: Optional[int] = None,
        group_by: str = "",
        format: str = "csv",
    ):
        import io, csv
        sp = _warehouse_path()
        if not sp or not os.path.exists(sp):
            return Response("累计库不存在，请先运行清洗落库", media_type="text/plain", status_code=404)
        is_xlsx = (format or "").lower() == "xlsx"
        ext = "xlsx" if is_xlsx else "csv"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if is_xlsx else "text/csv"
        filename = f"warehouse_export.{ext}"

        try:
            conn = sqlite3.connect(sp, check_same_thread=False)
            try:
                cols: List[str] = []
                try:
                    cur = conn.execute("PRAGMA table_info(warehouse_cleaned)")
                    cols = [str(r[1]) for r in (cur.fetchall() or []) if r and len(r) > 1]
                except Exception:
                    cols = []
                cols = [c for c in cols if c not in {"日期", "金额文本", "值状态"}]
                allowed_cols = set(cols) if cols else {
                    "源文件", "来源Sheet", "期间", "年份", "报表口径", "报表类型", "大类", "科目", "科目规范", "时间属性", "金额"
                }

                where: List[str] = []
                params: List[Any] = []

                def add_eq(col: str, val: str) -> None:
                    if val is None:
                        return
                    v = str(val).strip()
                    if not v:
                        return
                    if col not in allowed_cols:
                        return
                    where.append(f'"{col}" = ?')
                    params.append(v)

                add_eq("batch_id", batch_id)
                add_eq("源文件", source_file)
                add_eq("报表类型", report_type)
                add_eq("大类", category)
                add_eq("时间属性", time_attr)
                add_eq("期间", period)
                add_eq("年份", year)
                add_eq("报表口径", caliber)

                kw_input = (subject or q or "").strip()
                if kw_input:
                    import re
                    keywords = [k.strip() for k in re.split(r'[\s,，、;/；|/]+', kw_input) if k.strip()]
                    if keywords:
                        or_clauses: List[str] = []
                        for k in keywords:
                            k_norm = _normalize_subject_text(k)
                            if k_norm and "科目规范" in allowed_cols:
                                or_clauses.append('LOWER(CAST("科目规范" AS TEXT)) LIKE ?')
                                params.append(f"%{k_norm.lower()}%")
                            if "科目" in allowed_cols:
                                or_clauses.append('LOWER(CAST("科目" AS TEXT)) LIKE ?')
                                params.append(f"%{k.lower()}%")
                        if or_clauses:
                            where.append(f"({' OR '.join(or_clauses)})")

                if min_amount is not None and "金额" in allowed_cols:
                    try:
                        where.append('"金额" >= ?')
                        params.append(float(min_amount))
                    except Exception:
                        pass
                if max_amount is not None and "金额" in allowed_cols:
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
                    base_sql = f'SELECT {g_str}, SUM("金额") as "金额", COUNT(*) as "条数" FROM warehouse_cleaned' + where_sql + f' GROUP BY {g_str}'
                    order_sql = ""
                    if s_col == "金额":
                        order_sql = f' ORDER BY SUM("金额") {"ASC" if s_dir == "asc" else "DESC"}'
                    elif s_col == "条数":
                        order_sql = f' ORDER BY COUNT(*) {"ASC" if s_dir == "asc" else "DESC"}'
                    elif s_col in allowed_cols:
                        order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'
                else:
                    if cols:
                        select_list = ", ".join([f'"{c}"' for c in cols])
                    else:
                        select_list = "*"
                    base_sql = f"SELECT {select_list} FROM warehouse_cleaned" + where_sql
                    order_sql = ""
                    if s_col in allowed_cols:
                        order_sql = f' ORDER BY "{s_col}" {"ASC" if s_dir == "asc" else "DESC"}'

                cap = 50000
                if topn is not None:
                    try:
                        cap = int(max(1, min(int(topn), 50000)))
                    except Exception:
                        pass

                cur = conn.execute(base_sql + order_sql + " LIMIT ?", params + [cap])
                out_cols = [d[0] for d in (cur.description or [])]
                fetched = cur.fetchall()

                if is_xlsx:
                    output = io.BytesIO()
                    try:
                        df = pd.DataFrame([list(r) for r in fetched], columns=out_cols)
                        df.to_excel(output, index=False)
                        content = output.getvalue()
                    except Exception as e:
                        return Response(str(e), media_type="text/plain", status_code=500)
                else:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(out_cols)
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
    try:
        rp = os.path.abspath(os.path.join(_default_data_root(), "web_runtime.json"))
        _ensure_dir(os.path.dirname(rp))
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "host": str(host),
                    "port": int(chosen_port),
                    "pid": int(os.getpid()),
                    "config_path": os.path.abspath(str(config_path or "")),
                },
                f,
                ensure_ascii=False,
            )
    except Exception:
        pass
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

