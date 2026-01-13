import os
import queue
import threading
import logging
from dataclasses import asdict
from typing import Optional, Any, Dict, List

from financial_analyzer_core import (
    OUTPUT_PATH,
    AppConfig,
    AnalysisResult,
    load_config,
    save_config,
    analyze_directory,
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

    def set_config(self, data: Dict[str, Any]) -> AppConfig:
        cfg = AppConfig()
        for k, v in (data or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if not getattr(cfg, "input_dir", None):
            cfg.input_dir = os.getcwd()
        if not getattr(cfg, "file_glob", None):
            cfg.file_glob = "*.xlsx"
        if not getattr(cfg, "output_dir", None):
            cfg.output_dir = os.getcwd()
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


def _web_index_html() -> str:
    try:
        from pathlib import Path

        p = Path(__file__).resolve().parent / "web" / "index.html"
        return p.read_text(encoding="utf-8")
    except Exception:
        return "<!doctype html><html><head><meta charset='utf-8'><title>Web</title></head><body>缺少 web/index.html</body></html>"


def run_web(config_path: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception:
        print("缺少依赖：fastapi。请先安装：py -m pip install fastapi uvicorn")
        return 1

    runner = _WebRunner(config_path=config_path)
    app = FastAPI()
    try:
        from pathlib import Path

        web_dir = Path(__file__).resolve().parent / "web"
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    except Exception:
        pass

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _web_index_html()

    @app.get("/api/config")
    def api_get_config() -> Dict[str, Any]:
        return asdict(runner.get_config())

    @app.post("/api/config")
    def api_set_config(data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = runner.set_config(data)
        return asdict(cfg)

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
            s = _json.dumps(data_obj, ensure_ascii=False)
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

    if open_browser:
        try:
            import webbrowser

            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    try:
        import uvicorn
    except Exception:
        print("缺少依赖：uvicorn。请先安装：py -m pip install uvicorn")
        return 1

    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return 0


if __name__ == "__main__":
    from financial_analyzer_core import DEFAULT_CONFIG_PATH

    raise SystemExit(run_web(DEFAULT_CONFIG_PATH))

