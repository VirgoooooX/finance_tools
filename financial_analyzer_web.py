import os
import sys
import queue
import threading
import logging
import socket
from dataclasses import asdict
from typing import Optional, Any, Dict, List
from pathlib import Path

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

    def _get_cleaned_path(self) -> str:
        with self._lock:
            res = self.last_result
            cfg = self.cfg
        if res and getattr(res, "cleaned_path", None):
            return str(res.cleaned_path or "")
        return os.path.abspath(os.path.join(cfg.output_dir, cfg.output_basename))

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
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
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

        total = int(len(view))
        lim = int(max(1, min(int(limit or 200), 2000)))
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
        })

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

