import socket
from typing import Any, Dict, Optional, Tuple

from fa_platform.jsonx import sanitize_json
from fa_platform.paths import get_resource_path


def read_web_index_html() -> str:
    try:
        p = get_resource_path("web/index.html")
        return p.read_text(encoding="utf-8")
    except Exception:
        return "<!doctype html><html><head><meta charset='utf-8'><title>Web</title></head><body>缺少 web/index.html</body></html>"


def choose_web_port(host: str, requested_port: int) -> int:
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


def sse_encode(event: str, data_obj: Any) -> str:
    import json as _json

    s = _json.dumps(sanitize_json(data_obj), ensure_ascii=False, allow_nan=False)
    return f"event: {event}\ndata: {s}\n\n"


def get_tool_web_manifest(tool_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    import json as _json

    tid = str(tool_id or "").strip()
    if not tid:
        return None, None
    try:
        p = get_resource_path(f"tools/{tid}/web/manifest.json")
        if not p.exists() or not p.is_file():
            return None, None
        raw = p.read_text(encoding="utf-8")
        data = _json.loads(raw)
        if not isinstance(data, dict):
            return None, "manifest.json 必须是 JSON 对象"
        return data, None
    except Exception as e:
        return None, str(e)


def get_tool_web_entry_url(tool_id: str) -> Tuple[Optional[str], Optional[str]]:
    tid = str(tool_id or "").strip()
    if not tid:
        return None, None
    manifest, err = get_tool_web_manifest(tid)
    if err:
        return None, err
    if manifest is None:
        return None, None
    entry = str(manifest.get("entry") or "index.html").strip().lstrip("/")
    if not entry:
        entry = "index.html"
    return f"/tools/{tid}/web/{entry}", None
