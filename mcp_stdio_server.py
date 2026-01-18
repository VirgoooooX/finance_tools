import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from fa_platform.paths import default_data_root, get_base_dir


_SERVER_NAME = "fa-warehouse-mcp"
_SERVER_VERSION = "0.1.1"
_PROTOCOL_VERSION = "2025-06-18"


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger(_SERVER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    return logger


def _stdout_send(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _read_runtime_endpoint(runtime_path: str) -> Optional[str]:
    try:
        if not runtime_path or not os.path.exists(runtime_path):
            return None
        with open(runtime_path, "r", encoding="utf-8") as f:
            obj = json.load(f) or {}
        host = str(obj.get("host") or "").strip() or "127.0.0.1"
        port = int(obj.get("port") or 0)
        if port <= 0 or port >= 65536:
            return None
        return f"http://{host}:{port}"
    except Exception:
        return None


def _http_get_json(base_url: str, path: str, params: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    qs = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and str(v) != ""})
    url = base_url.rstrip("/") + path
    if qs:
        url = url + "?" + qs
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _resolve_base_url(explicit_base_url: str, runtime_path: str) -> Tuple[Optional[str], Optional[str]]:
    if explicit_base_url:
        return explicit_base_url.rstrip("/"), None
    env_url = str(os.environ.get("FA_WEB_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/"), None
    url = _read_runtime_endpoint(runtime_path)
    if url:
        return url.rstrip("/"), None
    return None, f"未找到 Web 服务地址：请先启动 Web，或设置 FA_WEB_URL，或确保存在 {runtime_path}"


def _contains_any(text: str, needles: List[str]) -> bool:
    t = str(text or "")
    return any(n for n in needles if n and n in t)


def _infer_caliber_from_text(text: str) -> str:
    t = str(text or "")
    if _contains_any(t, ["单体", "母公司", "个别"]):
        return "单体"
    if _contains_any(t, ["合并", "并表", "集团"]):
        return "合并"
    return ""


def _infer_report_type_from_text(text: str) -> str:
    t = str(text or "")
    if _contains_any(t, ["资产负债"]):
        return "资产负债表"
    if _contains_any(t, ["现金流", "现金流量"]):
        return "现金流量表"
    if _contains_any(t, ["利润表", "损益表", "损益", "利润"]):
        return "利润表"
    return ""


def _infer_time_attr_from_text(text: str, report_type: str) -> str:
    t = str(text or "")
    if _contains_any(t, ["期末", "期末余额"]):
        return "期末余额"
    if _contains_any(t, ["期初", "年初", "年初余额"]):
        return "年初余额"
    if _contains_any(t, ["累计", "本年累计", "年累计", "ytd", "YTD"]):
        return "本年累计金额"
    if _contains_any(t, ["本期", "当期", "本月", "发生额"]):
        return "本期金额"
    rt = str(report_type or "").strip()
    if rt == "资产负债表":
        return "期末余额"
    if rt in ("利润表", "现金流量表"):
        return "本期金额"
    return ""


def _normalize_period(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s2 = s.replace("-", "").replace("/", "").replace(".", "")
    digits = "".join([c for c in s2 if c.isdigit()])
    if len(digits) == 6:
        return digits
    if len(digits) == 8 and digits.startswith("20"):
        return digits[:6]
    return s


def _compact_query_keywords(text: str) -> str:
    t = str(text or "")
    if not t.strip():
        return ""
    for w in [
        "合并",
        "单体",
        "口径",
        "资产负债表",
        "利润表",
        "损益表",
        "现金流量表",
        "期初",
        "期末",
        "年初",
        "本期",
        "累计",
        "余额",
        "发生额",
        "金额",
    ]:
        t = t.replace(w, " ")
    t = "".join([c if (c.isalnum() or c in "（）()") else " " for c in t])
    parts = [p.strip() for p in t.split() if p.strip() and not p.strip().isdigit()]
    return " ".join(parts).strip()


def _get_str_arg(args: Dict[str, Any], key: str) -> str:
    return str((args or {}).get(key) or "").strip()


def _smart_fetch(
    base_url: str,
    timeout_s: float,
    args: Dict[str, Any],
    query_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw_q = _get_str_arg(args, "q") or _get_str_arg(args, "subject") or ""
    if not raw_q:
        return {"ok": False, "message": "缺少 q（或 subject）"}

    explicit_period = _get_str_arg(args, "period")
    explicit_caliber = _get_str_arg(args, "caliber")
    explicit_report_type = _get_str_arg(args, "report_type")
    explicit_time_attr = _get_str_arg(args, "time_attr")
    explicit_category = _get_str_arg(args, "category")
    explicit_batch_id = _get_str_arg(args, "batch_id")
    explicit_source_file = _get_str_arg(args, "source_file")

    period = _normalize_period(explicit_period)
    report_type = explicit_report_type
    if not report_type:
        report_type = _infer_report_type_from_text(raw_q)
    caliber = explicit_caliber
    if not caliber:
        caliber = _infer_caliber_from_text(raw_q) or "合并"
    time_attr = explicit_time_attr
    if not time_attr:
        time_attr = _infer_time_attr_from_text(raw_q, report_type)

    topn = args.get("topn", None)
    try:
        topn_i = int(topn) if topn is not None else 20
    except Exception:
        topn_i = 20
    topn_i = max(5, min(topn_i, 50))

    q_kw = _compact_query_keywords(raw_q) or raw_q
    call_query = query_fn
    if call_query is None:
        call_query = lambda p: _http_get_json(base_url, "/api/warehouse/query", p, timeout_s) or {}

    attempts: List[Dict[str, Any]] = []

    def try_once(tag: str, p: Dict[str, Any]) -> Dict[str, Any]:
        data = call_query(p)
        ok = bool(data.get("ok"))
        total = int(data.get("total") or 0) if ok else 0
        rows = data.get("rows") or []
        attempts.append(
            {
                "tag": tag,
                "ok": ok,
                "total": total,
                "rows_returned": len(rows) if isinstance(rows, list) else 0,
                "arguments": {k: v for k, v in p.items() if v is not None and str(v) != ""},
            }
        )
        return data

    strict_params: Dict[str, Any] = {
        "q": "",
        "subject": q_kw,
        "period": period,
        "caliber": caliber,
        "report_type": report_type,
        "time_attr": time_attr,
        "category": explicit_category,
        "batch_id": explicit_batch_id,
        "source_file": explicit_source_file,
        "group_by": "科目规范",
        "sort_by": "金额",
        "sort_dir": "desc",
        "topn": min(10, topn_i),
    }
    data = try_once("direct_strict", dict(strict_params))
    rows = data.get("rows") if isinstance(data, dict) else None

    if not (isinstance(rows, list) and rows):
        candidate_params: Dict[str, Any] = {
            "q": q_kw,
            "subject": "",
            "period": period,
            "caliber": caliber,
            "report_type": report_type,
            "time_attr": time_attr,
            "category": explicit_category,
            "batch_id": explicit_batch_id,
            "source_file": explicit_source_file,
            "group_by": "科目规范,报表类型,时间属性",
            "sort_by": "金额",
            "sort_dir": "desc",
            "topn": topn_i,
        }
        data = try_once("fallback_candidates", dict(candidate_params))
        rows = data.get("rows") if isinstance(data, dict) else None

        if not (isinstance(rows, list) and rows) and not explicit_time_attr and time_attr:
            p2 = dict(candidate_params)
            p2["time_attr"] = ""
            data = try_once("fallback_drop_time_attr", p2)
            rows = data.get("rows") if isinstance(data, dict) else None

        if not (isinstance(rows, list) and rows) and not explicit_report_type and report_type:
            p3 = dict(candidate_params)
            p3["report_type"] = ""
            data = try_once("fallback_drop_report_type", p3)
            rows = data.get("rows") if isinstance(data, dict) else None

        if not (isinstance(rows, list) and rows) and not explicit_caliber and caliber:
            alt = "单体" if caliber == "合并" else "合并"
            p4 = dict(candidate_params)
            p4["caliber"] = alt
            data = try_once("fallback_switch_caliber", p4)
            rows = data.get("rows") if isinstance(data, dict) else None
            if isinstance(rows, list) and rows:
                caliber = alt

    candidates: List[Dict[str, Any]] = []
    if isinstance(rows, list):
        for r in rows[: min(len(rows), 10)]:
            if not isinstance(r, dict):
                continue
            candidates.append(
                {
                    "科目规范": r.get("科目规范", ""),
                    "报表类型": r.get("报表类型", ""),
                    "时间属性": r.get("时间属性", ""),
                    "金额": r.get("金额", None),
                    "条数": r.get("条数", None),
                }
            )

    chosen = candidates[0] if candidates else {}
    choice = {
        "subject_norm": str(chosen.get("科目规范") or "").strip(),
        "report_type": str(chosen.get("报表类型") or report_type or "").strip(),
        "time_attr": str(chosen.get("时间属性") or time_attr or "").strip(),
        "caliber": str(caliber or "").strip(),
        "period": str(period or "").strip(),
        "category": str(explicit_category or "").strip(),
        "batch_id": str(explicit_batch_id or "").strip(),
        "source_file": str(explicit_source_file or "").strip(),
    }

    return {
        "ok": bool(candidates),
        "message": "" if candidates else "未检索到候选科目，请调整关键词或放宽筛选条件",
        "query": {"raw": raw_q, "keywords": q_kw},
        "choice": {k: v for k, v in choice.items() if v},
        "value": chosen.get("金额", None) if chosen else None,
        "rows_count": chosen.get("条数", None) if chosen else None,
        "candidates": candidates[:5],
        "provenance": {"attempts": attempts},
    }


def _batch_smart_fetch(base_url: str, timeout_s: float, args: Dict[str, Any]) -> Dict[str, Any]:
    reqs = (args or {}).get("requests")
    if not isinstance(reqs, list) or not reqs:
        return {"ok": False, "message": "缺少 requests（数组）", "results": []}

    cache: Dict[str, Dict[str, Any]] = {}

    def cached_query(p: Dict[str, Any]) -> Dict[str, Any]:
        key = json.dumps(p or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key in cache:
            return cache[key]
        data = _http_get_json(base_url, "/api/warehouse/query", p, timeout_s) or {}
        cache[key] = data
        return data

    results: List[Dict[str, Any]] = []
    for i, r in enumerate(reqs):
        req = r if isinstance(r, dict) else {}
        rid = str(req.get("id") or req.get("name") or i)
        out = _smart_fetch(base_url, timeout_s, req, query_fn=cached_query)
        out["id"] = rid
        results.append(out)

    return {"ok": True, "results": results}


def _tool_schemas() -> Dict[str, Dict[str, Any]]:
    common_props = {
        "batch_id": {"type": "string"},
        "q": {"type": "string"},
        "subject": {"type": "string"},
        "source_file": {"type": "string"},
        "report_type": {"type": "string"},
        "category": {"type": "string"},
        "time_attr": {"type": "string"},
        "period": {"type": "string"},
        "year": {"type": "string"},
        "caliber": {"type": "string"},
        "min_amount": {"type": "number"},
        "max_amount": {"type": "number"},
        "sort_by": {"type": "string"},
        "sort_dir": {"type": "string", "enum": ["asc", "desc"]},
        "topn": {"type": "integer"},
        "limit": {"type": "integer"},
        "offset": {"type": "integer"},
        "group_by": {"type": "string"},
    }
    return {
        "warehouse_options": {
            "name": "warehouse_options",
            "description": "读取累计库可选项（源文件/期间/报表类型/时间属性等）。",
            "inputSchema": {"type": "object", "properties": {"batch_id": {"type": "string"}}, "additionalProperties": False},
        },
        "warehouse_summary": {
            "name": "warehouse_summary",
            "description": "读取累计库概览（行数、覆盖期间、最近导入时间等）。",
            "inputSchema": {"type": "object", "properties": {"batch_id": {"type": "string"}}, "additionalProperties": False},
        },
        "warehouse_query": {
            "name": "warehouse_query",
            "description": "按条件查询累计库（支持筛选/排序/TopN/分组聚合）。",
            "inputSchema": {"type": "object", "properties": common_props, "additionalProperties": False},
        },
        "warehouse_smart_fetch": {
            "name": "warehouse_smart_fetch",
            "description": "智能取数：默认直取；空查时自动反推一次并重试，返回候选与推荐取数范围。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "period": {"type": "string"},
                    "caliber": {"type": "string"},
                    "report_type": {"type": "string"},
                    "time_attr": {"type": "string"},
                    "category": {"type": "string"},
                    "batch_id": {"type": "string"},
                    "source_file": {"type": "string"},
                    "topn": {"type": "integer"},
                },
                "required": ["q"],
                "additionalProperties": False,
            },
        },
        "warehouse_batch_smart_fetch": {
            "name": "warehouse_batch_smart_fetch",
            "description": "批量智能取数：一次调用传入多条请求，返回每条的候选与推荐取数范围。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "q": {"type": "string"},
                                "period": {"type": "string"},
                                "caliber": {"type": "string"},
                                "report_type": {"type": "string"},
                                "time_attr": {"type": "string"},
                                "category": {"type": "string"},
                                "batch_id": {"type": "string"},
                                "source_file": {"type": "string"},
                                "topn": {"type": "integer"},
                            },
                            "required": ["q"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["requests"],
                "additionalProperties": False,
            },
        },
        "warehouse_export_url": {
            "name": "warehouse_export_url",
            "description": "生成累计库导出链接（CSV/XLSX）。",
            "inputSchema": {
                "type": "object",
                "properties": {**common_props, "format": {"type": "string", "enum": ["csv", "xlsx"]}},
                "additionalProperties": False,
            },
        },
    }


def _tool_list_result() -> Dict[str, Any]:
    tools = []
    for t in _tool_schemas().values():
        tools.append({"name": t["name"], "description": t.get("description", ""), "inputSchema": t["inputSchema"]})
    return {"tools": tools}


def _tool_call(base_url: str, runtime_path: str, timeout_s: float, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "warehouse_options":
        data = _http_get_json(base_url, "/api/warehouse/options", {"batch_id": str((args or {}).get("batch_id") or "").strip()}, timeout_s)
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
    if name == "warehouse_summary":
        data = _http_get_json(base_url, "/api/warehouse/summary", {"batch_id": str((args or {}).get("batch_id") or "").strip()}, timeout_s)
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
    if name == "warehouse_export_url":
        p = dict(args or {})
        fmt = str(p.pop("format", "") or "csv").strip().lower()
        fmt = fmt if fmt in ("csv", "xlsx") else "csv"
        qs = urllib.parse.urlencode({k: v for k, v in (p or {}).items() if v is not None and str(v) != ""})
        url = base_url.rstrip("/") + "/api/warehouse/export"
        if qs:
            url = url + "?" + qs + "&format=" + urllib.parse.quote(fmt)
        else:
            url = url + "?format=" + urllib.parse.quote(fmt)
        return {"content": [{"type": "text", "text": url}]}
    if name == "warehouse_query":
        p = dict(args or {})
        data = _http_get_json(base_url, "/api/warehouse/query", p, timeout_s)
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
    if name == "warehouse_smart_fetch":
        data = _smart_fetch(base_url, timeout_s, args if isinstance(args, dict) else {})
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
    if name == "warehouse_batch_smart_fetch":
        data = _batch_smart_fetch(base_url, timeout_s, args if isinstance(args, dict) else {})
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}]}
    raise KeyError(name)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--base-url", default="", help="例如 http://127.0.0.1:8765")
    ap.add_argument("--runtime-file", default="", help="默认读取 data/web_runtime.json")
    ap.add_argument("--timeout", default="3.0")
    ap.add_argument("--log-level", default="INFO")
    ns = ap.parse_args()

    logger = _setup_logger(str(ns.log_level or "INFO"))
    try:
        timeout_s = float(ns.timeout or 3.0)
        if timeout_s <= 0:
            timeout_s = 3.0
    except Exception:
        timeout_s = 3.0

    runtime_path = str(ns.runtime_file or "").strip()
    if not runtime_path:
        runtime_path = os.path.abspath(os.path.join(default_data_root(), "web_runtime.json"))
    elif not os.path.isabs(runtime_path):
        runtime_path = os.path.abspath(os.path.join(get_base_dir(), runtime_path))

    initialized = False

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            logger.error(f"解析失败: {e}")
            continue

        method = str((msg or {}).get("method") or "").strip()
        req_id = (msg or {}).get("id", None)

        if not method:
            continue

        if method == "initialize":
            initialized = True
            result = {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
            }
            _stdout_send({"jsonrpc": "2.0", "id": req_id, "result": result})
            continue

        if method == "initialized":
            continue

        if method == "ping":
            if req_id is not None:
                _stdout_send({"jsonrpc": "2.0", "id": req_id, "result": {}})
            continue

        if not initialized and method not in ("initialize", "ping"):
            if req_id is not None:
                _stdout_send(_jsonrpc_error(req_id, -32002, "未初始化"))
            continue

        if method == "tools/list":
            if req_id is None:
                continue
            _stdout_send({"jsonrpc": "2.0", "id": req_id, "result": _tool_list_result()})
            continue

        if method == "tools/call":
            if req_id is None:
                continue
            params = (msg or {}).get("params") or {}
            tool_name = str((params or {}).get("name") or "").strip()
            args = (params or {}).get("arguments") or {}
            if not tool_name:
                _stdout_send(_jsonrpc_error(req_id, -32602, "缺少工具名"))
                continue
            base_url, err = _resolve_base_url(str(ns.base_url or "").strip(), runtime_path)
            if not base_url:
                _stdout_send(_jsonrpc_error(req_id, -32000, err or "未配置 Web 服务地址"))
                continue
            try:
                result = _tool_call(base_url, runtime_path, timeout_s, tool_name, args if isinstance(args, dict) else {})
                _stdout_send({"jsonrpc": "2.0", "id": req_id, "result": result})
            except KeyError:
                _stdout_send(_jsonrpc_error(req_id, -32601, f"未知工具: {tool_name}"))
            except Exception as e:
                _stdout_send(_jsonrpc_error(req_id, -32000, "工具执行失败", {"error": str(e), "tool": tool_name}))
            continue

        if req_id is not None:
            _stdout_send(_jsonrpc_error(req_id, -32601, f"未知方法: {method}"))

    time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
