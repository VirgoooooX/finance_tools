import os
import sys
import json
import logging
import threading
import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Dict, List, Tuple

from fa_platform.paths import (
    ensure_dir as _ensure_dir_common,
    default_output_root as _default_output_root_common,
    default_data_root as _default_data_root_common,
    resolve_under_base as _resolve_under_base_common,
    get_base_dir as _get_base_dir_common,
)

OUTPUT_PATH = "清洗后的AI标准财务表.xlsx"

def get_base_dir():
    return _get_base_dir_common()

@dataclass
class AppConfig:
    input_dir: str = field(default_factory=lambda: os.getcwd())
    file_glob: str = "*.xlsx"
    output_dir: str = field(default_factory=lambda: "output")
    output_basename: str = OUTPUT_PATH
    generate_validation: bool = True
    generate_metrics: bool = True
    exclude_output_files: bool = True
    # Tool specific fields (Legacy, kept for compatibility with tool code)
    sheet_keyword_bs: List[str] = field(default_factory=lambda: ["BS-合并"])
    sheet_keyword_pl: List[str] = field(default_factory=lambda: ["PL-合并"])
    sheet_keyword_cf: List[str] = field(default_factory=lambda: ["CF-合并"])
    header_keyword_bs: str = "期末余额"
    header_keyword_pl: str = "本年累计"
    header_keyword_cf: str = "本期金额"
    date_cells_bs: List[List[int]] = field(default_factory=lambda: [[2, 3], [2, 2]])
    date_cells_pl: List[List[int]] = field(default_factory=lambda: [[2, 2], [2, 1]])
    date_cells_cf: List[List[int]] = field(default_factory=lambda: [[2, 4], [2, 0]])
    validation_tolerance: float = 0.01
    col_keyword_subject: List[str] = field(default_factory=lambda: ["资产", "项目", "科目", "摘要"])
    col_keyword_period_end: List[str] = field(default_factory=lambda: ["期末", "本期", "本年", "金额"])
    col_keyword_period_start: List[str] = field(default_factory=lambda: ["上年", "上期", "年初"])
    saved_queries: Dict[str, Any] = field(default_factory=dict)
    tool_id: str = "monthly_report_cleaner"
    tool_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    cancelled: bool = False
    errors: List[str] = field(default_factory=list)
    found_files: List[str] = field(default_factory=list)
    processed_files: int = 0
    cleaned_rows: int = 0
    metrics_groups: int = 0
    validation_groups: int = 0
    unbalanced_count: int = 0
    cleaned_path: Optional[str] = None
    cleaned_sqlite_path: Optional[str] = None
    validation_path: Optional[str] = None
    metrics_path: Optional[str] = None
    unbalanced_preview: List[Dict[str, Any]] = field(default_factory=list)
    validation_preview: List[Dict[str, Any]] = field(default_factory=list)
    metrics_preview: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

ToolRunFn = Callable[..., "AnalysisResult"]

@dataclass(frozen=True)
class ToolSpec:
    id: str
    name: str
    run: ToolRunFn
    web_router: Optional[Callable[[], Any]] = None

DEFAULT_TOOL_ID = "monthly_report_cleaner"
# Default config path now points to the default tool's config
DEFAULT_CONFIG_PATH = os.path.join(get_base_dir(), "tools", DEFAULT_TOOL_ID, "config.json")

_TOOL_REGISTRY: Dict[str, ToolSpec] = {}
_TOOLS_DISCOVERED = False
_TOOL_DISCOVERY_ERRORS: List[Dict[str, str]] = []

def get_tool_discovery_errors() -> List[Dict[str, str]]:
    return list(_TOOL_DISCOVERY_ERRORS)

def register_tool(tool: ToolSpec) -> None:
    tid = str(getattr(tool, "id", "") or "").strip()
    if not tid:
        raise ValueError("tool.id 不能为空")
    _TOOL_REGISTRY[tid] = tool

def discover_tools(force: bool = False) -> None:
    global _TOOLS_DISCOVERED
    if _TOOLS_DISCOVERED and not force:
        return
    _TOOLS_DISCOVERED = True
    if force:
        _TOOL_DISCOVERY_ERRORS.clear()

    base_dirs: List[str] = []
    try:
        base_dirs.append(os.path.abspath(get_base_dir()))
    except Exception:
        pass
    
    seen_paths = set()
    for bd in base_dirs:
        tools_dir = os.path.join(bd, "tools")
        if not os.path.isdir(tools_dir):
            continue
        
        # Scan for builtin_tools.py or other .py files
        # Also scan for subdirectories with __init__.py?
        # Current logic only scanned .py files in tools/
        # We should update it to support packages if needed, but for now builtin_tools handles registration.
        try:
            names = os.listdir(tools_dir)
        except Exception:
            continue

        for fn in names:
            s = str(fn or "").strip()
            if not s.endswith(".py"):
                continue
            if s.startswith("_"):
                continue
            full = os.path.abspath(os.path.join(tools_dir, s))
            if full in seen_paths:
                continue
            seen_paths.add(full)
            mod_name = f"financial_analyzer_tools.{os.path.splitext(s)[0]}"
            try:
                if mod_name in sys.modules:
                    continue
                spec = importlib.util.spec_from_file_location(mod_name, full)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
            except Exception as e:
                _TOOL_DISCOVERY_ERRORS.append({"path": full, "module": mod_name, "error": str(e)})

def _ensure_builtin_tools() -> None:
    discover_tools(force=False)
    # Note: builtin_tools.py should handle registration.
    # We no longer hardcode default tool registration here if it's not in registry.
    # But for safety:
    if DEFAULT_TOOL_ID not in _TOOL_REGISTRY:
        # Fallback? No, we expect builtin_tools.py to run.
        pass

def list_tools() -> List[Dict[str, str]]:
    _ensure_builtin_tools()
    items = [{"id": t.id, "name": t.name} for t in _TOOL_REGISTRY.values()]
    items.sort(key=lambda x: (str(x.get("name") or ""), str(x.get("id") or "")))
    return items

def get_tool(tool_id: str) -> ToolSpec:
    _ensure_builtin_tools()
    tid = str(tool_id or "").strip()
    if tid and tid in _TOOL_REGISTRY:
        return _TOOL_REGISTRY[tid]
    if DEFAULT_TOOL_ID in _TOOL_REGISTRY:
        return _TOOL_REGISTRY[DEFAULT_TOOL_ID]
    raise ValueError(f"Tool {tid} not found and default tool missing")

def resolve_tool_id(tool_id: str) -> Tuple[str, bool]:
    _ensure_builtin_tools()
    tid = str(tool_id or "").strip()
    if tid and tid in _TOOL_REGISTRY:
        return tid, False
    return DEFAULT_TOOL_ID, True

def run_tool(
    tool_id: str,
    cfg: "AppConfig",
    logger: Optional[logging.Logger] = None,
    progress_cb: Optional["Callable"] = None,
    cancel_event: Optional[threading.Event] = None,
) -> "AnalysisResult":
    resolved_id, _ = resolve_tool_id(tool_id)
    tool = _TOOL_REGISTRY[resolved_id]
    return tool.run(cfg, logger=logger, progress_cb=progress_cb, cancel_event=cancel_event)

def _get_logger(name: str = "financial_analyzer", handler: Optional[logging.Handler] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger

# --- Helpers for Web Server & Tools ---

def _list_excel_files(cfg: AppConfig) -> List[str]:
    import glob
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

def _is_timestamp_folder(name: str) -> bool:
    import re
    return bool(re.match(r"^\d{8}_\d{6}$", str(name or "").strip()))

def _cleaned_sqlite_path_for(cleaned_path: str) -> str:
    name = os.path.basename(str(cleaned_path or "")).strip()
    if not name:
        base_name = "cleaned"
    else:
        base_name, _ = os.path.splitext(name)
        base_name = base_name or "cleaned"

    ts = ""
    try:
        parent = os.path.basename(os.path.dirname(str(cleaned_path or "")))
        if _is_timestamp_folder(parent):
            ts = parent
    except Exception:
        ts = ""

    filename = f"{base_name}_{ts}.sqlite" if ts else f"{base_name}.sqlite"
    return os.path.abspath(os.path.join(_default_data_root_common(), filename))

_RULES_CACHE: Dict[str, Any] = {"mtime": None, "data": None, "aliases": None}

def _load_rules() -> Dict[str, Any]:
    # Default to loading from the default tool's directory for now
    # In a real multi-tool platform, this should be passed the tool ID
    path = os.path.join(get_base_dir(), "tools", DEFAULT_TOOL_ID, "rules.json")
    if not os.path.exists(path):
        # Fallback to old config location
        path = os.path.join(get_base_dir(), "config", "rules.json")
    
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _repair_mojibake_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return obj.encode("latin1").decode("utf-8")
        except Exception:
            return obj
    if isinstance(obj, list):
        return [_repair_mojibake_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _repair_mojibake_obj(v) for k, v in obj.items()}
    return obj

ProgressCallback = Callable[[str, int, int, str], None]

# --- Config Loading ---
def load_config(path: str) -> AppConfig:
    if not path or not os.path.exists(path):
        # Try default location if path not found
        if path == DEFAULT_CONFIG_PATH and not os.path.exists(path):
            # Fallback to old location?
            old_path = os.path.join(get_base_dir(), "config", "financial_analyzer_config.json")
            if os.path.exists(old_path):
                path = old_path

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
    except Exception:
        return AppConfig()

def save_config(path: str, cfg: AppConfig) -> None:
    if not path:
        path = DEFAULT_CONFIG_PATH
    
    # Ensure dir exists
    _ensure_dir_common(os.path.dirname(path))
    
    data = {}
    # Convert dataclass to dict, excluding internal fields if any?
    # For now just dump everything in AppConfig
    # We need asdict but we imported it? No, need to import asdict
    from dataclasses import asdict
    data = asdict(cfg)
    
    try:
        # Try to make output_dir relative
        base = os.path.abspath(get_base_dir())
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
