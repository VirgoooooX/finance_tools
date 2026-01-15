import os
import sys
from pathlib import Path


def get_base_dir() -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_resource_path(relative_path: str) -> Path:
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).resolve().parent.parent
    return base_path / relative_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_output_root() -> str:
    return os.path.join(get_base_dir(), "output")


def default_data_root() -> str:
    return os.path.join(get_base_dir(), "data")


def resolve_under_base(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(get_base_dir(), p))

