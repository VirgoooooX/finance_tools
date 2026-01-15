import datetime as _dt
import math as _math
from typing import Any

try:
    import pandas as _pd
except Exception:
    _pd = None


def sanitize_json(obj: Any) -> Any:
    if obj is None:
        return None

    if _pd is not None:
        try:
            if _pd.isna(obj):
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

    if _pd is not None:
        try:
            if isinstance(obj, (_dt.datetime, _dt.date, _pd.Timestamp)):
                return str(obj)
        except Exception:
            pass
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return str(obj)

    if isinstance(obj, list):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}

    try:
        item = getattr(obj, "item", None)
        if callable(item):
            return sanitize_json(item())
    except Exception:
        pass

    return str(obj)

