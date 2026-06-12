from collections.abc import Mapping
from dataclasses import is_dataclass, asdict, fields
from pathlib import Path
from enum import Enum

def _make_json_safe(obj):
    # 1) Dataclasses → asdict recursion (must come before Enum handling)
    if is_dataclass(obj):
        data = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            data[f.name] = _make_json_safe(value)
        return data

    # 2) Anything with a .to_dict() method → use it
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        # Important: recurse on the result, in case it still contains Enums, etc.
        return _make_json_safe(to_dict())

    # 3) Enums → value (or name)
    if isinstance(obj, Enum):
        return _make_json_safe(obj.value)  # or obj.name if you prefer

    # 3) Paths → string
    if isinstance(obj, Path):
        return str(obj)

    # 4) Dict-like objects → recurse into items
    if isinstance(obj, Mapping):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    # 5) Lists / tuples / sets → recurse into elements
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    # 6) Primitives (str, int, float, bool, None, etc.) → leave as is
    return obj

def convert_config_to_json_default(dict_obj):
    return _make_json_safe(dict_obj)
