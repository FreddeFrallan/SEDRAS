from typing import Any, Dict
import json

def _json_dumps_safe(obj: Any) -> str:
    """Serialize any object to a JSON string; fallback to str() for unknown types."""
    try:
        return json.dumps(obj, default=lambda o: str(o))
    except Exception:
        return str(obj)


def _maybe_json_loads(x: Any) -> Any:
    """If x is a JSON-looking string, try to json.loads it; otherwise return x unchanged."""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _normalize_assignment(assignment: Any) -> Dict[str, int]:
    """
    Accepts:
      - dict with keys 'V0','V1',... or 0,1,2,3 (ints/strings)
      - stringified JSON of the above
    Returns a dict with 'V*' keys and int values.
    """
    assignment = _maybe_json_loads(assignment)

    if not isinstance(assignment, dict):
        raise ValueError(f"assignment must be a dict, got {type(assignment).__name__}: {assignment!r}")

    out: Dict[str, int] = {}
    for k, v in assignment.items():
        # normalize key to 'V{n}'
        if isinstance(k, str) and k.startswith("V") and k[1:].isdigit():
            key = k
        else:
            try:
                n = int(k)
                key = f"V{n}"
            except Exception:
                key = f"{k}"

        # normalize value to int
        vv = _maybe_json_loads(v)
        if isinstance(vv, bool):  # avoid bool being int-like
            val = int(vv)
        elif isinstance(vv, (int,)):
            val = int(vv)
        elif isinstance(vv, str) and vv.lstrip("-").isdigit():
            val = int(vv)
        else:
            # last resort: let it pass through (dataset evaluator may raise helpfully)
            try:
                val = int(vv)  # may raise
            except Exception:
                raise ValueError(f"assignment value for {key} must be int-like, got {vv!r}")
        out[key] = val
    return out
