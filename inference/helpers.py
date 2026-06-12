from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, List

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_last_code_block(s: str) -> str:
    """Return the LAST fenced code block from the response."""

    blocks = _CODE_BLOCK_RE.findall(s)
    return blocks[-1] if blocks else ""


def _load_classify_function_safely(code: str) -> Callable[[List[float]], Dict[str, Any]]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code from LLM: {e}") from e

    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(func_nodes) != 1 or func_nodes[0].name != "classify":
        raise ValueError("Expected exactly one function named 'classify'.")

    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            raise ValueError("Imports and classes are not allowed.")
        if not isinstance(n, ast.FunctionDef):
            raise ValueError("Only a single function definition is allowed.")

    allowed_builtins = {
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "int": int,
        "float": float,
        "range": range,
        "all": all,
        "any": any,
    }
    safe_globals: Dict[str, Any] = {"__builtins__": allowed_builtins}
    safe_locals: Dict[str, Any] = {}

    compiled = compile(tree, filename="<llm_classify>", mode="exec")
    exec(compiled, safe_globals, safe_locals)

    fn = safe_locals.get("classify") or safe_globals.get("classify")
    if not callable(fn):
        raise ValueError("Loaded object 'classify' is not callable.")

    def wrapper(x: List[float]) -> Dict[str, Any]:
        """Validate that the classifier returns a dict with an integer label."""

        y = fn(x)

        # Normalize output to a dict so callers always receive classify(x) -> dict
        if isinstance(y, dict):
            result = dict(y)
        else:
            result = {"label": y}

        if "label" not in result:
            raise ValueError("Classifier dictionary output must include a 'label' key.")

        try:
            result["label"] = int(result["label"])
        except Exception as e:
            raise ValueError(
                f"Classifier 'label' must be int-castable; got {result['label']!r} ({type(result['label']).__name__})."
            ) from e

        return result

    return wrapper
