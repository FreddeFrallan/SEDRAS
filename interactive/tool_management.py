# tool_management.py
"""
Utilities for generating and translating LLM tool definitions.

Key points:
- Keep everything as plain JSON dictionaries/lists; avoid SDK objects here.
- generate_tool_definition(func) -> OpenAI-style tool dict (pure JSON).
- convert_openai_to_gemini_tools(tools) -> Gemini-style tool list (pure JSON).
- parse_tool_call(tool_call) -> (function_name, parsed_arguments_dict, error_message).
"""

from __future__ import annotations

import inspect
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    get_origin,
    get_args,
    Union,
    Literal,
)
try:
    from docstring_parser import parse, Style
except ImportError:
    class Style:
        GOOGLE = "google"

    class _ParsedDocstring:
        short_description = ""
        long_description = ""
        params = []

    def parse(docstring: str, style: str = Style.GOOGLE):
        parsed = _ParsedDocstring()
        lines = [line.strip() for line in (docstring or "").strip().splitlines()]
        parsed.short_description = lines[0] if lines else ""
        parsed.long_description = "\n".join(lines[1:]).strip()
        return parsed
from copy import deepcopy

# Public type alias for the OpenAI-style tool definition
ToolDefinition: TypeAlias = Dict[str, Any]

__all__ = [
    "ToolDefinition",
    "generate_tool_definition",
    "convert_openai_to_gemini_tools",
    "parse_tool_call",
]


# =========================
# JSON safety / validations
# =========================

def _ensure_jsonable(obj: Any) -> Any:
    """
    Raise a helpful error if obj isn't JSON-serializable.
    Returns obj unchanged if it is serializable.
    """
    try:
        json.dumps(obj)
    except TypeError as e:
        raise TypeError(
            f"Non-JSON-serializable value encountered: {e}\nObject: {repr(obj)}"
        )
    return obj


# =========================
# Small JSON helpers
# =========================

def _maybe_json_loads_value(x: Any) -> Any:
    """
    If x is a JSON-looking string (object/array), try json.loads; otherwise return x.
    This helps undo double-encoding of tool arguments.
    """
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


# ==================================
# Type → JSON Schema (improved)
# ==================================

def _map_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """
    Maps Python type hints to a JSON Schema dict.
    Defaults conservatively to {"type":"string"} if unknown.
    """

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Optional[T] / Union[T, None]
    if origin is Union and any(a is type(None) for a in args):
        non_none = [a for a in args if a is not type(None)]
        return _map_type_to_json_schema(non_none[0]) if non_none else {"type": "null"}

    # Literal[...] -> enum
    if origin is Literal:
        enum_vals = list(args)
        _ensure_jsonable(enum_vals)
        return {"enum": enum_vals}

    # Primitives
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}
    if py_type is str:
        return {"type": "string"}

    # Arrays / sequences: List[T], Tuple[T], etc.
    if origin in (list, tuple) or py_type in (list, tuple):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _map_type_to_json_schema(item_type)}

    # Objects / mappings: Dict[K, V]
    if origin in (dict,) or py_type is dict:
        # OpenAI tolerates {"type":"object","additionalProperties": <schema>},
        # Gemini rejects 'additionalProperties'; we strip it in the converter.
        value_type = args[1] if len(args) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": _map_type_to_json_schema(value_type),
        }

    # Fallback: treat as string
    return {"type": "string"}


# ==================================
# Tool Generation (from Python code)
# ==================================

def generate_tool_definition(func: Callable) -> ToolDefinition:
    """
    Generates an OpenAI-compatible tool definition from a Python function.
    Reads name, docstring, and signature. Returns a pure-JSON dict:

    {
      "type": "function",
      "function": {
        "name": "...",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {...},
          "required": [...],
          "additionalProperties": false
        }
      }
    }
    """
    try:
        sig = inspect.signature(func)
    except ValueError:
        raise TypeError(f"Cannot get signature for {func.__name__}. Is it a Python function?")

    # Parse docstring for descriptions
    docstring = inspect.getdoc(func) or ""
    parsed_docstring = parse(docstring, Style.GOOGLE)

    func_description = (parsed_docstring.short_description or "").strip()
    if parsed_docstring.long_description:
        long_desc = parsed_docstring.long_description.strip()
        if long_desc:
            func_description = (
                f"{func_description}\n{long_desc}" if func_description else long_desc
            )
    if not func_description:
        func_description = f"No description for {func.__name__}"

    # Per-parameter descriptions from docstring (if present)
    param_descriptions = {
        p.arg_name: (p.description or "").strip() for p in parsed_docstring.params
    }

    schema_properties: Dict[str, Any] = {}
    required_params: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = (
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else str
        )
        schema_type = _map_type_to_json_schema(annotation)

        if name in param_descriptions and param_descriptions[name]:
            schema_type["description"] = param_descriptions[name]

        if param.default is not inspect.Parameter.empty:
            # default must be JSON-encodable
            try:
                json.dumps(param.default)
                schema_type["default"] = param.default
            except TypeError:
                schema_type["default"] = str(param.default)
        else:
            required_params.append(name)

        schema_properties[name] = schema_type

    tool_def: ToolDefinition = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_description,
            "parameters": {
                "type": "object",
                "properties": schema_properties,
                "required": required_params,
                "additionalProperties": False,  # OK for OpenAI; stripped for Gemini
            },
        },
    }

    return _ensure_jsonable(tool_def)


# ==================================
# Gemini Translation (pure dicts)
# ==================================

def _strip_additional_properties(schema: Any) -> Any:
    """
    Recursively remove 'additionalProperties' keys from a JSON schema,
    because Gemini rejects them.
    """
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "additionalProperties":
                continue
            out[k] = _strip_additional_properties(v)
        return out
    if isinstance(schema, list):
        return [_strip_additional_properties(x) for x in schema]
    return schema


def convert_openai_to_gemini_tools(tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Translate OpenAI-style tools into the Gemini REST shape, returning **pure dicts**.

    Gemini expects:
      [
        {
          "function_declarations": [
            {"name": "...", "description": "...", "parameters": {JSON Schema}},
            ...
          ]
        }
      ]

    Additionally, Gemini rejects 'additionalProperties', so we strip it recursively.
    """
    if not tools:
        return None

    declarations: List[Dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue

        desc = fn.get("description", "") or ""
        params = fn.get("parameters") or {"type": "object", "properties": {}}

        # Sanitize schema for Gemini
        params_clean = _strip_additional_properties(deepcopy(params))

        declarations.append({
            "name": name,
            "description": desc,
            "parameters": params_clean,
        })

    return [{"function_declarations": declarations}] if declarations else None


# ==================================
# Tool-call parsing (LLM responses)
# ==================================

def parse_tool_call(
    tool_call: Dict[str, Any],
    *,
    arg_normalizers: Optional[
        Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]
    ] = None,
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """
    Parse a single tool_call object from an LLM response into:

        (function_name, parsed_arguments_dict, error_message)

    Behavior:
      - Extracts function name and arguments from OpenAI/Gemini-style tool_call dict.
      - Safely json.loads the arguments string.
      - Recursively tries to json-decode top-level values (to undo double-encoding).
      - Optionally runs a per-function normalizer, e.g. to normalize 'assignment'.
      - NEVER raises: instead returns an error_message string if something went wrong.

    Returns:
        (function_name, args_dict, error_message)

        - function_name may be None if missing.
        - args_dict is best-effort parsed arguments (possibly empty).
        - error_message is None if everything looks ok; otherwise a human-readable string
          describing what went wrong, suitable to send back to the LLM in a tool response.
    """
    error_message: Optional[str] = None

    fn_dict = tool_call.get("function")
    if not isinstance(fn_dict, dict):
        return None, {}, "Malformed tool call: 'function' field is missing or not an object."

    function_name = fn_dict.get("name")
    if not function_name:
        error_message = "Malformed tool call: 'function.name' is missing."

    raw_args = fn_dict.get("arguments", "{}")

    # Base JSON parse of the arguments string (or dict)
    function_args: Dict[str, Any] = {}
    if isinstance(raw_args, str):
        try:
            function_args = json.loads(raw_args or "{}")
        except Exception as e:
            function_args = {}
            msg = f"Failed to parse tool call arguments as JSON: {e}. Raw: {raw_args!r}"
            error_message = msg if error_message is None else f"{error_message} | {msg}"
    elif isinstance(raw_args, dict):
        function_args = raw_args
    else:
        function_args = {}
        msg = f"Tool call arguments must be a JSON string or object, got: {type(raw_args).__name__}"
        error_message = msg if error_message is None else f"{error_message} | {msg}"

    # Undo double-encoding at top-level and normalize payloads
    for k, v in list(function_args.items()):
        function_args[k] = _maybe_json_loads_value(v)

    # Apply optional per-function normalizer
    if arg_normalizers and function_name in arg_normalizers:
        try:
            function_args = arg_normalizers[function_name](function_args)
        except Exception as e:
            msg = f"Error in normalizer for '{function_name}': {e}"
            error_message = msg if error_message is None else f"{error_message} | {msg}"

    return function_name, function_args, error_message
