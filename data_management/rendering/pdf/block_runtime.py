import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ReportLab imports
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from data_management.rendering.samples_tracker import SamplesTracker

# ============================================================
# Sandbox Configuration
# ============================================================

# Expanded whitelist to prevent 'name not defined' errors in SOTA models
_ALLOWED_BUILTINS = {
    # Math & Numbers
    "abs": abs,
    "float": float,
    "int": int,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sum": sum,
    "divmod": divmod,

    # Logic & Types
    "bool": bool,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "type": type,
    "getattr": getattr,
    "hasattr": hasattr,
    "len": len,

    # Collections & Iteration
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "list": list,
    "map": map,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "tuple": tuple,
    "zip": zip,
    "next": next,
    "iter": iter,

    # String & Representation
    "str": str,
    "format": format,
    "repr": repr,
    "chr": chr,
    "ord": ord,
}

# ============================================================
# Utilities & Parsing
# ============================================================

_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")

def _safe_eval_args(arg_src: str) -> Tuple[List[Any], Dict[str, Any]]:
    args, kwargs = [], {}
    text = arg_src.strip()
    if not text: return args, kwargs
    try:
        node = ast.parse(f"f({text})", mode="eval").body
        if not isinstance(node, ast.Call): raise ValueError("Invalid call")
        for a in node.args:
            if isinstance(a, (ast.Constant, ast.Dict, ast.List, ast.Tuple)):
                args.append(ast.literal_eval(a))
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, (ast.Constant, ast.Dict, ast.List, ast.Tuple)):
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        return args, kwargs
    except Exception as e:
        raise ValueError(f"Argument parse error: {e}") from e

def parse_block_call(call_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    m = _CALL_RE.match(call_str)
    if not m:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", call_str.strip()):
            return call_str.strip(), [], {}
        raise ValueError(f"Invalid block call: {call_str}")
    name, arg_src = m.group(1), m.group(2)
    pos, kw = _safe_eval_args(arg_src)
    return name, pos, kw

# ============================================================
# Compilation Logic (The Bulletproof Sandbox)
# ============================================================

def compile_llm_blocks(code: str, *, llm: Any) -> Dict[str, Callable[..., None]]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Syntax Error in LLM code: {e}") from e

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            raise ValueError(f"Illegal statement: {type(node).__name__}. Only 'def' allowed.")

    # Bound tool for text generation
    def generate_text_paragraph(prompt: str) -> str:
        return llm.make_call(prompt)

    # REPAIR: Support both generate_text_paragraph() AND llm.generate_text_paragraph()
    # SOTA models often assume tools are methods of the passed-in objects.
    if not hasattr(llm, 'generate_text_paragraph'):
        setattr(llm, 'generate_text_paragraph', generate_text_paragraph)

    safe_globals: Dict[str, Any] = {
        "__builtins__": _ALLOWED_BUILTINS,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
        "colors": colors,
        "generate_text_paragraph": generate_text_paragraph,
    }
    safe_locals: Dict[str, Any] = {}
    exec(compile(tree, filename="<llm_blocks>", mode="exec"), safe_globals, safe_locals)

    return {name: obj for name, obj in safe_locals.items() if callable(obj)}

# ============================================================
# Block Registry & Defaults
# ============================================================

@dataclass
class BlockRegistry:
    funcs: Dict[str, Callable[..., None]]
    def __init__(self): self.funcs = {}
    def register(self, name: str, fn: Callable[..., None]): self.funcs[name] = fn

    def call(self, name: str, **kwargs):
        fn = self.funcs.get(name)
        if not fn: raise KeyError(f"Block not found: {name}")
        fn(**kwargs)

# --- Built-in Implementations ---

def title_block(*, theme, tracker, story, styles, llm, **kwargs):
    title = theme.get("title") or kwargs.get("title") or "Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 10))

def intro_paragraph(*, theme, tracker, story, styles, llm, text: str = "", **kwargs):
    p = text or "Automated assay analysis."
    story.append(Paragraph(p.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 8))

def write_paragraph(*, theme, tracker, story, styles, llm, prompt: str = "", **kwargs):
    if not prompt: return
    # Use global helper if llm object is missing tool attribute
    func = getattr(llm, 'generate_text_paragraph', None) or (lambda p: llm.make_call(p))
    text = func(prompt).replace("\n", "<br/>")
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 8))

def labels_overview(*, theme, tracker, story, styles, llm, **kwargs):
    from collections import Counter
    counts = Counter([tracker.get_sample_metadata(i).get("label") for i in tracker.get_all_indices()])
    story.append(Paragraph("Label Distribution", styles["Heading3"]))
    rows = [("Label", "Count")] + [(str(k), str(v)) for k, v in sorted(counts.items())]
    t = Table(rows, hAlign="LEFT", colWidths=[120, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

def display_samples_in_range(*, tracker, story, styles, start=1, end=None, **kwargs):
    total = tracker.get_total_count()
    s_idx, e_idx = max(0, start - 1), total if end is None else min(end, total)
    indices = list(range(s_idx, e_idx))
    samples = tracker.get_samples_for_rendering(indices)
    for idx, s in zip(indices, samples):
        story.append(Paragraph(f"<b>#{idx + 1}</b>: {s['instance_text']}", styles["BodyText"]))
        story.append(Spacer(1, 8))

def get_default_registry() -> BlockRegistry:
    reg = BlockRegistry()
    reg.register("title_block", title_block)
    reg.register("intro_paragraph", intro_paragraph)
    reg.register("write_paragraph", write_paragraph)
    reg.register("labels_overview", labels_overview)
    reg.register("display_samples_in_range", display_samples_in_range)
    return reg