# generate_eval_prompt.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from data_management.dataset import DatasetInstance  # adjust import if your path differs
from data_management.utils import get_schema_lines_from_dataset, get_output_label_names

# -----------------------
# Internal helpers (self-contained)
# -----------------------

def _render_raw_samples(sample: Any) -> str:
    parts = [f"{k}={v}" for k, v in sample.assignment.items()]
    return ", ".join(parts)

def _render_from_assignment(text_mapping: Dict[str, Any], assignment: Dict[str, int]) -> Optional[str]:
    if not text_mapping:
        return None
    template = text_mapping.get("template")
    variables = text_mapping.get("variables")
    if not template or not variables:
        return None

    out = template
    for vname, ival in assignment.items():
        cats = variables.get(vname, {}).get("categories", {})
        replacement = str(cats.get(str(int(ival)), ival))
        out = out.replace("{var." + vname + "}", replacement)
    return out


def _render_output_labels(dataset: DatasetInstance) -> str:
    label_names = get_output_label_names(dataset)
    tm_labels = (dataset.text_mapping or {}).get("output_labels", {}) if dataset.text_mapping else {}
    use_textual_only = isinstance(tm_labels, dict) and len(tm_labels) > 0

    if use_textual_only:
        segments = [str(label_names[idx]) for idx in sorted(label_names.keys())]
    else:
        segments = [f"{idx}: {name}" for idx, name in label_names.items()]
    return f"Output labels ({len(label_names)} categories): " + ", ".join(segments)

def _pack_examples_block(dataset: DatasetInstance) -> str:
    # If no samples are present, return empty string
    if not dataset.samples:
        return ""

    tm = dataset.text_mapping or {}
    label_names = get_output_label_names(dataset)
    tm_labels = (dataset.text_mapping or {}).get("output_labels", {}) if dataset.text_mapping else {}
    use_textual_only = isinstance(tm_labels, dict) and len(tm_labels) > 0
    lines: List[str] = ["Labeled examples:"]
    for s in dataset.samples:
        label = int(s.label)
        label_text = label_names.get(label, str(label))
        text = (s.instance_text or "").strip() if s.instance_text else ""
        if not text:
            if not tm:
                text = _render_raw_samples(s)
            else:
                text = _render_from_assignment(tm, s.assignment) or ""
        if text:
            text = text.replace("\n", " ")
        if use_textual_only:
            label_display = label_text
        else:
            label_display = f"{label} ({label_text})"
        lines.append(f"Label: {label_display} | {text if text else '(no text)'}")
    return "\n".join(lines)


# -----------------------
# Public prompt builders (one per setting)
# -----------------------

def make_theory_prompt_only_task(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
) -> str:
    """
    Minimal prompt: only task + examples (+ high-level theme if present).
    """
    tm = dataset.text_mapping or {}
    instance_type = tm.get("instance_type")
    theme_summary = tm.get("theme_summary")
    instance_hint_used = dataset.instance_hint_used
    label_line = _render_output_labels(dataset)

    parts: List[str] = []
    parts.append(
        "Task: provide a clear, deterministic textual theory that can be used to derive the correct output label for each scenario."
    )
    parts.append(label_line)
    parts.append("The theory must explain all provided labeled examples.")
    if instance_type:
        parts.append(f"Instance type: {instance_type}")
    if theme_summary:
        parts.append(f"Theme summary: {theme_summary}")
    if instance_hint_used:
        parts.append(f"Instance hint used: {instance_hint_used}")

    parts.append("")
    parts.append(_pack_examples_block(dataset))
    parts.append("")
    parts.append(
        "THEORY requirements:\n"
        f"- Provide a clear textual description of the decision logic that maps integer variables to the correct label among: {label_line}.\n"
        "- The theory must be unambiguous and fully deterministic.\n"
        "- Do not provide code, formulas, or pseudocode.\n"
    )
    return "\n".join(parts)


def make_theory_prompt_variable_names(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
) -> str:
    """
    Includes variable names/roles/labels (schema), but no code-implementation constraints.
    """
    tm = dataset.text_mapping or {}
    instance_type = tm.get("instance_type")
    theme_summary = tm.get("theme_summary")
    instance_hint_used = dataset.instance_hint_used
    label_line = _render_output_labels(dataset)

    parts: List[str] = []
    parts.append(
        "Task: provide a clear, deterministic textual theory that can be used to derive the correct output label for each scenario."
    )
    parts.append(label_line)
    parts.append("The theory must explain all provided labeled examples.")
    if instance_type:
        parts.append(f"Instance type: {instance_type}")
    if theme_summary:
        parts.append(f"Theme summary: {theme_summary}")
    if instance_hint_used:
        parts.append(f"Instance hint used: {instance_hint_used}")

    parts.append("")
    parts.append(get_schema_lines_from_dataset(dataset))

    parts.append("")
    parts.append(_pack_examples_block(dataset))
    parts.append("")
    parts.append(
        "THEORY requirements:\n"
        f"- Provide a clear textual description of the decision logic that maps integer variables to the correct label among: {label_line}.\n"
        "- The theory must be unambiguous and fully deterministic.\n"
        "- Do not provide code, formulas, or pseudocode.\n"
    )
    return "\n".join(parts)


def make_theory_prompt_variable_names_and_code_info(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
) -> str:
    """
    Includes schema + explicit implementation constraints that will later be used by the compiler stage.
    """
    tm = dataset.text_mapping or {}
    instance_type = tm.get("instance_type")
    theme_summary = tm.get("theme_summary")
    instance_hint_used = dataset.instance_hint_used
    label_line = _render_output_labels(dataset)

    parts: List[str] = []
    parts.append(
        "Task: provide a clear, deterministic textual theory that can be used to derive the correct output label for each scenario."
    )
    parts.append(label_line)
    parts.append("The theory must explain all provided labeled examples.")
    if instance_type:
        parts.append(f"Instance type: {instance_type}")
    if theme_summary:
        parts.append(f"Theme summary: {theme_summary}")
    if instance_hint_used:
        parts.append(f"Instance hint used: {instance_hint_used}")

    parts.append("")
    parts.append(get_schema_lines_from_dataset(dataset))

    parts.append("")
    parts.append(_pack_examples_block(dataset))
    parts.append("")
    parts.append(
        "THEORY requirements:\n"
        f"- Provide a clear textual description of the decision logic that maps integer variables to the correct label among: {label_line}.\n"
        "- The theory must be unambiguous and fully deterministic.\n"
        "- Do not provide code, formulas, or pseudocode in this section.\n"
        "\n"
        "Implementation constraints (for your awareness):\n"
        "- At a later stage, your produced theory will be compiled into a function with signature: def classify(x: list[float]) -> int.\n"
        f"- Variable order will be exactly: {variable_order}\n"
        "- Each x[i] is a category or numeric value: categorical variables use integer categories; numerical variables use floating-point values.\n"
        "- Avoid fuzzy notions; specify concrete, checkable rules (equality, membership, ordering if applicable).\n"
    )
    return "\n".join(parts)


def make_theory_prompt_code_without_variable_info(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
) -> str:
    """
    Includes schema + explicit implementation constraints that will later be used by the compiler stage.
    """
    tm = dataset.text_mapping or {}
    instance_type = tm.get("instance_type")
    theme_summary = tm.get("theme_summary")
    instance_hint_used = dataset.instance_hint_used
    label_line = _render_output_labels(dataset)

    parts: List[str] = []
    parts.append(
        "Task: provide a clear, deterministic textual theory that can be used to derive the correct output label for each scenario."
    )
    parts.append(label_line)
    parts.append("The theory must explain all provided labeled examples.")
    if instance_type:
        parts.append(f"Instance type: {instance_type}")
    if theme_summary:
        parts.append(f"Theme summary: {theme_summary}")
    if instance_hint_used:
        parts.append(f"Instance hint used: {instance_hint_used}")

    parts.append("")
    parts.append(_pack_examples_block(dataset))
    parts.append("")
    parts.append(
        "THEORY requirements:\n"
        f"- Provide a clear textual description of the decision logic that maps integer variables to the correct label among: {label_line}.\n"
        "- The theory must be unambiguous and fully deterministic.\n"
        "- Do not provide code, formulas, or pseudocode in this section.\n"
        "\n"
        "Implementation constraints (for your awareness):\n"
        "- At a later stage, your produced theory will be compiled into a function with signature: def classify(x: list[float]) -> int.\n"
        "- Inputs will be integer categories for categorical variables and floating-point values for numerical variables.\n"
        "- It is therefore important that you explicitly name each variable that you are referring to in your theory.\n"
    )
    return "\n".join(parts)
