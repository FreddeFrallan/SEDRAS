from data_management.dataset import DatasetInstance
from typing import Any, Dict, List
from pathlib import Path
from enum import Enum


def _infer_variable_order(dataset: DatasetInstance) -> List[str]:
    """
    Prefer udd.variables keys if available; otherwise infer from the first sample assignment.
    """
    try:
        vars_from_udd = sorted(getattr(dataset.udd, "variables").keys())
        if vars_from_udd:
            return vars_from_udd
    except Exception:
        pass

    if not dataset.samples:
        raise ValueError("Dataset has no samples.")
    a0 = dataset.samples[0].assignment
    if not a0:
        raise ValueError("First sample has empty assignment.")
    return sorted(a0.keys())


def _schema_from_dataset(
    dataset: DatasetInstance,
    variable_order: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a compact schema per variable using observed integer values and text_mapping names/roles if present.

    Returns:
      { var_name: {"values": [...], "role": str|None, "labels": {int: str}|None } }
    """
    text_mapping = dataset.text_mapping or {}
    tm_vars = text_mapping.get("variables", {}) if isinstance(text_mapping, dict) else {}

    values_seen: Dict[str, set] = {v: set() for v in variable_order}
    for s in dataset.samples:
        for v in variable_order:
            if v in s.assignment:
                values_seen[v].add(int(s.assignment[v]))

    schema: Dict[str, Dict[str, Any]] = {}
    for v in variable_order:
        role = None
        labels = None
        if v in tm_vars:
            vinfo = tm_vars[v]
            role = vinfo.get("role")
            cats = vinfo.get("categories")
            if isinstance(cats, dict):
                labels = {int(k): str(val) for k, val in cats.items()}
        schema[v] = {
            "values": sorted(values_seen[v]),
            "role": role,
            "labels": labels,
        }
    return schema


def get_output_label_names(dataset: DatasetInstance) -> Dict[int, str]:
    """Return a mapping from output label index to a human-readable name."""

    tm_labels = (dataset.text_mapping or {}).get("output_labels", {}) if dataset.text_mapping else {}
    num_labels = int(getattr(dataset.udd, "num_output_labels", 2))

    names: Dict[int, str] = {}
    for lbl in range(num_labels):
        name = tm_labels.get(str(lbl)) if isinstance(tm_labels, dict) else None
        if name is None and isinstance(tm_labels, dict) and lbl in tm_labels:
            name = tm_labels.get(lbl)
        names[lbl] = str(name) if name is not None else f"Label {lbl}"

    return names

def get_schema_lines_from_dataset(dataset_instance: DatasetInstance) -> str:
    variable_order = _infer_variable_order(dataset_instance)
    schema = _schema_from_dataset(dataset_instance, variable_order)

    labels = get_output_label_names(dataset_instance)
    tm_labels = (dataset_instance.text_mapping or {}).get("output_labels", {}) if dataset_instance.text_mapping else {}
    if isinstance(tm_labels, dict) and len(tm_labels) > 0:
        label_line = ", ".join(str(labels[idx]) for idx in sorted(labels.keys()))
    else:
        label_line = ", ".join(f"{idx}:{name}" for idx, name in labels.items())

    lines = [
        f"Output labels ({len(labels)} categories): {label_line}",
        "The Following Variable Should Be Considered (integers are authoritative):",
    ]
    for v in variable_order:
        entry = schema[v]
        # print(entry)
        role = entry.get("role")
        labels = entry.get("labels")
        role_str = f" ({role})" if role else ""
        if labels:
            names_map = list(labels.keys())
            lines.append(f"- {v}{role_str}: categories={{ " + ", ".join(f'{k}:{labels[k]}' for k in names_map) + " }}")
        else:
            lines.append(f"- {v}{role_str}")
    return "\n".join(lines)


def _normalize_one_sample(s: Any) -> Dict[str, Any]:
    """
    Accepts either a DatasetSample or a dict-like sample and returns a plain dict with
    the keys expected by hard_evaluation(...): assignment, label, score, instance_text, instance_type.

    Extra optional fields such as numerical_values, property_labels and property_scores
    are preserved when present so downstream consumers can react to auxiliary outputs.
    """
    # Case A: already a dict-like
    if isinstance(s, dict):
        # ensure required keys exist (best-effort; missing score/text/type are fine)
        return {
            "assignment": s.get("assignment"),
            "label": s.get("label"),
            "score": s.get("score"),
            "instance_text": s.get("instance_text"),
            "instance_label_text": s.get("instance_label_text"),
            "instance_type": s.get("instance_type"),
            "numerical_values": s.get("numerical_values"),
            "property_labels": s.get("property_labels"),
            "property_scores": s.get("property_scores"),
        }

    # Case B: DatasetSample object (or duck-typed equivalent)
    # We don't crash if DatasetSample isn't importable; we just try attributes.
    assignment = getattr(s, "assignment", None)
    label = getattr(s, "label", None)
    score = getattr(s, "score", None)
    instance_text = getattr(s, "instance_text", None)
    instance_label_text = getattr(s, "instance_label_text", None)
    instance_type = getattr(s, "instance_type", None)
    numerical_values = getattr(s, "numerical_values", None)
    property_labels = getattr(s, "property_labels", None)
    property_scores = getattr(s, "property_scores", None)

    return {
        "assignment": assignment,
        "label": label,
        "score": score,
        "instance_text": instance_text,
        "instance_label_text": instance_label_text,
        "instance_type": instance_type,
        "numerical_values": numerical_values,
        "property_labels": property_labels,
        "property_scores": property_scores,
    }


def _normalize_samples(samples: List[Any]) -> List[Dict[str, Any]]:
    return [_normalize_one_sample(s) for s in samples]
