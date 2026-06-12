from data_management.utils import get_schema_lines_from_dataset, get_output_label_names
from data_management.dataset import DatasetInstance, RepresentationLevel
from typing import Any, Dict, List, Optional
import random



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


def _get_property_label_names(dataset: DatasetInstance) -> Dict[str, Dict[int, str]]:
    """Build a mapping from property name -> {label_idx: label_name}.

    Falls back to generic "Label {i}" names when no metadata is available.
    """

    names: Dict[str, Dict[int, str]] = {}

    meta_props = dataset.metadata.get("output_properties", {}) if isinstance(dataset.metadata, dict) else {}
    output_props = getattr(dataset, "output_properties", {}) or {}
    tm_props = (
        dataset.text_mapping.get("output_properties", {})
        if isinstance(dataset.text_mapping, dict)
        else {}
    )

    property_keys = set(meta_props.keys()) | set(output_props.keys()) | set(tm_props.keys())

    # If no metadata exists, infer property names from the samples themselves.
    if not property_keys:
        for s in dataset.samples:
            for pname in (getattr(s, "property_labels", {}) or {}).keys():
                property_keys.add(pname)

    for pname in sorted(property_keys):
        # First priority: explicit representation mapping for properties
        tm_labels = tm_props.get(pname) if isinstance(tm_props, dict) else None
        if isinstance(tm_labels, dict) and tm_labels:
            numeric_keys = {k: v for k, v in tm_labels.items() if str(k).isdigit()}
            if numeric_keys:
                names[pname] = {int(k): str(v) for k, v in numeric_keys.items()}
                continue

        num_labels: Optional[int] = None

        if pname in meta_props:
            try:
                num_labels = int(meta_props[pname].get("num_output_labels"))
            except Exception:
                num_labels = None

        if num_labels is None and pname in output_props:
            try:
                num_labels = int(getattr(output_props[pname], "num_output_labels", None))
            except Exception:
                num_labels = None

        # Fallback: infer the maximum observed category from the samples
        if num_labels is None:
            observed: List[int] = []
            for s in dataset.samples:
                val = (getattr(s, "property_labels", {}) or {}).get(pname)
                if val is not None:
                    observed.append(int(val))
            if observed:
                num_labels = max(observed) + 1

        if num_labels is None or num_labels < 1:
            num_labels = 1

        names[pname] = {idx: str(idx) for idx in range(int(num_labels))}

    return names


def _get_property_hint(
    dataset: DatasetInstance, *, num_hints: int = 3
) -> Dict[str, Any]:
    """Return property hints tailored to the dataset's representation level.

    For RAW datasets, this mirrors ``_get_property_label_names``. For textualized datasets,
    it samples ``num_hints`` values for each property to provide concise hints.
    """

    property_label_names = _get_property_label_names(dataset)

    if dataset.representation_level == RepresentationLevel.RAW:
        return property_label_names

    hints: Dict[str, List[str]] = {}
    for pname, label_map in property_label_names.items():
        values = list(label_map.values())
        if not values:
            continue
        sample_size = min(num_hints, len(values))
        hints[pname] = random.sample(values, sample_size)

    return hints


def _render_label_and_properties(
    *,
    label_idx: int,
    label_names: Dict[int, str],
    property_labels: Dict[str, Optional[int]],
    property_label_names: Dict[str, Dict[int, str]],
    use_textual_only: bool,
) -> str:
    """Render the main label plus any property labels for a sample."""

    label_text = label_names.get(label_idx, str(label_idx))
    label_parts = [label_text]

    for prop, raw_val in sorted((property_labels or {}).items()):
        if raw_val is None:
            continue
        try:
            val_idx = int(raw_val)
        except Exception:
            continue

        prop_names = property_label_names.get(prop, {})
        prop_text = prop_names.get(val_idx, str(val_idx))
        if use_textual_only:
            rendered = prop_text
        else:
            rendered = f"{prop}={prop_text}"
        label_parts.append(rendered)

    return ", ".join(label_parts)

def _pack_examples_block(
    dataset: DatasetInstance, *, max_examples: Optional[int] = None
) -> str:
    if dataset.representation_level == RepresentationLevel.FILES or (
        getattr(dataset, "rendered_documents", None)
        and dataset.representation_level != RepresentationLevel.RAW
    ):
        return (
            "Labeled examples are provided exclusively via the attached document(s). "
            "Review the PDF files to extract every sample; no textual summaries are embedded in this prompt."
        )
    # If no samples are present, return empty string
    if not dataset.samples:
        return ""

    if max_examples is not None and max_examples <= 0:
        return ""

    tm = dataset.text_mapping or {}
    label_names = get_output_label_names(dataset)
    tm_labels = (dataset.text_mapping or {}).get("output_labels", {}) if dataset.text_mapping else {}
    use_textual_only = isinstance(tm_labels, dict) and len(tm_labels) > 0
    property_label_names = _get_property_label_names(dataset)
    lines: List[str] = ["Labeled examples:"]
    samples = dataset.samples
    if max_examples is not None:
        samples = samples[:max_examples]

    for s in samples:
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
        label_block = _render_label_and_properties(
            label_idx=label,
            label_names=label_names,
            property_labels=getattr(s, "property_labels", {}) or {},
            property_label_names=property_label_names,
            use_textual_only=use_textual_only,
        )
        lines.append(f"Labels: {label_block} | {text if text else '(no text)'}")
    return "\n".join(lines)
