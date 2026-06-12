"""
Create an LLM-generated representation mapping *from a GeneratedDataset* (uses gds.udd).

- Calls your OpenAIChatWrapper to propose:
    * instance_type (theme title)
    * theme_summary
    * variables: each var gets a 'role' and per-category labels
    * template: uses {var.VAR_NAME} placeholders, includes every var exactly once
    * rendering_guidelines (fixed, concise)

- Attaches the mapping to the dataset object (gds.text_mapping) and records the hint (gds.instance_hint_used).
- Returns the mapping as a JSON string (no file I/O).

Programmatic:
  from create_text_mapping_from_udd import (
      generate_text_mapping_json_from_dataset,
      add_text_mapping_to_dataset,
  )
"""

from __future__ import annotations

import json
import random
import warnings
from typing import Any, Dict, List, Optional

from inference.model_wrappers.llm_wrapper import LLMModel, get_llm_wrapper  # <-- change to your real import
from data_management.dataset import AbstractDataset  # <-- points to your new dataset class
from data_management.underlying_data.underlying_data import VariableType
from data_management.underlying_data.output_properties import OutputPropertyType

# Suggestion seeds for themes; one will be picked if style_hint is None.
INSTANCE_HINTS = [
    "crime investigation case summary",
    "medical diagnosis report",
    "restaurant customer review",
    "space exploration mission log",
    "scientific experiment notes",
    "fantasy world character profile",
    "corporate job interview transcript",
    "weather forecasting bulletin",
    "historical event chronicle",
    "sports competition commentary",
    "political election analysis",
    "school classroom attendance log",
    "romantic relationship diary entry",
    "military operation debrief",
    "video game quest description",
    "AI system diagnostic summary",
    "psychological personality assessment",
    "film or TV show plot synopsis",
    "archaeological site excavation record",
    "travel journal entry from a tourist",
    "legal court hearing transcript",
    "architectural blueprint specification",
    "culinary recipe instructions",
    "cybersecurity incident report",
    "financial quarterly earnings summary",
    "maritime shipping manifest",
    "music album production notes",
    "real estate property listing",
    "automotive repair workshop manual",
    "fashion runway show description",
    "wildlife observation field notes",
    "customer support ticketing log",
    "urban planning zoning report",
    "emergency room triage notes",
    "stock market technical analysis",
    "aerospace engineering stress test",
    "poetry anthology thematic breakdown",
    "software API documentation snippet",
    "industrial factory safety audit",
    "botanical garden species catalog",
    "non-profit grant proposal executive summary",
    "theatrical stage play stage directions",
    "insurance claim damage assessment",
    "cryptocurrency whitepaper abstract",
    "gym workout routine log",
    "podcast episode show notes",
    "retail inventory management report",
    "geological survey core sample analysis",
    "human resources employee performance review",
    "logistics supply chain disruption alert",
    "academic thesis abstract",
    "patent application technical description",
    "construction site daily progress report",
    "philanthropic foundation impact report",
    "social media influencer campaign brief",
    "e-commerce product return reasoning",
    "parliamentary debate hansard record",
    "veterinary clinical examination record",
    "hotel concierge guest request log",
    "astronomy observatory star chart notes",
    "public transport commuter survey",
    "biotechnology lab culture log",
    "marketing agency brand identity guidelines",
    "disaster relief coordination plan",
    "film festival submission metadata",
    "library book conservation report",
    "yoga retreat itinerary and goals",
    "renewable energy farm output log",
    "venture capital pitch deck summary",
    "oceanographic deep-sea sonar log",
    "nutritionist meal plan recommendation",
    "intellectual property infringement notice",
    "e-sports tournament bracket analysis",
    "telecommunications network outage report",
    "museum curator exhibition checklist",
    "farming harvest yield projection",
    "linguistic field study phonetic data",
    "software bug bounty submission",
    "real-time traffic congestion report",
    "mining operation mineral grade report",
    "board game rulebook clarification",
    "furniture assembly instruction step",
    "coffee roastery cupping notes",
    "environmental impact study summary",
    "interior design mood board description",
    "blockchain transaction ledger entry",
    "advertising copy A/B test results",
    "aviation flight deck pre-check list",
    "sociological urban migration study",
    "pharmaceutical clinical trial protocol",
    "waste management recycling efficiency log",
    "government census demographic breakdown",
    "charity gala seating arrangement logic",
    "appliance energy star rating report",
    "private investigator surveillance log",
    "classical music symphony score analysis",
    "renewable resource sustainability audit",
    "online course syllabus outline",
    "semiconductor manufacturing cleanroom log",
    "marathon race day split times"
]


# -------------------------------
# UDD → variables / categories
# -------------------------------

def _extract_var_cats_from_variable_obj(var_obj: Any) -> List[int]:
    cats = getattr(var_obj, "categories", None)
    if cats is not None:
        if isinstance(cats, dict):
            keys = []
            for k in cats.keys():
                try:
                    keys.append(int(k))
                except Exception:
                    return list(range(len(cats)))
            return sorted(set(keys))
        if isinstance(cats, list):
            return list(range(len(cats)))
        if isinstance(cats, int):
            return list(range(cats))
    for attr in ("num_categories", "n_categories", "n_cats"):
        n = getattr(var_obj, attr, None)
        if isinstance(n, int) and n > 0:
            return list(range(n))
    try:
        td = var_obj.to_dict()
    except Exception:
        td = None
    if isinstance(td, dict):
        if "categories" in td:
            c = td["categories"]
            if isinstance(c, dict):
                keys = []
                for k in c.keys():
                    try:
                        keys.append(int(k))
                    except Exception:
                        return list(range(len(c)))
                return sorted(set(keys))
            if isinstance(c, list):
                return list(range(len(c)))
        for k in ("weights", "scores", "weight_by_cat", "score_by_cat"):
            m = td.get(k)
            if isinstance(m, dict):
                keys = []
                for kk in m.keys():
                    try:
                        keys.append(int(kk))
                    except Exception:
                        return list(range(len(m)))
                return sorted(set(keys))
            if isinstance(m, list):
                return list(range(len(m)))
    # Fallback: assume binary indices
    return [0, 1]


def _gather_variable_metadata_from_udd(udd: Any) -> Dict[str, Dict[str, Any]]:
    """Return per-variable metadata needed for prompt + validation."""

    variables = getattr(udd, "variables", None)
    if not isinstance(variables, dict) or not variables:
        raise ValueError("UDD has no variables to extract.")

    meta: Dict[str, Dict[str, Any]] = {}
    for name, var in variables.items():
        try:
            vtype = getattr(var, "variable_type", VariableType.CATEGORICAL)
            if isinstance(vtype, str):
                vtype = VariableType(vtype)
        except Exception:
            vtype = VariableType.CATEGORICAL

        meta[name] = {
            "categories": _extract_var_cats_from_variable_obj(var),
            "variable_type": vtype,
        }
    return meta


def _gather_property_metadata_from_dataset(dataset: AbstractDataset) -> Optional[Dict[str, Dict[str, Any]]]:
    """Merge property metadata from dataset metadata and attached UDDs."""

    properties: Dict[str, Dict[str, Any]] = {}

    # Prefer explicit metadata attached to the dataset
    meta_props = {}
    try:
        meta_props = dataset.metadata.get("output_properties", {}) if isinstance(dataset.metadata, dict) else {}
    except Exception:
        meta_props = {}

    for name, meta in meta_props.items():
        if not isinstance(meta, dict):
            continue

        entry: Dict[str, Any] = {}
        ptype = str(meta.get("type") or meta.get("property_type") or "").lower()
        if ptype:
            entry["type"] = ptype
        if "num_output_labels" in meta:
            try:
                entry["num_output_labels"] = int(meta["num_output_labels"])
            except Exception:
                pass
        if "num_categories" in meta and "num_output_labels" not in entry:
            try:
                entry["num_output_labels"] = int(meta["num_categories"])
            except Exception:
                pass
        for key in ("min_value", "max_value", "unit", "description"):
            if key in meta:
                entry[key] = meta[key]

        if entry:
            properties[name] = entry

    # Fill in any missing categorical properties from attached UDDs
    for name, udd in getattr(dataset, "output_properties", {}).items():
        if name not in properties:
            try:
                num = int(getattr(udd, "num_output_labels", 0))
            except Exception:
                num = 0
            properties[name] = {
                "type": OutputPropertyType.CATEGORICAL.value,
                "num_output_labels": num,
            }
        else:
            entry = properties[name]
            entry.setdefault("type", OutputPropertyType.CATEGORICAL.value)
            if "num_output_labels" not in entry:
                try:
                    entry["num_output_labels"] = int(getattr(udd, "num_output_labels", 0))
                except Exception:
                    pass

    return properties or None


# -------------------------------
# Prompt building & validation
# -------------------------------

def _build_system_prompt() -> str:
    return (
        "You are an expert dataset theming and representation designer. "
        "Given variables with integer categories, produce one coherent theme. "
        "Return STRICT JSON only (no backticks or commentary)."
    )


def _build_user_prompt(
        var_metadata: Dict[str, Dict[str, Any]],
        style_hint: Optional[str],
        num_output_labels: int,
        property_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    lines = [
        "Create a single theme and per-variable/category mapping to textualize assignments.",
        "Variables with category indices:",
    ]
    for v in sorted(var_metadata.keys()):
        cats = var_metadata[v]["categories"]
        vtype = var_metadata[v]["variable_type"]
        type_label = "numerical spans" if vtype is VariableType.NUMERICAL else "categorical"
        lines.append(f"- {v}: {type_label} {cats}")

    if style_hint:
        lines += ["", f"Optional style hint: {style_hint}"]

    lines += [
        "",
        "Return STRICT JSON with schema:",
        "{",
        '  "instance_type": "short title, e.g., Crime Investigation Case Brief",',
        '  "theme_summary": "1-3 sentences describing the theme and mapping idea.",',
        '  "output_labels": { "0": "name for label 0", "1": "name for label 1", "...": "..." },',
        '  "output_properties": {',
        '      "prop_name": { "0": "label for 0", "1": "label for 1" },                    # categorical property',
        '      "numeric_prop": { "type": "numerical", "value_range": { "min": number, "max": number, "unit": "optional" }, "description": "short phrase" }',
        "  },",
        '  "variables": {',
        '     "<VAR_NAME>": {',
        '         "role": "what this variable represents in the theme",',
        '         "value_range": { "min": number, "max": number, "unit": "optional unit" },  # ONLY for numerical vars',
        '         "categories": { "0": "label for 0", "1": "label for 1", "...": "..." }',
        "     }, ...",
        "  },",
        '  "template": "a single sentence containing placeholders like {var.V0}, {var.V1}, ...",',
        "}",
        "",
        "Constraints:",
        "- Cover all listed category indices for each variable.",
        "- Provide short, distinct names for every output label index from 0 through "
        f"{num_output_labels - 1}.",
        "- If output_properties are listed below, provide names for each category index of every categorical property.",
        "- For numerical output properties, include a value_range with realistic min/max (min < max) and optional unit.",
        "- Keep category phrases short and unambiguous.",
        "- For numerical variables, choose a realistic numeric range that fits the theme and specify it via value_range (min < max).",
        "- Units are optional but recommended for clarity (e.g., '°C', 'mph').",
        "- The template MUST be a single natural-language sentence.",
        "- The template MUST contain one placeholder per variable, formatted exactly as {var.<VAR_NAME>}.",
        "- The template MUST NOT contain any additional information not derivable from those placeholders.",
        "",
        "Purpose of the template:",
        "- This template will later be used by another deterministic model that replaces each placeholder",
        "  with the mapped category phrase for that variable.",
        "- Therefore, the template must be simple, literal, and fully deterministic.",
        "- No creativity, no narrative expansions, no conditionals, no optional fragments.",
    ]

    if property_metadata:
        lines += [
            "",
            "Output properties (provide labels or ranges for each):",
        ]
        for prop, meta in sorted(property_metadata.items()):
            ptype = str(meta.get("type") or OutputPropertyType.CATEGORICAL.value).lower()
            if ptype == OutputPropertyType.NUMERICAL.value:
                min_v = meta.get("min_value")
                max_v = meta.get("max_value")
                unit = meta.get("unit")
                range_hint = ""
                if min_v is not None and max_v is not None:
                    range_hint = f" ~[{min_v}, {max_v}]"
                unit_hint = f" ({unit})" if unit else ""
                lines.append(
                    f"- {prop}: numerical property{range_hint}{unit_hint}. "
                    "Include a value_range (min/max and optional unit) and a one-phrase description."
                )
            else:
                count = meta.get("num_output_labels", 0) or 0
                lines.append(f"- {prop}: categorical indices {list(range(int(count)))}")

    return "\n".join(lines)


def _extract_json_block(text: str) -> str:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    raise ValueError("Unbalanced JSON in model response.")


def _normalize_value_range(raw_range: Any) -> Dict[str, Any]:
    if raw_range is None:
        raise ValueError("value_range missing")

    if isinstance(raw_range, dict):
        min_v = raw_range.get("min")
        max_v = raw_range.get("max")
        unit = raw_range.get("unit")
    elif isinstance(raw_range, (list, tuple)):
        if len(raw_range) < 2:
            raise ValueError("value_range list must have at least two entries")
        min_v, max_v = raw_range[:2]
        unit = raw_range[2] if len(raw_range) > 2 else None
    else:
        raise ValueError(f"Unsupported value_range type: {type(raw_range)!r}")

    try:
        min_f = float(min_v)
        max_f = float(max_v)
    except Exception as exc:
        raise ValueError(f"value_range min/max must be numeric: {raw_range!r}") from exc

    if max_f <= min_f:
        raise ValueError(f"value_range max must be greater than min: {raw_range!r}")

    norm = {"min": min_f, "max": max_f}
    if unit:
        norm["unit"] = str(unit)
    return norm


def _validate_mapping(
        mapping: Dict[str, Any],
        var_metadata: Dict[str, Dict[str, Any]],
        *,
        num_output_labels: int,
        property_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    for key in ("instance_type", "variables"):
        if key not in mapping:
            raise ValueError(f"Mapping JSON missing required key: {key}")

    output_labels = mapping.setdefault("output_labels", {})
    for lbl in range(num_output_labels):
        key = str(lbl)
        if key not in output_labels:
            output_labels[key] = f"Label {lbl}"
            warnings.warn(
                f"Missing output label name for index {lbl}; inserting default value."
            )

    variables = mapping.setdefault("variables", {})
    for v, meta in var_metadata.items():
        cats = meta["categories"]
        if v not in variables:
            variables[v] = {}
        cat_map = variables[v].setdefault("categories", {})
        missing = [str(c) for c in cats if str(c) not in cat_map]
        if missing:
            base_phrase = variables[v].get("role") or v
            for cat in missing:
                if meta["variable_type"] is VariableType.NUMERICAL:
                    cat_map[str(cat)] = ""
                else:
                    cat_map[str(cat)] = f"{base_phrase} category {cat}"
            warnings.warn(
                f"Variable '{v}' missing category mappings; added defaults for: {missing}"
            )

        if meta["variable_type"] is VariableType.NUMERICAL:
            normalized = _normalize_value_range(variables[v].get("value_range"))
            variables[v]["value_range"] = normalized

    # Validate optional output properties
    if property_metadata:
        output_props = mapping.setdefault("output_properties", {})
        for prop, meta in property_metadata.items():
            expected_type = str(meta.get("type") or OutputPropertyType.CATEGORICAL.value).lower()
            entry = output_props.get(prop, {})
            if expected_type == OutputPropertyType.NUMERICAL.value:
                if not isinstance(entry, dict):
                    entry = {}
                entry["type"] = OutputPropertyType.NUMERICAL.value
                raw_range = entry.get("value_range")
                fallback_range = None
                if raw_range is None and "min_value" in meta and "max_value" in meta:
                    fallback_range = {
                        "min": meta.get("min_value"),
                        "max": meta.get("max_value"),
                        **({"unit": meta["unit"]} if meta.get("unit") else {}),
                    }
                if raw_range is None and fallback_range is None:
                    fallback_range = {"min": 0.0, "max": 1.0}
                entry["value_range"] = _normalize_value_range(raw_range or fallback_range)
                entry.setdefault("description", meta.get("description") or f"{prop} value")
                output_props[prop] = entry
            else:
                labels = entry if isinstance(entry, dict) else {}
                if isinstance(labels, dict) and labels.get("type") == OutputPropertyType.NUMERICAL.value:
                    labels = {}
                count = int(meta.get("num_output_labels", 0) or 0)
                missing = [str(idx) for idx in range(count) if str(idx) not in labels]
                for idx in missing:
                    labels[str(idx)] = f"{prop} label {idx}"
                labels.setdefault("type", OutputPropertyType.CATEGORICAL.value)
                if missing:
                    warnings.warn(
                        f"Property '{prop}' missing labels for categories {missing}; inserted defaults."
                    )
                output_props[prop] = labels


# -------------------------------
# Public API (dataset-centric)
# -------------------------------

def generate_text_mapping_json_from_dataset(
        dataset: AbstractDataset,
        style_hint: Optional[str] = None,
        indent: int = 2,
        llm_model: LLMModel = LLMModel.GPT_4O,
) -> str:
    """
    Generate a representation mapping using gds.udd, attach it to the dataset, and
    return the mapping as a JSON string.

    Side effects on gds:
      - gds.instance_hint_used set to the resolved hint (random from INSTANCE_HINTS if None)
      - gds.text_mapping set to the structured mapping (dict)
    """
    # Resolve theme hint
    if style_hint is None:
        print(f"[generate_text_mapping_json_from_dataset] No style hint provided; picking a random theme.")
        # Use SystemRandom to ensure the pick is never predictable via seeds
        cryptogen = random.SystemRandom()
        hint = cryptogen.choice(INSTANCE_HINTS)
    else:
        print(f"[generate_text_mapping_json_from_dataset] Using provided style hint: {style_hint!r}")
        hint = style_hint

    print(f"[generate_text_mapping_json_from_dataset] Resolved style hint: {hint!r}")

    # Build prompt & call LLM
    var_metadata = _gather_variable_metadata_from_udd(dataset.udd)
    property_metadata = _gather_property_metadata_from_dataset(dataset)
    system = _build_system_prompt()
    user = _build_user_prompt(
        var_metadata,
        hint,
        int(getattr(dataset.udd, "num_output_labels", 2)),
        property_metadata,
    )

    # llm = OpenAIChatWrapper(model="gpt-4o")
    llm = get_llm_wrapper(model=llm_model)
    raw = llm.make_call(input_text=f"{system}\n\n---\n\n{user}")

    # Parse & validate
    mapping = json.loads(_extract_json_block(raw))
    _validate_mapping(
        mapping,
        var_metadata,
        num_output_labels=int(getattr(dataset.udd, "num_output_labels", 2)),
        property_metadata=property_metadata,
    )

    # Return JSON string (pretty)
    return json.dumps(mapping, indent=indent, ensure_ascii=False)
