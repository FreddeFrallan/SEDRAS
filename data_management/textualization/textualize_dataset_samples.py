# textualize_dataset_samples.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from data_management.dataset import DatasetInstance, DatasetSample, AbstractDataset  # your updated file
from data_management.dataset import RepresentationLevel
from data_management.underlying_data.underlying_data import VariableType
from data_management.underlying_data.output_properties import OutputPropertyType
from inference.model_wrappers.llm_wrapper import LLMModel, get_llm_wrapper, LLMWrapper          # your wrapper
import tqdm
import random
import time


def _format_numeric_value(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    if not text:
        return "0"
    return text


def _inject_numeric_value(
    variable_info: Dict[str, Any],
    category_index: int,
    phrase: str,
) -> str:
    value_range = variable_info.get("value_range")
    categories = variable_info.get("categories") or {}
    if not value_range or not isinstance(value_range, dict):
        return phrase

    try:
        min_v = float(value_range.get("min"))
        max_v = float(value_range.get("max"))
    except Exception:
        return phrase

    if max_v <= min_v:
        return phrase

    total_spans = max(len(categories), 1)
    width = (max_v - min_v) / float(total_spans)
    span_start = min_v + width * category_index
    span_end = span_start + width
    value = random.uniform(span_start, span_end)
    formatted = _format_numeric_value(value)
    unit = value_range.get("unit")
    if unit:
        unit = str(unit)
        needs_space = not unit.startswith(("°", "%", "/", "-"))
        unit_suffix = f" {unit}" if needs_space else unit
    else:
        unit_suffix = ""
    numeric_text = (formatted + unit_suffix).strip()

    if phrase:
        if numeric_text in phrase:
            return phrase
        return f"{phrase} ({numeric_text})"

    role = variable_info.get("role")
    if role:
        return f"{role}: {numeric_text}"
    return numeric_text


def _resolve_phrases_for_assignment(mapping: Dict[str, Any], assignment: Dict[str, int]) -> Dict[str, str]:
    """
    Given a mapping and an assignment (e.g., {'V0': 2, 'V1': 0}), return the textual
    phrase for each variable from mapping['variables'][var]['categories'][str(cat)].
    """
    phrases: Dict[str, str] = {}
    variables = mapping["variables"]
    for var, cat in assignment.items():
        cat_idx = int(cat)
        cat_str = str(cat_idx)
        try:
            base_phrase = variables[var]["categories"][cat_str]
        except Exception as e:
            raise KeyError(f"Missing phrase for variable '{var}' category '{cat_str}': {e}")
        phrases[var] = _inject_numeric_value(variables[var], cat_idx, base_phrase)
    return phrases


def _infer_variable_types(text_mapping: Dict[str, Any]) -> Dict[str, VariableType]:
    """Infer per-variable types from the representation mapping.

    Defaults to ``VariableType.CATEGORICAL`` when the type cannot be determined.
    """

    var_types: Dict[str, VariableType] = {}
    tm_vars = text_mapping.get("variables", {}) if isinstance(text_mapping, dict) else {}
    if not isinstance(tm_vars, dict):
        return var_types

    for name, meta in tm_vars.items():
        vtype: Optional[VariableType] = None
        if isinstance(meta, dict):
            raw_type = meta.get("variable_type")
            if raw_type:
                try:
                    vtype = VariableType(raw_type)
                except Exception:
                    vtype = None

            if vtype is None:
                value_range = meta.get("value_range")
                if isinstance(value_range, dict):
                    try:
                        min_v = float(value_range.get("min"))
                        max_v = float(value_range.get("max"))
                        if max_v > min_v:
                            vtype = VariableType.NUMERICAL
                    except Exception:
                        pass

        if vtype is None:
            vtype = VariableType.CATEGORICAL

        var_types[name] = vtype

    return var_types


def _format_numeric_with_unit(
    value: float,
    *,
    value_range: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> str:
    """Format a numeric output with optional unit and description."""
    formatted = _format_numeric_value(value)
    unit = None
    if isinstance(value_range, dict):
        unit = value_range.get("unit")
    if unit:
        unit = str(unit)
        needs_space = not unit.startswith(("°", "%", "/", "-"))
        formatted = f"{formatted}{' ' if needs_space else ''}{unit}"
    if description:
        formatted = f"{formatted} ({description})"
    return formatted


def _resolve_instance_label_text(
    mapping: Dict[str, Any],
    label: int,
    property_labels: Optional[Dict[str, Optional[int]]],
    property_scores: Optional[Dict[str, Optional[float]]],
) -> Dict[str, Any]:
    """Build a textual representation of the main and property labels for one sample."""
    tm_labels = mapping.get("output_labels", {}) if isinstance(mapping, dict) else {}
    label_text = None
    if isinstance(tm_labels, dict):
        label_text = tm_labels.get(str(label))
    if label_text is None:
        label_text = str(label)

    properties_text: Dict[str, str] = {}
    tm_props = mapping.get("output_properties", {}) if isinstance(mapping, dict) else {}
    if isinstance(tm_props, dict):
        for prop_name, prop_info in tm_props.items():
            if not isinstance(prop_info, dict):
                continue

            prop_type = str(prop_info.get("type") or OutputPropertyType.CATEGORICAL.value).lower()

            if prop_type == OutputPropertyType.NUMERICAL.value:
                raw_val = None
                if property_scores:
                    raw_val = property_scores.get(prop_name)
                if raw_val is None and property_labels:
                    raw_val = property_labels.get(prop_name)
                if raw_val is None:
                    continue
                try:
                    num_val = float(raw_val)
                except Exception:
                    continue
                properties_text[prop_name] = _format_numeric_with_unit(
                    num_val,
                    value_range=prop_info.get("value_range") if isinstance(prop_info.get("value_range"), dict) else None,
                    description=prop_info.get("description"),
                )
            else:
                if not property_labels:
                    continue
                raw_val = property_labels.get(prop_name)
                if raw_val is None:
                    continue
                try:
                    idx = int(raw_val)
                except Exception:
                    continue
                prop_text = prop_info.get(str(idx))
                if prop_text is None:
                    prop_text = str(idx)
                properties_text[prop_name] = prop_text

    return {
        "label": label_text,
        "properties": properties_text,
    }


def _apply_template_locally(template: str, phrases: Dict[str, str]) -> str:
    out = template
    for var, phrase in phrases.items():
        out = out.replace(f"{{var.{var}}}", phrase)
    return out.strip()


def _missing_keywords(text: str, keywords: List[str]) -> List[str]:
    """Return any keywords that are not found in the provided text (case-insensitive)."""
    lowered = text.lower()
    missing = []
    for kw in keywords:
        if not kw:
            continue
        if kw.lower() not in lowered:
            missing.append(kw)
    return missing


def _categorical_keywords(
    phrases: Dict[str, str],
    variable_types: Optional[Dict[str, VariableType]],
) -> List[str]:
    """Return phrases that must appear in the output (categorical variables only)."""

    if not variable_types:
        return [phrase for phrase in phrases.values() if isinstance(phrase, str)]

    required: List[str] = []
    for var, phrase in phrases.items():
        if not isinstance(phrase, str):
            continue
        vtype = variable_types.get(var, VariableType.CATEGORICAL)
        if vtype is VariableType.CATEGORICAL:
            required.append(phrase)
    return required


# ---------------------------
# LLM prompt builder
# ---------------------------

def _create_template_prompt() -> str:
    return (
        "You convert structured assignments into one concise sentence using a given template.\n"
        "Rules:\n"
        "1) Replace each {var.NAME} placeholder exactly once with the provided phrase for that variable.\n"
        "2) When a variable is included it must be included using the EXACT same phrase as provided\n"
        "3) Do not add extra facts, numbers, or variables. No creative additions.\n"
        "4) Output ONLY the final sentence (no quotes, no JSON, no commentary)."
    )

def _create_free_system_prompt() -> str:
    return (
        "You convert structured assignments into a 1-2 sentence free flowing text.\n"
        "Rules:\n"
        "1) Each variable must be included at least once\n"
        "2) When a variable is included it must be included using the EXACT same phrase as provided\n"
        "3) You should be creative in the text, but do not add extra facts that can be interpreted as important.\n"
        "4) Output ONLY the final text (no quotes, no JSON, no commentary)."
    )

def _create_free_system_prompt_long() -> str:
    return (
        "You convert structured assignments into a 5-8 sentence free flowing text.\n"
        "Rules:\n"
        "1) Each variable must be included at least once, but can occur multiple times\n"
        "2) When a variable is included it must be included using the EXACT same phrase as provided\n"
        "3) You should be creative in the text, feel free to add extra context, but not facts that can be interpreted as important.\n"
        "4) Output ONLY the final text (no quotes, no JSON, no commentary)."
    )

def _build_user_prompt(instance_type: str, template: str, phrases: Dict[str, str]) -> str:
    # Provide resolved phrases explicitly to avoid any ambiguity
    lines = []
    lines.append(f"Theme: {instance_type}")
    if(template):
        lines.append("Template:")
        lines.append(template)
    lines.append("")
    lines.append("Resolved phrases for this assignment (use exactly as given):")
    for var in sorted(phrases.keys()):
        lines.append(f"- {var}: {phrases[var]}")
    lines.append("")
    lines.append("Now produce the final sentence by substituting the placeholders exactly once.")
    return "\n".join(lines)


# ---------------------------
# Core representation logic
# ---------------------------

def _textualize_one_sample_via_llm(
    llm: LLMWrapper,
    instance_type: str,
    template: str,
    phrases: Dict[str, str],
    variable_types: Optional[Dict[str, VariableType]],
    *,
    max_retries: int = 6,
    base_backoff: float = 1.5,     # seconds
    max_backoff: float = 30.0,     # cap for exponential backoff
    inter_call_delay: float = 0.15, # small delay between sequential calls to reduce burstiness
    max_keyword_retries: int = 3,   # max attempts to ensure all required keywords are present
    target_levels: Optional[Set[RepresentationLevel]] = None,
) -> Dict[RepresentationLevel, str]:
    """
    Textualize one sample across prompt styles with retry & backoff on 429/5xx.
    Fails fast on 'insufficient_quota' to let the outer loop stop gracefully.
    """

    def _safe_llm_call(prompt: str, user: str) -> str:
        delay = base_backoff
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                raw = llm.make_call(input_text=f"{prompt}\n\n---\n\n{user}")
                return (raw or "").strip()

            except Exception as e:
                msg = str(e)
                name = e.__class__.__name__

                # Fast-fail on true quota exhaustion (retrying won't help)
                if ("insufficient_quota" in msg) or ("You exceeded your current quota" in msg):
                    # Re-raise as-is so upstream can handle / stop the job
                    raise

                # Retry on rate limiting or transient errors
                retryable = (
                    "RateLimit" in name or
                    "429" in msg or
                    "rate" in msg.lower() or
                    "APIConnectionError" in name or
                    "Timeout" in name or
                    "ServiceUnavailable" in name or
                    "InternalServerError" in name or
                    "BadGateway" in name or
                    "502" in msg or
                    "503" in msg or
                    "504" in msg
                )

                if retryable and attempt < max_retries:
                    # Exponential backoff with jitter
                    jitter = random.uniform(0, delay * 0.25)
                    time.sleep(delay + jitter)
                    delay = min(delay * 2.0, max_backoff)
                    last_err = e
                    continue

                # Non-retryable or exhausted retries -> surface the error
                raise e if not last_err else RuntimeError(
                    f"LLM call failed after {attempt} attempts"
                ) from e

    prompt_order = {
        RepresentationLevel.TEMPLATE_BASED: _create_template_prompt,
        RepresentationLevel.FREE_TEXT: _create_free_system_prompt,
        RepresentationLevel.FREE_TEXT_LONG: _create_free_system_prompt_long,
    }
    required_keywords = _categorical_keywords(phrases, variable_types)

    results: Dict[RepresentationLevel, str] = {}
    for level, prompt_func in prompt_order.items():
        if target_levels and level not in target_levels:
            continue
        prompt = prompt_func()
        temp_template = template if level == RepresentationLevel.TEMPLATE_BASED else None
        user = _build_user_prompt(instance_type, temp_template, phrases)

        text = ""
        missing: List[str] = []
        for attempt in range(1, max_keyword_retries + 1):
            text = _safe_llm_call(prompt, user)
            missing = _missing_keywords(text, required_keywords)
            if not missing:
                break
            print(
                f"[textualize_dataset_samples] Missing keywords for level '{level.value}' "
                f"(attempt {attempt}/{max_keyword_retries}). Retrying. Missing: {missing}"
            )
            # Small spacing between sequential calls helps avoid bursty 429s
            if inter_call_delay > 0 and attempt < max_keyword_retries:
                time.sleep(inter_call_delay)
        if missing:
            print(
                f"[textualize_dataset_samples] Proceeding with latest generation despite missing keywords "
                f"for level '{level.value}'. Missing: {missing}"
            )
        results[level] = text

        # Small spacing between sequential calls helps avoid bursty 429s
        if inter_call_delay > 0:
            time.sleep(inter_call_delay)

    return results


def textualize_dataset_samples(
    dataset: AbstractDataset,
    text_mapping: Dict[str, Any],
    *,
    llm_model: LLMModel = LLMModel.GPT_4O,
    levels: Optional[List[RepresentationLevel]] = None,
    limit: Optional[int] = None,          # for quick tests: only process first N samples
    quiet: bool = False,
    num_workers: int = 40,
) -> Dict[RepresentationLevel, DatasetInstance]:

    def _resolve_generation_level(level: RepresentationLevel) -> RepresentationLevel:
        if level in (RepresentationLevel.FILES, RepresentationLevel.FREE_TEXT_LONG_FILES):
            return RepresentationLevel.FREE_TEXT_LONG
        return level

    # Default: only generate long free-text descriptions
    if levels is None:
        levels = [RepresentationLevel.FREE_TEXT_LONG]
    levels = [RepresentationLevel.parse(lv) for lv in levels]

    instance_type = text_mapping.get("instance_type")
    template = text_mapping.get("template", "")
    variable_types = _infer_variable_types(text_mapping)

    llm = get_llm_wrapper(model=llm_model)

    # Only create buffers for the requested levels
    level_generation_map: Dict[RepresentationLevel, RepresentationLevel] = {
        level: _resolve_generation_level(level) for level in levels
    }
    new_samples: Dict[RepresentationLevel, List[DatasetSample]] = {
        level: [] for level in levels
    }
    generation_targets: Set[RepresentationLevel] = set(level_generation_map.values())

    num = len(dataset.raw_samples) if limit is None else min(limit, len(dataset.raw_samples))
    workers = max(1, int(num_workers))

    def _process_sample(idx: int):
        original_s: DatasetSample = dataset.raw_samples[idx]
        phrases = _resolve_phrases_for_assignment(text_mapping, original_s.assignment)

        level_2_text = _textualize_one_sample_via_llm(
            llm,
            instance_type,
            template,
            phrases,
            variable_types,
            target_levels=generation_targets,
        )

        return idx, original_s, level_2_text

    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterable = executor.map(_process_sample, range(num))
        for _, original_s, level_2_text in tqdm.tqdm(
            iterable,
            total=num,
            disable=quiet,
            desc="Textualizing samples",
        ):
            # Keep only the levels we explicitly asked for
            for target_level, samples_bucket in new_samples.items():
                base_level = level_generation_map[target_level]
                text = level_2_text.get(base_level)
                if text is None:
                    continue
                label_texts = _resolve_instance_label_text(
                    text_mapping,
                    original_s.label,
                    getattr(original_s, "property_labels", None),
                    getattr(original_s, "property_scores", None),
                )
                s_level = DatasetSample(
                    assignment=original_s.assignment,
                    score=original_s.score,
                    label=original_s.label,
                    instance_text=text,
                    instance_label_text=label_texts,
                    instance_type=instance_type,
                    numerical_values=original_s.numerical_values,
                    property_labels=original_s.property_labels,
                    property_scores=original_s.property_scores,
                )
                samples_bucket.append(s_level)

    # Create a name for the instance, based on instance_type
    name = instance_type.replace(" ", "_").lower()
    metadata = {
        "note": "Samples textualized via LLM using provided mapping.",
        "model": llm_model.name,
        "instance_type": instance_type,
    }

    return_results: Dict[RepresentationLevel, DatasetInstance] = {}
    for level, samples in new_samples.items():
        instance_name = name
        new_instance = dataset.add_instance(
            instance_name,
            samples,
            level=level,
            metadata=metadata,
            text_mapping=text_mapping,
        )
        return_results[level] = new_instance

    return return_results
