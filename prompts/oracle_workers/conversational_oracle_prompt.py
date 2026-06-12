"""Prompt utilities for conversational oracle workers."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from data_management.dataset import DatasetInstance, DatasetSample
from data_management.utils import get_output_label_names
from prompts.task_descriptions.utils import (
    _get_property_label_names,
    _render_from_assignment,
    _render_label_and_properties,
    _render_output_labels,
    _render_raw_samples,
)

DEFAULT_ASSISTANT_INSTRUCTION = (
    "You are a helpful LLM assistant. You have been provided with reference "
    "samples above. When the user asks for help, return the most relevant "
    "samples. At most, you may return 3 samples at a time. If no sample "
    "fits, say so explicitly."
)

# Response formatting requirement to enable downstream tracking of shared samples
RESPONSE_FORMAT_INSTRUCTION = (
    "Always format your response as JSON with the keys 'return_msg' and "
    "'shared_sample_ids'.\n"
    "- 'return_msg' should contain the text you would normally return to the user, including the samples that you want to show the user. The user will only see this message, so make sure that it is self-contained.\n"
    "- 'shared_sample_ids' should be a list of the numeric reference sample IDs you "
    "shared (use the IDs shown before each sample above; these match the dataset's "
    "global indices). Include an empty list if you did not share any samples."
)


def _format_samples_with_utils(
    samples: Iterable[Tuple[int, DatasetSample]],
    dataset_instance: DatasetInstance,
) -> str:
    """Format samples using the shared prompt utilities for consistency."""

    if not samples:
        return ""

    tm = dataset_instance.text_mapping or {}
    label_names = get_output_label_names(dataset_instance)
    tm_labels = (dataset_instance.text_mapping or {}).get("output_labels", {}) if dataset_instance.text_mapping else {}
    use_textual_only = isinstance(tm_labels, dict) and len(tm_labels) > 0
    property_label_names = _get_property_label_names(dataset_instance)

    lines = []
    for idx, sample in samples:
        text = (sample.instance_text or "").strip() if sample.instance_text else ""
        if not text:
            if not tm:
                text = _render_raw_samples(sample)
            else:
                text = _render_from_assignment(tm, sample.assignment) or ""
        if text:
            text = text.replace("\n", " ")

        label_block = _render_label_and_properties(
            label_idx=int(sample.label),
            label_names=label_names,
            property_labels=getattr(sample, "property_labels", {}) or {},
            property_label_names=property_label_names,
            use_textual_only=use_textual_only,
        )
        lines.append(f"{idx}. Labels: {label_block} | {text if text else '(no text)'}")

    return "\n".join(lines)


def build_sample_augmented_system_prompt(
    samples: Sequence[Tuple[int, DatasetSample]],
    dataset_instance: DatasetInstance,
    *,
    assistant_instruction: str = DEFAULT_ASSISTANT_INSTRUCTION,
) -> str:
    """
    Build a conversational oracle system prompt that embeds the provided samples.

    Args:
        samples: Ordered collection of (dataset_index, sample) pairs to prepend to
            the prompt.
        dataset_instance: Dataset providing metadata used for consistent formatting.
        assistant_instruction: Custom instruction explaining how the oracle should
            respond to user questions using the supplied samples.

    Returns:
        Full system prompt text containing the samples followed by the assistant
        instruction.
    """
    label_line = _render_output_labels(dataset_instance)
    sample_block = _format_samples_with_utils(samples, dataset_instance)

    parts = [label_line]
    if sample_block:
        parts.append("Reference samples:")
        parts.append(sample_block)
    parts.append(assistant_instruction.strip())
    parts.append(RESPONSE_FORMAT_INSTRUCTION.strip())

    return "\n\n".join(parts).strip()
