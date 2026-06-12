"""Prompt utilities for retrieval-aware conversational oracle workers."""
from __future__ import annotations

from data_management.dataset import DatasetSample, DatasetInstance
from typing import Iterable, Sequence, Tuple, List, Dict, Any
from data_management.utils import get_schema_lines_from_dataset
from configs import interactive_config

def _build_default_assistant_instruction(max_responses_per_question):
    instruction = (
        "You are a helpful LLM assistant. You are provided with reference samples "
        "showing variable assignments and binary labels. However, the questions you will receive, and the answers you will provide are all in natural text language."
        "Your task is to decode the variable assignments of the questions using the variable schema provided. \n\n"
        "You are then able to retrieve the text associated with a sample via a call to the `fetch_sample_text` tool with the "
        "sample's numeric ID to retrieve it. This allows you to respond to user queries using the textual descriptions of the requested samples.\n\n"
        
        "The task is therefore to identify which samples best match the user's request based on the variable assignments, retrieve their textual descriptions using the tool, and present them to the user in your response. \n\n"
        "1. Carefully analyze the user's question to determine the variable assignments they are interested in. \n"
        f"2. Identify the most relevant samples that best match these variable assignments. At most select {max_responses_per_question}\n"
        "3. Get the sample IDs of these relevant samples. \n"
        "4. Use the `fetch_sample_text` tool to retrieve the textual descriptions of these samples. \n"
        "5. Craft a response to the user that includes the textual descriptions of the relevant samples.\n\n"
        
        "If a users requests does not explicitly refer to any of the samples, you may choose to share the most relevant samples, but clarify your reasoning for sharing them. "
        "If no sample fits, say so explicitly.\n\n"
        
        "-------- Example --------\n\n"
        "Variable assignment:\n"
        "- V0 (Overall Temperature Trend): categories={ 0:cooling, 1:stable, 2:warming, 3:rapidly rising }} \n"
        "- V1 (Wind Conditions): categories={ 0:calm, 1:light breeze, 2:moderate winds, 3:strong winds, 4:gale-force winds }}\n"
        "- V2 (Sky Cover): categories={ 0:clear skies, 1:mostly cloudy }}\n"
        "- V3 (Precipitation Probability): categories={ 0:no chance, 1:slight chance, 2:moderate chance, 3:high chance }}\n"
        "- V4 (Air Quality Index): categories={ 0:good, 1:moderate, 2:unhealthy for sensitive groups, 3:unhealthy }}\n\n"
        
        "Question: I need a data point where there's a cooling trend and clear skies in the weather forecast, as we don't have such a case yet. What is the output for this scenario\n\n"
        
        "*Internal reasoning*: The user is looking for a sample with V0=0, V2=0\n"
        "I will look through the reference samples to find one that matches these criteria. \n\n"
        
        "-------- Important to Note --------\n\n"
        "The user will only request samples using natural language descriptions of the variable assignments, never the sample IDs or raw assignments directly. \n"
        "You must use the variable schema to decode these descriptions into variable assignments, identify the relevant samples, and then retrieve their textual descriptions using the tool.\n"
        "Never mention the variable assignments or IDs directly to the user; always provide the textual descriptions obtained via the tool.\n"
        f"At most select {max_responses_per_question} samples to share per user question.\n"
    )

    return instruction

RESPONSE_FORMAT_INSTRUCTION = (
    "Always format your response as JSON with the keys 'return_msg' and "
    "'shared_sample_ids'.\n"
    "- 'return_msg' should contain the text you would normally return to the user, including the samples that you want to show the user. The user will only see this message, so make sure that it is self-contained.\n"
    "- 'shared_sample_ids' should be a list of the numeric reference sample IDs you "
    "shared (use the IDs shown before each sample above; these match the dataset's "
    "global indices). Include an empty list if you did not share any samples."
)


def _format_assignment_only_samples(
    samples: Iterable[Tuple[int, DatasetSample]], label_names: dict[int, str]
) -> str:
    """Format samples with assignments and labels (no text) for prompts."""

    lines = []
    for idx, sample in samples:
        label = label_names.get(sample.label, f"Label {sample.label}")
        lines.append(f"{idx}. Label: {label} | Assignment: {sample.assignment}")
    return "\n".join(lines)


def build_retrieval_conversational_system_prompt(
    dataset_instance: DatasetInstance,
    config: interactive_config.InteractiveConfig,
    samples: Sequence[Tuple[int, DatasetSample]],
    *,
    label_names: dict[int, str] | None = None,
) -> str:
    """Build a system prompt that lists assignments/labels and mentions the text tool."""

    label_names = label_names or {}
    sample_block = _format_assignment_only_samples(samples, label_names)

    parts = []
    if sample_block:
        parts.append("Reference samples (assignments + labels only):")
        parts.append(sample_block)
    parts.append(
        _build_default_assistant_instruction(
            max_responses_per_question=config.max_retrievals_per_question
        ).strip()
    )
    parts.append(
        "You can call the `fetch_sample_text` tool with a sample ID to retrieve the "
        "textual description for that sample when needed."
    )
    parts.append(RESPONSE_FORMAT_INSTRUCTION.strip())


    schema = get_schema_lines_from_dataset(dataset_instance)
    if schema:
        intro_schema = ("Following is a variable mappings that's used to interpret assignments as text.\n"
                        "Use this to interpret what assignments that question refer to.\n"
                        )
        parts.append(intro_schema)
        parts.append(schema)


    return "\n\n".join(parts).strip()
