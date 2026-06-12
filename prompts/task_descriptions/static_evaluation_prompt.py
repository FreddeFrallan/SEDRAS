from __future__ import annotations

import random

from typing import Any, Dict, List, Optional
from data_management.dataset import DatasetInstance, RepresentationLevel
from prompts.task_descriptions.utils import _get_property_hint, _pack_examples_block, _render_output_labels
import numpy as np




def _generate_theory_requirements(*, dataset: DatasetInstance, label_line: str, max_property_hints=4) -> str:
    """Build the THEORY requirements block.

    If the dataset contains additional output properties, we instruct the LLM
    to provide a theory for both the main label and those properties, and
    update the implementation constraints accordingly.
    """

    property_hints = _get_property_hint(dataset)
    has_properties = bool(property_hints)

    if dataset.representation_level == RepresentationLevel.RAW:
        property_list = ", ".join(sorted(property_hints.keys()))
    else:
        formatted_properties: List[str] = []
        all_hints = []
        for pname, values in sorted(property_hints.items()):
            all_hints.extend(list(values))

        np.random.shuffle(all_hints)
        selected_hints = all_hints[:max_property_hints]
        property_list = ", ".join(selected_hints)


    parts: List[str] = [
        "THEORY requirements:",
        f"- Provide a clear description of the decision logic that maps integer variables to the correct label among: {label_line}.",
    ]

    if has_properties:
        parts.append(
            f"- Also, it's possible that certain outputs also have corresponding properties, such as ({property_list}).\n"
            f"  You theory should also to determine the values of these additional properties as well, and when they apply."
        )

    parts.extend(
        [
            "- The theory must be unambiguous and fully deterministic.",
            "- Make sure to cover all possible input assignments in your theory. There is no noise in the data, so a good theory should be able to explain all observed data.",
            "- You are allowed to provide equations, rules, pseudocode, or python-like code snippets to explain your theory.",
            "- One must be able to follow your theory to a concrete decision for any possible input assignment.",
            "",
            "Implementation constraints (for your awareness):",
        ]
    )

    if has_properties:
        parts.append(
            "- At a later stage, your produced theory will be compiled into a function with signature: def classify(x: list[float]) -> dict."
        )
        parts.append(
            "- The returned dict will include the predicted label and values for potential additional properties.\n"
            "- If you don't want to predict a specific property for some input, you can simply omit it. This means the default behavior is to not predict any property unless explicitly specified in your theory."
        )
    else:
        parts.append(
            "- At a later stage, your produced theory will be compiled into a function with signature: def classify(x: list[float]) -> int."
        )

    parts.append(
        "- It is therefore important that you explicitly name each variable that you are referring to in your theory."
    )

    return "\n".join(parts)


# -----------------------
# Public prompt builders (one per setting)
# -----------------------

def make_theory_prompt_code_without_variable_info(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
    max_examples=None
) -> str:
    """
    Non-interactive prompt: same task and theory requirements as the interactive
    setup, but without any experimental / tool-calling elements.
    Uses only the provided labeled examples.
    """
    tm = dataset.text_mapping or {}
    instance_type = tm.get("instance_type")
    theme_summary = tm.get("theme_summary")
    instance_hint_used = dataset.instance_hint_used
    label_line = _render_output_labels(dataset)

    parts: List[str] = []

    # High-level task description (aligned with build_interactive_prompt.start_prompt,
    # but with no mention of tools or experiments).
    parts.append(
        "You are a scientific researcher tasked with formulating a theory to explain a categorical outcome "
        "based on several input variables. There may also be additional output properties associated with certain outcomes, these should also be predicted.\n "
        "We aim to predict the outcome by understanding the "
        "relationships between inputs and the outcome.\n"
        "The goal is to produce a theory that not only accurately predicts the outcome, but is also "
        "interpretable and grounded in the provided data.\n"
        "Additionally, the theory will be evaluated on unseen data to assess its generalization capabilities.\n"
        "\n"
    )

    parts.append(label_line)

    if instance_type:
        parts.append(f"Instance type: {instance_type}")
    if theme_summary:
        parts.append(f"Theme summary: {theme_summary}")
    if instance_hint_used:
        parts.append(f"Instance hint used: {instance_hint_used}")

    parts.append("")
    parts.append(_pack_examples_block(dataset, max_examples=max_examples))
    parts.append("")

    # THEORY requirements: aligned with build_interactive_prompt.theory_requirements,
    # plus explicit implementation constraints (classify signature, variable naming).
    parts.append(_generate_theory_requirements(dataset=dataset, label_line=label_line))

    return "\n".join(parts)


def build_static_prompt(
    *,
    dataset: DatasetInstance,
    variable_order: List[str],
    max_examples: Optional[int] = None,
) -> str:
    """Build the starting prompt for static (non-interactive) evaluation runs.

    Static evaluation now always uses the "code_no_variable_info" guidance,
    which instructs the model to write executable code without revealing the
    original variable names.
    """
    return make_theory_prompt_code_without_variable_info(
        dataset=dataset, variable_order=variable_order, max_examples=max_examples
    )
