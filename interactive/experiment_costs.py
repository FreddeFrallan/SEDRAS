"""Centralized configuration for experiment costs and related utilities."""

from __future__ import annotations

from typing import Dict, Iterable

EXPERIMENT_COSTS: Dict[str, int] = {
    # Direct experimentation
    "run_experiment": 1,
    "run_textual_experiment": 1,
    # Theory evaluation
    "evaluate_theory": 5,
    # Symbolic regression helpers
    "derive_numerical_regression": 1,
    "derive_categorical_logic": 1,
}

TOOL_DISPLAY_NAMES: Dict[str, str] = {
    "run_experiment": "Direct experiment",
    "evaluate_theory": "Evaluate theory",
    "derive_numerical_regression": "Symbolic regression (numerical)",
    "derive_categorical_logic": "Symbolic regression (categorical)",
}


def get_experiment_cost(function_name: str) -> int:
    """Return the experiment cost for a given tool/function name."""

    return int(EXPERIMENT_COSTS.get(function_name, 1))


def format_experiment_costs_for_prompt(tool_names: Iterable[str]) -> str:
    """Render a human-readable list of experiment costs for prompt text."""

    lines = []
    for tool_name in tool_names:
        if tool_name not in EXPERIMENT_COSTS:
            continue

        display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        lines.append(f"- {display_name}: {EXPERIMENT_COSTS[tool_name]}")

    if not lines:
        return ""

    return "Experiment costs per tool:\n" + "\n".join(lines)
