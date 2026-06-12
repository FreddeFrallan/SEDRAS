from __future__ import annotations

from typing import List, Optional

CLASSIFIER_SYSTEM_PROMPT = (
    "You are an expert model builder. You may reason in natural language, "
    "analyze patterns, and outline your plan. "
    "At the END of your response, you MUST include a SINGLE fenced Python code block "
    "that defines exactly:\n"
    "def classify(x: list[float]) -> dict:\n"
    "    # x is a list of values (categorical variables use integer categories; numerical variables use floats)\n"
    "    # Must return a dictionary containing an integer 'label' entry and any other optional properties.\n"
    "    # No prints or external imports.\n"
    "    # Deterministic and side-effect free.\n"
    "\n"
    "Constraints for the FINAL code block:\n"
    "- No imports or classes.\n"
    "- Use only pure Python and builtins (len, sum, min, max, abs, int, float, range, all, any).\n"
    "- The code block must be the LAST thing in your reply, fenced by triple backticks.\n"
    "- The dict returned must contain the key 'label', which maps to an integer label.\n"
)


def render_classifier_user_prompt(
    instruction: str,
    variable_order: Optional[List[str]] = None,
) -> str:
    order_text = ""
    if variable_order:
        order_text = (
            "Use this fixed variable order for x (x[i] is the category of this variable):\n"
            f"{variable_order}\n\n"
        )
    return (
        f"{order_text}"
        "First, reason about the task as long as needed. "
        "Then END your answer with a single fenced Python code block implementing:\n"
        "def classify(x: list[float]) -> dict\n"
        "- Always return a dictionary containing an integer entry 'label' plus any optional properties requested.\n"
        "- x[i] is an integer category for categorical variables and a float for numerical variables.\n"
        "- No prints, no imports, deterministic.\n\n"
        "Task-specific guidance:\n"
        f"{instruction.strip()}\n"
    )
