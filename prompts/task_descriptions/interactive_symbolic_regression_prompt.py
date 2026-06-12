from __future__ import annotations

from configs.interactive_config import InteractiveConfig
from data_management.dataset import DatasetInstance
from prompts.task_descriptions import interactive_direct_experiments_and_evaluation_prompt, interactive_direct_experiments_prompt
from interactive.experiment_costs import format_experiment_costs_for_prompt
from data_management.dataset import RepresentationLevel
from prompts.task_descriptions import static_evaluation_prompt


def get_tool_str_prompt() -> str:
    costs = format_experiment_costs_for_prompt([
        "derive_numerical_regression",
        "derive_categorical_logic",
    ])

    tool_instructions = [
        "### Analytical Tools",
        "You can call the following functions to help you analyze the data you collect.",
        "Both tools are **stateless**; you must provide the full list of samples in each call.",
        "",
        "**1. derive_numerical_regression(samples)**",
        "   - **Input:** A list of dicts. Use string indices for variables and include a 'score' (float).",
        "   - **Output:** A Python function `def predict(x): return ...` and an R^2 score.",
        "",
        "**2. derive_categorical_logic(samples)**",
        "   - **Input:** A list of dicts. Use string indices for variables and include a 'label' (string).",
        "   - **Output:** A Python function `def predict(x):` with IF/ELSE logic and accuracy score.",
        "",
        "**Example Input Format for both tools:**",
        "```json",
        "[",
        "  {'0': 1.5, '1': 0.0, 'score': 0.85, 'label': 'Stable'},",
        "  {'0': 2.0, '1': 1.0, 'score': 0.12, 'label': 'Volatile'}",
        "]",
        "```",
        "",
        "Note: Variables in Python output (x0, x1, ...) map directly to these indices ('0', '1', ...)."
    ]

    if costs:
        tool_instructions.extend(["", costs])

    return "\n".join(tool_instructions)


def build_direct_only_symbolic_regression_prompt(dataset_instance: DatasetInstance, config: InteractiveConfig) -> str:
    """
    Build the starting prompt for the symbolic reasoning agent.
    Updated for stateless Python-outputting tools.
    """

    interactive_prompt = interactive_direct_experiments_prompt.build_interactive_prompt(
        dataset_instance=dataset_instance,
        config=config,
    )

    full_prompt = (
        interactive_prompt
        + "\n\n"
        + get_tool_str_prompt()
    )

    return full_prompt

def build_direct_and_eval_symbolic_regression_prompt(dataset_instance: DatasetInstance, config: InteractiveConfig) -> str:
    """
    Build the starting prompt for the symbolic reasoning agent.
    Updated for stateless Python-outputting tools.
    """

    base_prompt = static_evaluation_prompt.make_theory_prompt_code_without_variable_info(
        dataset=dataset_instance, variable_order=list(dataset_instance.udd.variables.keys()),
        max_examples=config.number_of_intro_samples,
    )

    tool_loop = (
        "\n\nYou will interact with the experimental setup in a loop. In each turn, you can choose to either:\n"
        "1) Propose one or several tool calls. \n"
        "2) Formulate and submit your theory based on the data you've gathered so far.\n"
        "If you do not provide any new experiments in a turn, the answer will be interpreted as submitting your final theory.\n\n"
    )

    # Check if current dataset is RAW or not
    representation_level = RepresentationLevel.parse(
        getattr(dataset_instance, "representation_level", RepresentationLevel.UNKNOWN)
    )
    if representation_level != RepresentationLevel.RAW:
        direct_experiment_prompt = "run_textual_experiment(query: str) takes a textual query describing the experiment to run and returns the corresponding output."
        direct_experiment_name = "run_textual_experiment"
    else:
        direct_experiment_prompt = "run_experiment(assignment: dict[str, int]) takes a defined experiment and returns the corresponding output."
        direct_experiment_name = "run_experiment"

    # Add explicit experiment costs for evaluation tooling.
    direct_experiment_costs = format_experiment_costs_for_prompt([direct_experiment_name, "evaluate_theory", "derive_categorical_logic"])

    available_tools = (
        f" --- Available tools: \n"
        f"- {direct_experiment_prompt}\n"
        f"- evaluate_theory(theory: str) takes a proposed theory in textual form and returns its overall accuracy on unseen samples.\n"
        f"- derive_categorical_logic(samples:  List[Dict[str, Any]]) takes a list of direct variable assignments samples with categorical, see details below.\n"
        f"In total you have an experiment budget of: {config.max_experiments}.\n"
        f"{direct_experiment_costs}\n\n"
    )

    example_usage = """ 
    Example call for derive_categorical_logic:
    # IMPORTANT: Variable names must be passed as their string indices (e.g., "0", "1") 
    # and the target class must be provided with the key "label".
    derive_categorical_logic(samples=[
        {"0": 10, "1": 5, "label": "class_a"},
        {"0": 2, "1": 12, "label": "class_b"}
    ])\n\n"""

    tips = (
        "Tips for effective experimentation:\n"
        "- Start with a single experiment to make sure that the setup is working as expected.\n"
        "- Use the results of initial experiments to design more targeted follow-up experiments.\n"
        "- Use all available experiments wisely to cover diverse scenarios and edge cases.\n"
        "- Don't stop running experiments too early; sometimes patterns only emerge after several trials.\n"
        "- Keep track of all experiments and their outcomes to inform your theory development.\n\n"
    )

    return base_prompt + tool_loop + available_tools + example_usage + tips
