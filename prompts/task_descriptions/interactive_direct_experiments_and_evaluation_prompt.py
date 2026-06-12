from prompts.task_descriptions import interactive_direct_experiments_prompt
from data_management.dataset import DatasetInstance
from configs.interactive_config import InteractiveConfig
from interactive.experiment_costs import format_experiment_costs_for_prompt
from data_management.dataset import RepresentationLevel

from prompts.task_descriptions import static_evaluation_prompt

def build_interactive_prompt(
    dataset_instance: DatasetInstance,
    config: InteractiveConfig,
) -> str:
    """
    Build the interactive prompt for combined experimentation and theory evaluation.

    This leverages the existing direct experiment prompt and augments it with
    guidance on the ``evaluate_theory`` tool so that models know they can
    request full-dataset feedback on draft rules.
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
    direct_experiment_costs = format_experiment_costs_for_prompt([direct_experiment_name, "evaluate_theory"])

    available_tools = (
        f" --- Available tools: \n"
        f"- {direct_experiment_prompt}\n"
        f"- evaluate_theory(theory: str) takes a proposed theory in textual form and returns its overall accuracy on unseen samples.\n"
        f"In total you have an experiment budget of: {config.max_experiments}.\n"
        f"{direct_experiment_costs}\n\n"
    )

    tips = (
        "Tips for effective experimentation:\n"
        "- Start with a single experiment to make sure that the setup is working as expected.\n"
        "- Use the results of initial experiments to design more targeted follow-up experiments.\n"
        "- Use all available experiments wisely to cover diverse scenarios and edge cases.\n"
        "- Don't stop running experiments too early; sometimes patterns only emerge after several trials.\n"
        "- Keep track of all experiments and their outcomes to inform your theory development.\n\n"
    )

    return base_prompt + tool_loop + available_tools + tips
