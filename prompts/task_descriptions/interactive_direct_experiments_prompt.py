from prompts.task_descriptions.utils import _pack_examples_block, _render_output_labels
from data_management.dataset import DatasetInstance, UnderlyingDataDistribution
from prompts.task_descriptions import static_evaluation_prompt
from configs.interactive_config import InteractiveConfig
from interactive.experiment_costs import format_experiment_costs_for_prompt
from typing import Dict
import numpy as np

def _create_examples_prompt(dataset_instance: DatasetInstance, num_samples: int) -> str:
    examples_block = _pack_examples_block(
        dataset_instance, max_examples=num_samples
    )

    if not examples_block:
        return ""

    return (
        examples_block
        + "\n\nIt is very important that your theory also explains these examples correctly.\n\n"
    )


def get_variable_mapping(textual_mapping, udd: UnderlyingDataDistribution) -> str:
    """
    Return a human-readable mapping of variable names to number of categories,
    and include a random example assignment. Useful for prompting the model to
    choose a concrete assignment for experiments.
    """
    response_lines = [
        "To run the experiment, please use the following variable mapping where the input object is a dict[str, int]:"
    ]
    example_mapping: Dict[str, int] = {}

    for i, (var_name, var_info) in enumerate(udd.variables.items()):
        text_variable = textual_mapping['variables'][var_name]
        # var_info.categories should be iterable (e.g., [0,1,2])
        response_lines.append(f"- {var_name}: num categories = {var_info.categories}: {text_variable}")
        # Choose an example category deterministically random-ish
        choice = np.random.choice(var_info.categories)
        # Cast to plain int to avoid numpy scalar surprises
        try:
            choice = int(choice)
        except Exception:
            pass
        example_mapping[var_name] = choice

    response_lines.append(f"\nExample assignment: {example_mapping}\n")
    return "\n".join(response_lines)

def build_interactive_prompt(
    dataset_instance: DatasetInstance,
    config: InteractiveConfig,
) -> str:

    initial_static_prompt = static_evaluation_prompt.make_theory_prompt_code_without_variable_info(
        dataset=dataset_instance, variable_order=list(dataset_instance.udd.variables.keys()),
        max_examples=config.number_of_intro_samples,
    )

    tool_costs = format_experiment_costs_for_prompt(["run_experiment"])

    tool_loop = (
        "\n\nYou will interact with the experimental setup in a loop. In each turn, you can choose to either:\n"
        "1) Propose an experiment by specifying values for the input variables to receive the corresponding output.\n"
        "2) Formulate and submit your theory based on the data you've gathered so far.\n"
        "If you do not provide any new experiments in a turn, the answer will be interpreted as submitting your final theory.\n"
        f"In total you have {config.max_experiments} experiments available to use.\n"
        + ("" if not tool_costs else tool_costs + "\n\n")
    )

    tips = (
        "Tips for effective experimentation:\n"
        "- Start with a single experiment to make sure that the setup is working as expected.\n"
        "- Use the results of initial experiments to design more targeted follow-up experiments.\n"
        "- Use all available experiments wisely to cover diverse scenarios and edge cases.\n"
        "- Don't stop running experiments too early; sometimes patterns only emerge after several trials.\n"
        "- Keep track of all experiments and their outcomes to inform your theory development.\n\n"
    )

    if(config.number_of_intro_samples > 0):
        examples_prompt = _create_examples_prompt(dataset_instance, config.number_of_intro_samples)
    else:
        examples_prompt = ""

    full_prompt = (
        initial_static_prompt +
        tool_loop +
        tips +
        examples_prompt
    )

    return full_prompt
