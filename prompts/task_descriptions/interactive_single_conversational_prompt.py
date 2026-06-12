from data_management.dataset import DatasetInstance
from configs.interactive_config import InteractiveConfig
from prompts.task_descriptions import static_evaluation_prompt


def build_interactive_prompt(dataset_instance: DatasetInstance, config: InteractiveConfig) -> str:

    initial_static_prompt = static_evaluation_prompt.make_theory_prompt_code_without_variable_info(
        dataset=dataset_instance,
        variable_order=list(dataset_instance.udd.variables.keys()),
        max_examples=config.number_of_intro_samples,
    )

    oracle_setup = (
        "\n\nIn this interactive setting, you can consult an expert oracle to retrieve historical experimental results for input assignments of your choosing.\n"
        "The oracle may not always know the answer to every query, so choose your questions carefully.\n"
        f"You can ask up to {config.max_experiments} questions over a maximum of {config.max_turns} turns.\n\n"
    )

    tool_loop = (
        "You will interact with the expert oracle in a loop. In each turn, you can choose to either:\n"
        "1) Ask a question by using the provided tool function to query a specific assignment.\n"
        "2) Formulate and submit your theory based on the data you've gathered so far.\n\n"
        "If you do not provide any new tool calls in a turn, the answer will be interpreted as submitting your final theory.\n\n"
    )

    tips = (
        "Tips for effective experimentation:\n"
        "- Start with a single question to make sure that the setup is working as expected.\n"
        "- Use the results of initial questions to design more targeted follow-up experiments.\n"
        "- Use all available experiments wisely to cover diverse scenarios and edge cases.\n"
        "- Keep track of all experiments and their outcomes to inform your theory development.\n\n"
    )

    full_prompt = initial_static_prompt + oracle_setup + tool_loop + tips

    return full_prompt
