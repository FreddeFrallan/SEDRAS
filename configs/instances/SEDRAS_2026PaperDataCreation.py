from pathlib import Path

from configs.paper_data_creation import PAPER_DATA_CREATION_CONFIG
from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from inference.model_wrappers.llm_wrapper import LLMModel

# Instance of the paper data creation config tailored for SEDRAS_2026 runs.
SEDRAS_2026_PAPER_DATA_CREATION_CONFIG = {
    **PAPER_DATA_CREATION_CONFIG,
    # Root folder for saving generated datasets
    # "save_folder": Path("udd_paper_datasets_balanced_udd"),
    "save_folder": Path("debug_datasets"),

    # Number of worker processes to use during dataset creation
    "number_of_parallel_workers": 1,
    "max_genetic_search_configurations": 4000,
    "genetic_search_max_iterations": 2500,

    "udd_complexities": [1, 2, 3, 4, 5],
    # "udd_complexities": [2],

    # Dataset sizes to generate in the sweep
    "num_samples": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # "num_samples": [10, 20, 40, 60, 80, 100],
    # "num_samples": [10, 20],

    # Representation levels to create for each dataset
    "representation_levels": [
        # None,
        RepresentationLevel.FREE_TEXT,
        # RepresentationLevel.FREE_TEXT_LONG,
        # RepresentationLevel.FILES
    ],

    # Interaction modes / dynamics to include in the experiments
    "dynamics": [
        InteractiveMode.STATIC,
        InteractiveMode.DIRECT_EXPERIMENT_AND_EVALUATION,
        InteractiveMode.SYMBOLIC_REGRESSION
    ],

    # LLM used for representation during paper dataset creation.
    "llm_model": LLMModel.GPT_4O,

    # "file_rendering_config": None,
    # File rendering options (can include multiple entries)
    # "file_rendering_config": [
    #     {
    #         "rendering_mode": RenderingMode.PDF,
    #         "rendering_prompt_md_path": (
    #             "data_management/rendering/pdf/prompts/programmatic_template_prompt.md"
    #         ),
    #         "rendering_LLM_model": LLMModel.GEMINI_3_PRO,
    #         "num_files": 1,
    #     }
    # ],
}

__all__ = ["SEDRAS_2026_PAPER_DATA_CREATION_CONFIG"]
