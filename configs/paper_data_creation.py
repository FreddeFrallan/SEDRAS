"""
Paper-facing data creation configuration template.

This module defines a dictionary-based config for running paper experiments.
It mirrors other config modules in the repo by exporting a single top-level
dictionary that callers can copy and modify as needed.

Keys:
- ``save_folder`` (Path): Base output directory for generated datasets.
- ``number_of_parallel_workers`` (int): Number of worker processes to use for dataset creation.
- ``num_samples`` (List[int]): Dataset sizes to sweep over.
- ``representation_levels`` (List[RepresentationLevel]): Representation levels
  to generate.
- ``dynamics`` (List[InteractiveMode]): Interaction protocols to evaluate.
- ``file_renderings`` (List[dict]): Rendering configs describing which file
  formats to materialize for each dataset instance.
- ``llm_model`` (LLMModel): LLM used for representation when generating
  text-based dataset instances.
- ``max_genetic_search_configurations`` (Optional[int]): Cap on configurations
  evaluated during genetic UDD search (None for exhaustive).
- ``genetic_search_max_iterations`` (Optional[int]): Max iterations for genetic
  UDD search (None for default behavior).
"""

from pathlib import Path

from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from data_management.rendering import RenderingMode
from inference.model_wrappers.llm_wrapper import LLMModel

PAPER_DATA_CREATION_CONFIG = {
    # Root folder for saving generated datasets
    "save_folder": Path("datasets/paper_runs"),

    # Number of worker processes to use during dataset creation
    "number_of_parallel_workers": 1,

    "udd_complexities": [1, 2, 3, 4, 5],

    # Dataset sizes to generate in the sweep
    "num_samples": [40, 80],

    # Representation levels to create for each dataset
    "representation_levels": [
        RepresentationLevel.FREE_TEXT,
        RepresentationLevel.FREE_TEXT_LONG,
    ],

    # Interaction modes / dynamics to include in the experiments
    "dynamics": [
        InteractiveMode.STATIC,
    ],

    # LLM used to generate textualized dataset instances
    "llm_model": LLMModel.GPT_4O,

    # File rendering options (can include multiple entries)
    "file_rendering_config": [
        {
            "rendering_mode": RenderingMode.RANDOM,
            "rendering_LLM_model": LLMModel.GEMINI_2_5_PRO,
            "num_files": 1,
        }
    ],

    # Limit configurations evaluated during genetic UDD search (None = exhaustive).
    "max_genetic_search_configurations": None,
    # Max iterations for genetic UDD search (None = default behavior).
    "genetic_search_max_iterations": None,
}

__all__ = ["PAPER_DATA_CREATION_CONFIG"]
