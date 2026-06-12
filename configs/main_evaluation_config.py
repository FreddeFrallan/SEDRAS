"""
Template configuration for the paper evaluation discovery script.

The config intentionally mirrors the lightweight, dictionary-driven style used
throughout the project. It is consumed by ``paper.main_evaluation`` to find and
validate datasets before running evaluation sweeps.

Fields
------
- ``dataset_root``: Base path where datasets are stored. The script will
  recursively search this tree for ``paper_dataset_instance.json`` files.
- ``target_representation_levels``: Allowed representation levels (as
  ``RepresentationLevel`` members or strings). Use ``None`` to match raw
  datasets without representation.
- ``target_dynamics_levels``: Allowed interaction protocols for the dataset
  (``InteractiveMode`` values or strings).
- ``target_complexity_levels``: Allowed UDD complexity integers.
- ``target_llm_model``: Target LLM to use during evaluation.
"""

from pathlib import Path

from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from inference.enums import LLMModel


MAIN_EVALUATION_CONFIG = {
    "dataset_root": Path("udd_paper_datasets_balanced_udd"),

    "target_representation_levels": [
        None,
        RepresentationLevel.FREE_TEXT,
        RepresentationLevel.FREE_TEXT_LONG,
    ],

    "target_dynamics_levels": [
        InteractiveMode.STATIC,
    ],

    "target_complexity_levels": [1, 2, 3, 4, 5],

    "target_llm_model": LLMModel.GPT_5,
    "max_workers": 5,
}

__all__ = ["MAIN_EVALUATION_CONFIG"]
