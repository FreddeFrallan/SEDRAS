# evaluation_config.py
"""
Template config for LLM evaluation over dataset instances.

Typical usage:

    from evaluation_config import EVALUATION_CONFIG
    from data_management.dataset import AbstractDataset
    from evaluation.textual_llm_evaluation import evaluate_parallel

    summary = evaluate_parallel([EVALUATION_CONFIG])
"""

from data_management.dataset import RepresentationLevel
from configs.interactive_config import InteractiveConfig, InteractiveMode, OracleConfig
from inference.model_wrappers.llm_wrapper import LLMModel


EVALUATION_CONFIG = {
    # --- Dataset roots to evaluate over ---
    # These should be paths passed to AbstractDataset.load(path)
    "dataset_paths": [
        # Example:
        # "datasets/Fredrik_pipeline_experiment_ns10_nv5_nr4_c2-5",
    ],

    # --- Which dataset instances to evaluate ---
    # Names of instances inside AbstractDataset (e.g., "medical_case_report")
    "instance_names": [

    ],

    # --- Models to evaluate ---
    # Use actual LLMModel enum values (consistent with your other configs)
    "model_names": [LLMModel.GEMINI_2_5_FLASH],

    # --- Textualization levels ---
    # Multi-level support: list of TextualizationLevel or strings.
    # If you want to evaluate a specific level, set this; otherwise leave as None.
    "levels": [RepresentationLevel.FREE_TEXT_LONG],

    # --- Evaluation behavior ---
    "num_iterations": 3,
    "return_per_sample": True,
    "max_workers": 40,
    "timeout_per_task": None,   # or e.g. 60.0 for 60 seconds per task
    "verbose": True,

    # --- Debug configuration ---
    "debug": None,

    # --- Interactive sessions ---
    "interactive_config": InteractiveConfig(
        mode=InteractiveMode.STATIC,
        verbose=False,
        max_experiments=50,
        oracle_config=OracleConfig(
            llm_model=LLMModel.GEMINI_2_5_FLASH,
            verbose=False,
            kwargs={},
        ),
    ),
}
