from pathlib import Path

from configs.main_evaluation_config import MAIN_EVALUATION_CONFIG
from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from inference import LLMModel
from inference.enums import LLMModel


SEDRAS_2026_MAIN_EVALUATION_CONFIG = {
    **MAIN_EVALUATION_CONFIG,
    "dataset_root": Path("evals_20_no_evals"),
    # "dataset_root": Path("udd_paper_datasets_balanced_udd/udd_paper_datasets_balanced_udd/static/raw"),
    
    "target_representation_levels": [
        None,
        RepresentationLevel.FREE_TEXT,
        RepresentationLevel.FREE_TEXT_LONG,
        RepresentationLevel.FILES,
    ],
    "target_dynamics_levels": [
        InteractiveMode.STATIC,
        # InteractiveMode.DIRECT_EXPERIMENT_AND_EVALUATION,
        # InteractiveMode.SYMBOLIC_REGRESSION,
    ],
    "max_workers": 1,
    # "target_complexity_levels": [1, 2, 3, 4, 5],
    "target_complexity_levels": [1],

    # "target_llm_model": LLMModel.GPT_4O,
    # "target_llm_model": LLMModel.GPT_5_2,
    # "target_llm_model": LLMModel.GPT_5_2_MEDIUM,
    # "target_llm_model": LLMModel.GPT_5_2_HIGH,

    # "target_llm_model": LLMModel.CLAUDE_OPUS_4_5,
    # "target_llm_model": LLMModel.CLAUDE_SONNET_4_5,
    # "target_llm_model": LLMModel.CLAUDE_HAIKU_3,

    # "target_llm_model": LLMModel.GROK_4_1_reasoning,

    "target_llm_model": LLMModel.GEMINI_2_5_FLASH,
    # "target_llm_model": LLMModel.GEMINI_2_5_PRO,
    # "target_llm_model": LLMModel.GEMINI_3_PRO,

    # "target_llm_model": LLMModel.MAGISTRAL_SMALL,
    # "target_llm_model": LLMModel.MAGISTRAL_MEDIUM,
}

__all__ = ["SEDRAS_2026_MAIN_EVALUATION_CONFIG"]
