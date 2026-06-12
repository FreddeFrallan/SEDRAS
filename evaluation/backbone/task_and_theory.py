from inference.model_wrappers.llm_wrapper import LLMModel
from data_management.dataset import RepresentationLevel
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

# dataset_idx, model, level, instance_name, iteration
Task = Tuple[
    int,
    LLMModel,
    RepresentationLevel,
    str,
    int,
]


@dataclass
class InducedTheory:
    theory_text: str
    variable_order: List[str]
    schema: Dict[str, Dict[str, Any]]
    guidance_prompt: str  # exact prompt sent to LLM (for debugging)
