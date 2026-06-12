from typing import Optional, List, Dict
import json

from data_management.dataset import AbstractDataset, RepresentationLevel, DatasetInstance
from data_management.textualization import text_mappings, textualize_dataset_samples
from inference.model_wrappers.llm_wrapper import LLMModel


def add_new_text_instance_dataset(
    abstract_dataset: AbstractDataset,
    llm_model: LLMModel = LLMModel.GPT_4O,
    style_hint: Optional[str] = None,
    levels: Optional[List[RepresentationLevel]] = None,
    max_text_mapping_reruns: int = 3,
) -> Dict[RepresentationLevel, DatasetInstance]:
    """Generate and attach a new textualized instance dataset.

    If ``style_hint`` is provided, it is passed through to theme generation; otherwise,
    the representation logic will randomly pick a theme as before.
    """
    rerun_counter = 0
    while True:
        try:
            mapping_json = text_mappings.generate_text_mapping_json_from_dataset(
                abstract_dataset,
                style_hint=style_hint,
                indent=2,
                llm_model=llm_model,
            )

            mapping_json = json.loads(mapping_json)
            break
        except Exception as exc:
            rerun_counter += 1
            if rerun_counter >= max_text_mapping_reruns:
                raise RuntimeError(
                    f"Failed to generate text mapping JSON after {max_text_mapping_reruns} attempts."
                ) from exc

    return textualize_dataset_samples.textualize_dataset_samples(
        abstract_dataset,
        mapping_json,
        llm_model=llm_model,
        levels=levels,
    )
