from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from data_management.dataset_builder import create_new_textualized_dataset
from data_management.udd_complexities import COMPLEXITY_LEVELS as UDD_COMPLEXITY_LEVELS
from data_management.underlying_data.udd_search import UDDSearchMethod

if TYPE_CHECKING:  # pragma: no cover
    from paper_data_creation import DatasetInstanceConfig


def _normalize_representation_levels(
    representation_level: Optional[RepresentationLevel],
) -> Tuple[List[RepresentationLevel], int]:
    """Return a list containing the provided level, or an empty list if ``None``.

    The paper sweep includes ``None`` as a sentinel for "no representation".
    Downstream representation utilities accept an empty list to skip generating
    textualized instances, so we translate ``None`` accordingly here.
    """

    if representation_level is None:
        return [], 0
    return [representation_level], 1


def _pick_rendering_config(rendering: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Select the first truthy rendering config from the provided value."""

    if rendering is None:
        return None

    if isinstance(rendering, list):
        for candidate in rendering:
            if candidate:
                return candidate
        return None

    if isinstance(rendering, dict):
        return rendering

    return None

def load_udd_complexity(complexity_level: int) -> Dict[str, Any]:
    """Load UDD complexity parameters for the given level."""
    if complexity_level not in UDD_COMPLEXITY_LEVELS:
        raise ValueError(f"Unrecognized UDD complexity level: {complexity_level}")
    return UDD_COMPLEXITY_LEVELS[complexity_level]


def create_dataset_for_paper_run(
    dataset_config: "DatasetInstanceConfig", save_root: str | Path
) -> Path:
    """Materialize a dataset for the paper sweep.

    Args:
        dataset_config: The configuration describing the dataset instance to create.
        save_root: Root directory where generated datasets should be stored.

    Returns:
        Path to the folder where the dataset (and its config JSON) was saved.
    """

    representation_levels, num_text_instances = _normalize_representation_levels(
        dataset_config.representation_level
    )
    rendering_config = _pick_rendering_config(dataset_config.file_rendering)
    udd_configs = load_udd_complexity(dataset_config.udd_complexity)
    render_to_files = rendering_config is not None and dataset_config.representation_level == RepresentationLevel.FILES
    render_maximum_num_samples = None
    if (
        dataset_config.representation_level == RepresentationLevel.FILES
        and dataset_config.dynamic != InteractiveMode.STATIC
    ):
        render_maximum_num_samples = 10


    dynamic_label = getattr(dataset_config.dynamic, "value", str(dataset_config.dynamic))
    representation_label = (
        dataset_config.representation_level.name
        if isinstance(dataset_config.representation_level, RepresentationLevel)
        else "raw"
    )


    main_name = Path(save_root) / dynamic_label / representation_label / "data"
    main_name.parent.mkdir(parents=True, exist_ok=True)

    # print(f"LLM model: {dataset_config.llm_model}, UDD complexity level: {dataset_config.udd_complexity}, dynamic: {dataset_config.dynamic}, representation level: {dataset_config.representation_level}")
    # input(f"Representation levels: {representation_levels}, rendering_config: {rendering_config}, udd_configs: {udd_configs}, render_to_files: {render_to_files}. Press Enter to continue...")

    save_paths = create_new_textualized_dataset(
        main_name=str(main_name),
        udd_num_samples=dataset_config.num_samples,
        representation_levels=representation_levels,
        num_text_instances=num_text_instances,
        render_to_files=render_to_files,
        rendering_config=rendering_config,
        render_maximum_num_samples=render_maximum_num_samples,
        max_genetic_search_configurations=dataset_config.max_genetic_search_configurations,
        genetic_search_max_iterations=dataset_config.genetic_search_max_iterations,
        **udd_configs,

        # Settings that are fixed for the paper datasets
        udd_search_method=UDDSearchMethod.GENETIC,
        llm_model=dataset_config.llm_model,
    )

    dataset_folder = Path(save_paths[0])
    # input(f"Dataset created at {dataset_folder}. Press Enter to save config...")
    dataset_config.save_to_json(dataset_folder / "paper_dataset_instance.json")

    return dataset_folder
