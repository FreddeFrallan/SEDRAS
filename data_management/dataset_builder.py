# dataset_builder.py
from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path

import tqdm

from data_management.rendering import RenderingMode, render_dataset_instance_to_file
from data_management.dataset import DatasetSample, RepresentationLevel
from data_management.underlying_data.output_properties import (
    OutputProperty,
)
from data_management.creation.create_balanced_dataset import create_balanced_dataset
from data_management.textualization import create_text_instance_dataset
from data_management.underlying_data.udd_search import UDDSearchMethod
from inference.model_wrappers.llm_wrapper import LLMModel

# =========================
# Dataset creation utilities
# =========================

def build_save_path(
    main_name: str,
    *,
    num_samples: int,
    num_variables: int,
    num_rules: int,
    min_categories: int,
    max_categories: int,
    num_numerical_variables: int,
    num_output_labels: int,
    output_properties: Optional[List[OutputProperty]] = None,
) -> str:
    """
    Build a save_path string that encodes the key structural parameters.

    Example:
        main_name="textualized_dataset" ->
        "textualized_dataset_ns80_nv5_nr4_c2-5"
    """
    return (
        f"{main_name}"
        f"_ns{num_samples}"
        f"_nv{num_variables}"
        f"_nr{num_rules}"
        f"_c{min_categories}-{max_categories}"
        f"_nnv{num_numerical_variables}"
        f"_ol{num_output_labels}"
        f"_props{len(output_properties or [])}"
    )

def _textualize_raw_samples(samples: List[DatasetSample]):
    for s in samples:
        parts = [f"{k}={v}" for k, v in s.assignment.items()]
        s.instance_text = ", ".join(parts)
        s.instance_type = "raw"


def create_new_textualized_dataset(
    main_name: str = "textualized_dataset",
    num_dataset_runs: int = 1,
    num_text_instances: int = 3,
    llm_model: LLMModel = LLMModel.GPT_4O,
    representation_theme: Optional[str] = None,
    udd_num_samples: int = 80,
    udd_num_variables: int = 5,
    udd_min_categories: int = 2,
    udd_max_categories: int = 5,
    udd_num_numerical_variables: int = 0,
    udd_numerical_min_spans: int = 2,
    udd_numerical_max_spans: int = 4,
    udd_num_rules: int = 4,
    udd_num_output_labels: int = 2,
    udd_max_ratio_diff: float = 0.25,
    udd_min_decisive: float = 0.75,
    udd_allow_multi_use: bool = False,
    udd_seed: int = None,
    udd_sample_labels: bool = False,
    udd_search_method: UDDSearchMethod = UDDSearchMethod.REANDOM,
    max_genetic_search_configurations: Optional[int] = None,
    genetic_search_max_iterations: Optional[int] = None,
    output_properties: Optional[List[OutputProperty]] = None,
    representation_levels: Optional[List[RepresentationLevel]] = None,
    render_to_files: bool = False,
    rendering_config: Optional[Dict[str, Any]] = None,
    render_maximum_num_samples: Optional[int] = None,
) -> List[str]:
    # Default representation levels: only FREE_TEXT_LONG
    if representation_levels is None:
        representation_levels = [RepresentationLevel.FREE_TEXT_LONG]

    if num_dataset_runs < 1:
        raise ValueError("num_dataset_runs must be at least 1")

    def _run_single_dataset(run_main_name: str) -> str:
        # Derive the save path from the parameters (kept explicit here for clarity)
        # Build the save path from parameters
        save_path = build_save_path(
            main_name,
            num_samples=udd_num_samples,
            num_variables=udd_num_variables,
            num_rules=udd_num_rules,
            min_categories=udd_min_categories,
            max_categories=udd_max_categories,
            num_numerical_variables=udd_num_numerical_variables,
            num_output_labels=udd_num_output_labels,
            output_properties=output_properties,
        )
        create_kwargs: Dict[str, Any] = {
            "main_name": run_main_name,
            "save_path": save_path,
            "num_samples": udd_num_samples,
            "num_variables": udd_num_variables,
            "min_categories": udd_min_categories,
            "max_categories": udd_max_categories,
            "num_numerical_variables": udd_num_numerical_variables,
            "numerical_min_spans": udd_numerical_min_spans,
            "numerical_max_spans": udd_numerical_max_spans,
            "num_rules": udd_num_rules,
            "num_output_labels": udd_num_output_labels,
            "max_ratio_diff": udd_max_ratio_diff,
            "min_decisive": udd_min_decisive,
            "allow_multi_use": udd_allow_multi_use,
            "seed": udd_seed,
            "sample_labels": udd_sample_labels,
            "search_method": udd_search_method,
            "max_genetic_search_configurations": max_genetic_search_configurations,
            "output_properties": output_properties,
        }

        if (
            max_genetic_search_configurations is not None
            and udd_search_method == UDDSearchMethod.GENETIC
        ):
            create_kwargs["max_iterations"] = max_genetic_search_configurations

        dataset, analysis = create_balanced_dataset(
            **create_kwargs,
        )

        # Optional extra save to ensure it's persisted under the same derived path
        dataset.save(save_path)

        # Create textualized instances (possibly multiple runs)
        for _ in tqdm.tqdm(range(num_text_instances), desc="Creating text instances"):
            # Expect this to return Dict[RepresentationLevel, DatasetInstance]
            level_to_instance = create_text_instance_dataset.add_new_text_instance_dataset(
                dataset,
                llm_model=llm_model,
                style_hint=representation_theme,
                levels=representation_levels,
            )

            if render_to_files:
                rendering_cfg = rendering_config or {}
                rendering_mode = rendering_cfg.get("rendering_mode", RenderingMode.PDF)
                rendering_llm_model = rendering_cfg.get(
                    "rendering_LLM_model", LLMModel.GEMINI_3_PRO
                )
                rendering_num_files = rendering_cfg.get("num_files", 1)

                # Render the selected file format for each representation level created in this pass
                for level, inst in level_to_instance.items():
                    # `inst.name` should be the instance name registered in the dataset
                    instance_name = inst.name

                    # `level` here is a RepresentationLevel enum; render function expects a string
                    if isinstance(level, RepresentationLevel):
                        level_str = RepresentationLevel.parse(level).value
                    else:
                        level_str = str(level)

                    render_outputs = render_dataset_instance_to_file(
                        dataset_root=save_path,
                        instance_name=instance_name,
                        level=level_str,
                        rendering_mode=rendering_mode,
                        num_files=rendering_num_files,
                        # other params use defaults (prompt path, PDF page size, etc.)
                        llm_model=rendering_llm_model,
                        maximum_num_samples=render_maximum_num_samples,
                    )
                    dataset.register_rendered_artifacts(
                        level=level,
                        instance_name=instance_name,
                        artifact_paths=render_outputs,
                        rendering_mode=rendering_mode,
                    )

        return save_path

    save_paths = []
    for run_idx in range(num_dataset_runs):
        if num_dataset_runs > 1:
            base_path = Path(main_name)
            prefixed_name = f"{run_idx + 1}_{base_path.name}"
            run_main_name = str(base_path.with_name(prefixed_name))
        else:
            run_main_name = main_name
        print(f"▶️ Starting dataset run {run_idx + 1}/{num_dataset_runs} with name {run_main_name}")
        s_path = _run_single_dataset(run_main_name)
        save_paths.append(s_path)

    return save_paths
