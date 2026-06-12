#!/usr/bin/env python
"""
Generate DatasetInstanceConfig combinations for paper data creation sweeps.

Usage:
    python paper_data_creation.py SEDRAS_2026
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional
from enum import Enum
import multiprocessing

from dotenv import load_dotenv

# 1️⃣ Load environment variables from .env at startup
load_dotenv()

from configs.instances.SEDRAS_2026PaperDataCreation import SEDRAS_2026_PAPER_DATA_CREATION_CONFIG
from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from data_management.rendering import RenderingMode
from inference.model_wrappers.llm_wrapper import LLMModel
from dataclasses import asdict
from pathlib import Path
import json

from data_management.paper_data_creation import create_dataset_for_paper_run

AVAILABLE_CONFIGS = {
    "SEDRAS_2026": SEDRAS_2026_PAPER_DATA_CREATION_CONFIG,
}

DEFAULT_RENDERING_CONFIG = {
    'rendering_mode': RenderingMode.RANDOM,
    'num_files': 1,
},


def _parse_llm_model(value: Any) -> LLMModel:
    """Parse an LLM model from enum/name/value-like input."""
    if isinstance(value, LLMModel):
        return value
    if isinstance(value, str):
        if value in LLMModel.__members__:
            return LLMModel[value]
        for model in LLMModel:
            if model.value.litellm == value or model.value.native == value:
                return model
    return LLMModel.GPT_4O


@dataclass(frozen=True)
class DatasetInstanceConfig:
    udd_complexity: int
    num_samples: int
    representation_level: Optional[RepresentationLevel]
    dynamic: InteractiveMode
    llm_model: LLMModel
    file_rendering: Optional[Dict[str, Any]]
    max_genetic_search_configurations: Optional[int]
    genetic_search_max_iterations: Optional[int]

    def get_core_settings(self) -> Dict[str, Any]:
        """Returns core settings (excluding file_rendering) as a dict."""
        return {
            "udd_complexity": self.udd_complexity,
            "num_samples": self.num_samples,
            "representation_level": getattr(self.representation_level, "value", None),
            "dynamic": self.dynamic.value,
            "llm_model": self.llm_model.name,
            "max_genetic_search_configurations": self.max_genetic_search_configurations,
            "genetic_search_max_iterations": self.genetic_search_max_iterations,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns only the configuration fields required to reconstruct
        the object via load_from_json.
        """
        # 1. Start with the core settings used by the loader
        data = self.get_core_settings()

        # 2. Only add the specific keys your load_from_json looks for
        # We serialize them safely, but we keep it lean.
        data["file_rendering"] = self._serialize_value(self.file_rendering)

        # If LLMIdentifier isn't needed for reconstruction,
        # the recursion below will safely stringify it or you could
        # even filter it out of the dictionary here.
        return data

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """A lean recursive serializer with a safety catch-all."""
        from enum import Enum
        from pathlib import Path
        from dataclasses import is_dataclass, asdict

        # If it's a dataclass (like LLMIdentifier) and we don't need to
        # reconstruct it as an object, converting it to a dict is safest.
        if is_dataclass(value):
            return DatasetInstanceConfig._serialize_value(asdict(value))

        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {str(k): DatasetInstanceConfig._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [DatasetInstanceConfig._serialize_value(item) for item in value]

        # Final safety: If it's not a basic JSON type, stringify it.
        # This prevents the LLMIdentifier crash even if it's still in the data.
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return str(value)

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Adding default=str here is the ultimate 'foolproof' fix
            json.dump(self.to_dict(), f, indent=4, sort_keys=True, default=str)
        print(f"✅ Config saved to {path}")

    @classmethod
    def load_from_json(cls, path: str | Path) -> DatasetInstanceConfig:
        """Loads a configuration from a JSON file and returns an instance."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_text_level = data.get("representation_level")
        parsed_text_level = RepresentationLevel.parse(raw_text_level)
        representation_level = None if parsed_text_level == RepresentationLevel.UNKNOWN else parsed_text_level

        llm_model_raw = data.get("llm_model", LLMModel.GPT_4O.name)
        llm_model = _parse_llm_model(llm_model_raw)

        return cls(
            udd_complexity=data["udd_complexity"],
            num_samples=data["num_samples"],
            # Convert string/int values back into Enum members
            representation_level=representation_level,
            dynamic=InteractiveMode(data["dynamic"]),
            llm_model=llm_model,
            file_rendering=data.get("file_rendering"),
            max_genetic_search_configurations=data.get("max_genetic_search_configurations"),
            genetic_search_max_iterations=data.get("genetic_search_max_iterations"),
        )

    def is_identical_to(self, other: Any) -> bool:
        if not isinstance(other, DatasetInstanceConfig):
            return False

        keys_to_ignore = ["file_rendering"]

        dict_self = self.to_dict()
        dict_other = other.to_dict()

        for key in keys_to_ignore:
            dict_self.pop(key, None)
            dict_other.pop(key, None)

        if dict_self != dict_other:
            return False
        return True


def _load_config(name: str) -> Dict[str, Any]:
    if name not in AVAILABLE_CONFIGS:
        available = ", ".join(sorted(AVAILABLE_CONFIGS.keys()))
        print(f"Unknown config '{name}'.")
        print(f"Available options: {available}")
        sys.exit(1)
    return AVAILABLE_CONFIGS[name]


def filter_away_existing_datasets(
        root_path: str | Path,
        configs: List[DatasetInstanceConfig],
        config_filename: str = "paper_dataset_instance.json"
) -> List[DatasetInstanceConfig]:
    """
    Recursively finds existing config files and filters them out of the provided list.

    Args:
        root_path: The base directory where datasets are stored.
        configs: The list of configurations generated by the sweep.
        config_filename: The name of the JSON file to look for.

    Returns:
        A list of DatasetInstanceConfig objects that do not exist on disk.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Directory {root} does not exist. Returning all configs.")
        return configs

    # 1. Recursively find all config files and load them into a list
    existing_configs: List[DatasetInstanceConfig] = []

    # rglob("*") searches recursively for the specific filename
    for path in root.rglob(config_filename):
        try:
            existing_configs.append(DatasetInstanceConfig.load_from_json(path))
        except Exception as e:
            print(f"⚠️ Warning: Failed to load config at {path}: {e}")

    print(f"🔍 Found {len(existing_configs)} existing datasets on disk.")

    # 2. Filter the input list
    # We keep 'cfg' only if it is NOT identical to any config found on disk
    remaining_configs = [
        cfg for cfg in configs
        if not any(cfg.is_identical_to(existing) for existing in existing_configs)
    ]

    print(f"✨ {len(remaining_configs)} / {len(configs)} configs remaining to be processed.")
    return remaining_configs

def build_dataset_instance_configs(config: Dict[str, Any]) -> List[DatasetInstanceConfig]:
    """Return all DatasetInstanceConfig combinations from a config dict."""
    udd_complexities = config.get("udd_complexities", [])
    num_samples = config.get("num_samples", [])
    representation_levels = config.get("representation_levels", [])
    dynamics = config.get("dynamics", [])
    rendering_config = config.get("file_rendering_config", [None])
    llm_model = _parse_llm_model(config.get("llm_model", LLMModel.GPT_4O))
    max_genetic_search_configurations = config.get("max_genetic_search_configurations")
    genetic_search_max_iterations = config.get("genetic_search_max_iterations")

    combos: List[DatasetInstanceConfig] = []
    for udd_complexity, sample_count, text_level, dynamic in product(
        udd_complexities, num_samples, representation_levels, dynamics
    ):
        if(text_level == RepresentationLevel.FILES and rendering_config is None):
            rendering_config = DEFAULT_RENDERING_CONFIG

        combos.append(
            DatasetInstanceConfig(
                udd_complexity=udd_complexity,
                num_samples=sample_count,
                representation_level=text_level,
                dynamic=dynamic,
                llm_model=llm_model,
                file_rendering=rendering_config,
                max_genetic_search_configurations=max_genetic_search_configurations,
                genetic_search_max_iterations=genetic_search_max_iterations,
            )
        )
    return combos


def _create_dataset_worker(args: tuple[DatasetInstanceConfig, Path]) -> DatasetInstanceConfig:
    ds_config, save_root = args
    create_dataset_for_paper_run(ds_config, save_root)
    return ds_config


def main() -> None:
    if len(sys.argv) != 2:
        available = ", ".join(sorted(AVAILABLE_CONFIGS.keys()))
        print("Usage: python paper_data_creation.py <PersonName>")
        print(f"Available PersonName options: {available}")
        sys.exit(1)

    person_name = sys.argv[1]
    config = _load_config(person_name)

    # 1. Generate all possible combinations from the config
    all_combos = build_dataset_instance_configs(config)
    total_count = len(all_combos)

    print(f"🚀 Building DatasetInstanceConfigs for: {person_name}")
    print(f"Total potential combinations: {total_count}")

    # 2. Filter out already existing datasets on disk
    # Adjust "data/output" to your actual storage directory
    save_root = config.get("save_folder", Path("data/paper_data"))

    run_configs = filter_away_existing_datasets(
        root_path=save_root,
        configs=all_combos
    )

    # 3. Calculate and report stats
    remaining_count = len(run_configs)
    already_done = total_count - remaining_count

    print("-" * 30)
    print(f"✅ Already created: {already_done}")
    print(f"⏳ To be created:    {remaining_count}")
    print("-" * 30)

    if remaining_count == 0:
        print("All datasets already exist. Nothing to do! 🎉")
        return


    print(f"💾 Datasets will be stored under: {save_root}")
    input("Press Enter to start dataset creation...")

    number_of_parallel_workers = max(1, int(config.get("number_of_parallel_workers", 1)))
    if number_of_parallel_workers == 1:
        for idx, ds_config in enumerate(run_configs, start=1):
            print(f"\n[{idx}/{remaining_count}] Creating dataset:")
            print(json.dumps(ds_config.to_dict(), indent=2, sort_keys=True, default=str))
            create_dataset_for_paper_run(ds_config, save_root)
            print("✅ Completed dataset creation.")
        return

    print(f"⚙️ Using {number_of_parallel_workers} worker processes.")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=number_of_parallel_workers) as pool:
        for idx, ds_config in enumerate(
            pool.imap_unordered(
                _create_dataset_worker,
                [(ds_config, save_root) for ds_config in run_configs],
            ),
            start=1,
        ):
            print(f"\n[{idx}/{remaining_count}] Completed dataset creation:")
            print(json.dumps(ds_config.to_dict(), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
