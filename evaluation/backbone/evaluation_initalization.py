from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import copy
import json

from configs.interactive_config import InteractiveConfig
from configs.utils import convert_config_to_json_default
from data_management.dataset import AbstractDataset, RepresentationLevel
from data_management.file_management import load_all_disks_from_root_path
from evaluation.backbone.debugging_utils import DebugConfig
from evaluation.backbone.task_and_theory import Task
from inference.model_wrappers.llm_wrapper import LLMModel


def create_evaluation_tasks(
    model_names: List[LLMModel],
    targets: List[Tuple[int, RepresentationLevel, str]],  # (dataset_idx, level, instance_name)
    num_iterations: int,
) -> List[Task]:
    """Build the full task grid: dataset_idx × model × level × instance × iteration."""
    tasks: List[Task] = []
    for (ds_idx, lv, inst) in targets:
        for m in model_names:
            for it in range(num_iterations):
                tasks.append((ds_idx, m, lv, inst, it))
    return tasks


def prepare_evaluation_jobs(configs: List[Dict[str, Any]], *, verbose: bool = True) -> List[Dict[str, Any]]:
    """Create normalized per-config job descriptors for parallel evaluation."""
    config_jobs: List[Dict[str, Any]] = []

    for cfg_idx, cfg in enumerate(configs, start=1):
        cfg = copy.deepcopy(cfg)
        config_label = cfg.get("config_name") or f"in_memory_config_{cfg_idx}"

        interactive_config = cfg.get("interactive_config")
        interactive_configs: List[Optional[InteractiveConfig]] = (
            interactive_config if isinstance(interactive_config, list) else [interactive_config]
        )

        for idx, single_interactive_config in enumerate(interactive_configs, start=1):
            config_suffix = ""
            if len(interactive_configs) > 1:
                mode = getattr(single_interactive_config, "mode", "unknown")
                config_suffix = f"::interactive[{idx}]({mode})"

            config_key = f"{config_label}{config_suffix}"
            run_cfg = {**cfg, "interactive_config": single_interactive_config}

            if run_cfg.get("model_names") is None or len(run_cfg.get("model_names")) == 0:
                raise ValueError(
                    f"model_names must be a non-empty list of LLMModel in {config_label}."
                )

            for model_name in run_cfg.get("model_names"):
                print(f"🔧 Config {config_key} includes model: {model_name.name}")

            debug = run_cfg.get("debug") or DebugConfig(enabled=False)

            evaluation_config = {"config_key": config_key, **run_cfg}
            print(f"🔍 Validating evaluation configuration serializability for {config_key}...")
            temp = convert_config_to_json_default(evaluation_config)
            json.dumps(temp)
            print("   ✅ Configuration is JSON-serializable.")

            dataset_jobs: List[Tuple[str, AbstractDataset]] = []
            dataset_paths = run_cfg.get("dataset_paths")
            abstract_dataset = run_cfg.get("abstractDataset")

            if dataset_paths:
                dataset_jobs = load_all_disks_from_root_path(dataset_paths, verbose=verbose)
                if verbose:
                    print(f"🚀 Loading {len(dataset_jobs)} dataset(s) for {config_key} ...")
            elif abstract_dataset is not None:
                dataset_jobs.append(("<in_memory_dataset>", abstract_dataset))
            else:
                raise ValueError(
                    "You must provide either `abstractDataset` (single dataset) "
                    "or `dataset_paths` (list of dataset roots)."
                )

            if not dataset_jobs:
                if verbose:
                    print(f"⚠️ No datasets provided for {config_key}; skipping.")
                continue

            num_iterations = run_cfg.get("num_iterations", 1)
            if num_iterations < 1:
                raise ValueError("num_iterations must be >= 1")

            instance_names = run_cfg.get("instance_names")
            levels = run_cfg.get("levels")
            level = run_cfg.get("level")

            all_targets: List[Tuple[int, RepresentationLevel, str]] = []
            for ds_idx, (ds_path, ds) in enumerate(dataset_jobs):
                if levels is not None:
                    resolved_levels = [RepresentationLevel.parse(lv) for lv in levels]
                elif level is not None:
                    resolved_levels = [RepresentationLevel.parse(level)]
                else:
                    resolved_levels = list(ds.dataset_instances.keys())
                    if not resolved_levels:
                        raise ValueError(f"Dataset '{ds_path}' has no instances at any representation level.")

                for lv in resolved_levels:
                    if lv not in ds.dataset_instances:
                        raise KeyError(
                            f"Dataset '{ds_path}': Level '{lv.value}' has no dataset instances. "
                            f"Available: {[l.value for l in ds.dataset_instances]}"
                        )

                for lv in resolved_levels:
                    available = ds.dataset_instances[lv]
                    if not instance_names:
                        selected = list(available.keys())
                    else:
                        missing = [n for n in instance_names if n not in available]
                        if missing:
                            print(
                                f"⚠️ Dataset '{ds_path}', level '{lv.value}': "
                                f"missing instance(s): {missing}. Skipping these."
                            )
                        selected = [n for n in instance_names if n in available]

                    for inst in selected:
                        all_targets.append((ds_idx, lv, inst))

            tasks = create_evaluation_tasks(
                model_names=run_cfg.get("model_names"),
                targets=all_targets,
                num_iterations=num_iterations,
            )

            if not tasks:
                if verbose:
                    print(
                        f"⚠️ No evaluation tasks constructed for {config_key} "
                        "(check levels/instances/settings)."
                    )
                continue

            config_jobs.append(
                {
                    "config_key": config_key,
                    "config_path": None,
                    "dataset_jobs": dataset_jobs,
                    "tasks": tasks,
                    "targets": all_targets,
                    "evaluation_config": evaluation_config,
                    "return_per_sample": run_cfg.get("return_per_sample", True),
                    "timeout_per_task": run_cfg.get("timeout_per_task"),
                    "debug": debug,
                    "interactive_config": single_interactive_config,
                }
            )

    return config_jobs
