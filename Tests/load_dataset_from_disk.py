#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

from pathlib import Path
from typing import Any

# Ensure project-root imports work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_management.dataset import RepresentationLevel
from data_management.file_management import load_all_disks_from_root_path


def _count_eval_runs(model_payload: Any) -> int:
    """Recursively count evaluation run dicts from mixed/legacy structures."""
    if isinstance(model_payload, list):
        return len(model_payload)
    if isinstance(model_payload, dict):
        return sum(_count_eval_runs(v) for v in model_payload.values())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load all datasets under a root path and print summary statistics "
            "(dataset count, instance-level distribution, and eval runs per model)."
        )
    )
    parser.add_argument("root_path", help="Root path containing one or more dataset folders.")
    args = parser.parse_args()

    root_path = os.path.abspath(args.root_path)
    loaded = load_all_disks_from_root_path([root_path], verbose=False)

    print(f"Scanned root path: {root_path}")
    print(f"Datasets loaded: {len(loaded)}")

    instance_level_counter: Counter[str] = Counter()
    eval_runs_per_model: Counter[str] = Counter()
    raw_only_dataset_count = 0

    # Track datasets with > 1 "files" instances
    multi_files_datasets: list[str] = []

    for path, dataset in loaded:
        for level, by_name in dataset.dataset_instances.items():
            level_key = getattr(level, "value", str(level))
            instance_level_counter[level_key] += len(by_name)

        # Track if files level has more than 1 instance
        files_instances = dataset.dataset_instances.get(RepresentationLevel.FILES, {})
        if len(files_instances) > 1:
            multi_files_datasets.append(path)

        has_raw_instances = bool(dataset.dataset_instances.get(RepresentationLevel.RAW, {}))
        has_non_raw_instances = any(
            bool(by_name)
            for level, by_name in dataset.dataset_instances.items()
            if level != RepresentationLevel.RAW
        )
        if has_raw_instances and not has_non_raw_instances:
            raw_only_dataset_count += 1

        for model_name, model_payload in dataset.evaluation_results.items():
            eval_runs_per_model[model_name] += _count_eval_runs(model_payload)

    print("\nInstance distribution by dataset type/representation level:")
    if instance_level_counter:
        for level, count in instance_level_counter.most_common():
            print(f"  - {level}: {count}")
    else:
        print("  - No dataset instances were found.")

    print(f"\nRaw-only datasets (only raw representation available): {raw_only_dataset_count}")

    # Print out our newly tracked "files" metrics
    print(f"\nDatasets with > 1 'files' representation level: {len(multi_files_datasets)}")
    if multi_files_datasets:
        for ds_path in multi_files_datasets:
            print(f"  - {ds_path}")

    print("\nLoaded evaluation results per model:")
    if eval_runs_per_model:
        for model_name, count in eval_runs_per_model.most_common():
            print(f"  - {model_name}: {count}")
    else:
        print("  - No evaluation results were found.")


if __name__ == "__main__":
    main()