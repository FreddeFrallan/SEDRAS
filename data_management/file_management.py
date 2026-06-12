from __future__ import annotations

from typing import List, Tuple
import os

from data_management.dataset import AbstractDataset


def _find_datasets_in_path(path: str) -> List[str]:
    """Return dataset roots (folders containing UDD.json) under the given path."""
    path = os.path.abspath(path)
    udd_filename = "UDD.json"

    if os.path.isfile(os.path.join(path, udd_filename)):
        return [path]

    discovered: List[str] = []
    if os.path.isdir(path):
        for current_root, dirnames, filenames in os.walk(path):
            if udd_filename in filenames:
                discovered.append(current_root)
                dirnames[:] = []
    return discovered


def _resolve_dataset_paths(dataset_paths: List[str], *, verbose: bool = False) -> List[str]:
    """Expand and deduplicate dataset paths while preserving discovery order."""
    resolved: List[str] = []
    seen: set[str] = set()

    for path in dataset_paths:
        found = _find_datasets_in_path(path)
        if not found and verbose:
            print(f"   ⚠️ No datasets found under {path}")
        for candidate in found:
            if candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)

    return resolved


def load_all_disks_from_root_path(dataset_paths: List[str], *, verbose: bool = False) -> List[Tuple[str, AbstractDataset]]:
    """Resolve dataset roots and load all matching datasets from disk."""
    resolved_paths = _resolve_dataset_paths(dataset_paths, verbose=verbose)
    loaded_datasets: List[Tuple[str, AbstractDataset]] = []

    for path in resolved_paths:
        if verbose:
            print(f"   🔎 {path}")
        ds = AbstractDataset.load(path)
        loaded_datasets.append((path, ds))

    return loaded_datasets
