#!/usr/bin/env python3
"""
Create a subset clone of a dataset folder produced by AbstractDataset.

This script:
  1) Loads an existing dataset root (expects UDD.json, raw_samples.json, instances/, etc.).
  2) Selects a subset of raw samples, optionally balanced by label (0/1).
  3) Applies the SAME subset (by sample-id) to ALL instances.
  4) Saves the subset as a new dataset root.

Assumptions & behavior:
- Sample IDs are inferred in this priority: sample.metadata['sample_id'] → sample.sample_id → dict['sample_id'] → dict['id'] → positional index fallback.
- If an instance has explicit IDs for all its samples, we filter by those IDs.
- Otherwise, we require positional alignment (len(instance.samples) == len(raw_samples)). If not, we raise unless you pre-add explicit IDs.
- Balanced selection aims for 50/50 labels (0/1). If impossible, we pick the largest even balanced total ≤ requested and warn.

Usage:
    python subset_dataset.py \
        --in /path/to/dataset_root \
        --out /path/to/subset_root \
        --num 200 \
        --balance 1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple



# --- Import your existing classes ---

from data_management.dataset import AbstractDataset, DatasetInstance, DatasetSample  # noqa: F401



def _extract_label(s: Any) -> Optional[int]:
    if hasattr(s, "label"):
        try:
            return int(getattr(s, "label"))
        except Exception:
            return None
    if isinstance(s, dict) and "label" in s:
        try:
            return int(s["label"])
        except Exception:
            return None
    return None


def _extract_sample_id(s: Any, fallback_index: Optional[int]) -> Optional[int]:
    # metadata.sample_id
    try:
        meta = getattr(s, "metadata", None)
        if isinstance(meta, dict) and "sample_id" in meta:
            return int(meta["sample_id"])
    except Exception:
        pass
    # attribute sample_id
    try:
        if hasattr(s, "sample_id"):
            return int(getattr(s, "sample_id"))
    except Exception:
        pass
    # dict keys
    if isinstance(s, dict):
        for key in ("sample_id", "id"):
            if key in s:
                try:
                    return int(s[key])
                except Exception:
                    pass
    # fallback
    return fallback_index


def choose_subset_indices(labels: List[Optional[int]], num_target: int, balance: bool, seed: Optional[int]) -> List[int]:
    N = len(labels)
    all_idx = list(range(N))
    if seed is not None:
        random.seed(seed)

    if not balance:
        return sorted(random.sample(all_idx, num_target))

    # Balanced path (binary labels 0/1)
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]

    if not idx0 or not idx1:
        print("⚠️ Balance requested but one class is missing; falling back to uniform sampling.")
        return sorted(random.sample(all_idx, num_target))

    per_class = num_target // 2
    cap = min(len(idx0), len(idx1), per_class)
    if cap == 0:
        raise ValueError(
            f"Cannot create a balanced subset of size {num_target}: class0={len(idx0)}, class1={len(idx1)}"
        )
    final_size = 2 * cap
    if final_size < num_target:
        print(
            f"ℹ️ Requested {num_target} balanced samples, limited to {final_size} by class counts. Using {final_size}."
        )
    sel = random.sample(idx0, cap) + random.sample(idx1, cap)
    return sorted(sel)


def subset_dataset(in_root: str, out_root: str, num_target: int, balance: bool = True, seed: Optional[int] = None) -> None:
    # Load
    in_root = os.path.abspath(in_root)
    out_root = os.path.abspath(out_root)

    ds = AbstractDataset.load(in_root)

    N = len(ds.raw_samples)
    if num_target >= N:
        raise ValueError(f"--num ({num_target}) must be smaller than number of raw samples ({N}).")

    # Build raw ids + labels
    raw_ids: List[int] = []
    labels: List[Optional[int]] = []
    for i, s in enumerate(ds.raw_samples):
        sid = _extract_sample_id(s, fallback_index=i)
        raw_ids.append(sid)
        labels.append(_extract_label(s))

    # Choose subset
    selected_indices = choose_subset_indices(labels, num_target, balance, seed)
    selected_raw_ids = {raw_ids[i] for i in selected_indices}

    # Filter raw samples
    new_raw_samples = [copy.deepcopy(ds.raw_samples[i]) for i in selected_indices]

    # ---- NEW: handle nested layout: {level -> {name -> DatasetInstance}} ----
    new_instances_by_level: Dict[Any, Dict[str, DatasetInstance]] = {}

    if not isinstance(ds.dataset_instances, dict) or not ds.dataset_instances:
        raise ValueError("Dataset has no instances at any representation level.")

    for lvl, by_name in ds.dataset_instances.items():
        # lvl is a RepresentationLevel (or UNKNOWN in legacy), by_name is {name -> DatasetInstance}
        if not isinstance(by_name, dict):
            continue

        new_instances_by_level.setdefault(lvl, {})

        for name, inst in by_name.items():
            inst_samples = getattr(inst, "samples", None)
            if not isinstance(inst_samples, list):
                raise ValueError(f"Instance '{name}' at level '{getattr(lvl, 'value', lvl)}' has no 'samples' list; cannot subset.")

            # Collect per-sample IDs for this instance
            inst_ids: List[Optional[int]] = []
            explicit_count = 0
            for s in inst_samples:
                sid = _extract_sample_id(s, fallback_index=None)
                inst_ids.append(sid)
                if sid is not None:
                    explicit_count += 1

            # Filter this instance's samples using either explicit IDs or positional fallback
            if explicit_count == len(inst_samples):
                # All samples have explicit IDs
                id_to_samples: Dict[int, List[Any]] = {}
                for s, sid in zip(inst_samples, inst_ids):
                    id_to_samples.setdefault(int(sid), []).append(s)

                filtered_samples: List[Any] = []
                for idx in selected_indices:
                    want_id = raw_ids[idx]
                    if want_id not in id_to_samples or not id_to_samples[want_id]:
                        raise ValueError(
                            f"Instance '{name}' (level='{getattr(lvl, 'value', lvl)}') missing required sample id={want_id}. "
                            f"Add consistent IDs across instances."
                        )
                    filtered_samples.append(id_to_samples[want_id].pop(0))
            else:
                # Positional fallback: require same length as raw set
                if len(inst_samples) != N:
                    raise ValueError(
                        f"Instance '{name}' (level='{getattr(lvl, 'value', lvl)}') cannot align by position: "
                        f"len={len(inst_samples)} != raw={N}. Add explicit IDs."
                    )
                filtered_samples = [copy.deepcopy(inst_samples[i]) for i in selected_indices]

            # Rebuild instance (preserve fields, including representation_level)
            new_inst = type(inst)(
                name=inst.name,
                udd=inst.udd,
                samples=filtered_samples,
                metadata=copy.deepcopy(getattr(inst, "metadata", {})),
                text_mapping=getattr(inst, "text_mapping", None),
                instance_hint_used=getattr(inst, "instance_hint_used", None),
                representation_level=getattr(inst, "representation_level", getattr(lvl, "value", "unknown")),
            )
            new_instances_by_level[lvl][name] = new_inst

    # Assemble new dataset (preserve nested structure)
    new_ds = type(ds)(
        udd=copy.deepcopy(ds.udd),
        dataset_instances=new_instances_by_level,
        raw_samples=new_raw_samples,
        metadata=copy.deepcopy(ds.metadata),
    )
    new_ds.evaluation_results = {}   # reset evals for the subset
    new_ds._root_folder = None

    # lineage note
    try:
        new_ds.metadata = copy.deepcopy(new_ds.metadata) or {}
        new_ds.metadata.setdefault(
            "subset_info",
            {
                "source": "subset_dataset.py",
                "original_num_samples": N,
                "selected_num_samples": len(selected_indices),
                "requested_num_target_samples": num_target,
                "balance": balance,
                "seed": seed,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
        )
    except Exception:
        pass

    # Save
    os.makedirs(out_root, exist_ok=True)
    new_ds.save(out_root)
    print(f"✅ Subset dataset saved to {out_root}")





def main() -> None:
    # p = argparse.ArgumentParser(description="Create a subset clone of an AbstractDataset folder.")
    # p.add_argument("--in", dest="in_root", required=True, help="Input dataset root folder")
    # p.add_argument("--out", dest="out_root", required=True, help="Output subset dataset root folder")
    # p.add_argument("--num", dest="num_target", type=int, required=True, help="Target number of samples (< total)")
    # p.add_argument("--balance", type=int, default=1, help="1 to balance labels 0/1, 0 for uniform sampling")
    # p.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    # args = p.parse_args()

    starting_samples = 80
    original_f_path = "/home/fredde/Documents/Github/InterpretableTheories/textualized_dataset-5vars-{}samples"

    target_sizes = [60, 40, 20, 10]
    for size in target_sizes:
        in_dataset_path = original_f_path.format(starting_samples)
        out_dir = original_f_path.format(size)
        num_samples = size
        balance_flag = True

        subset_dataset(
            in_root=in_dataset_path,
            out_root=out_dir,
            num_target=num_samples,
            balance=balance_flag,
            # seed=args.seed,
        )
    # out_dir = "/home/fredde/Documents/Github/InterpretableTheories/textualized_dataset-5vars-60samples"



if __name__ == "__main__":
    main()
