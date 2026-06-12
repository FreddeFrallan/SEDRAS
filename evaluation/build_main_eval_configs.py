#!/usr/bin/env python
"""
Build in-memory evaluation configs for pending paper datasets.

Configs are grouped by textualization level and interactive dynamics to
minimize the number of evaluation jobs while still covering all pending
datasets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from configs import evaluation_config
from configs.interactive_config import InteractiveConfig, InteractiveMode, OracleConfig
from data_management.dataset import RepresentationLevel
from inference.enums import LLMModel


def _normalize_textualization_level(level: Any) -> RepresentationLevel:
    if isinstance(level, RepresentationLevel):
        return level
    if level is None:
        return RepresentationLevel.RAW
    parsed = RepresentationLevel.parse(str(level))
    return RepresentationLevel.RAW if parsed == RepresentationLevel.UNKNOWN else parsed


def _normalize_dynamic(dynamic: Any) -> InteractiveMode:
    if isinstance(dynamic, InteractiveMode):
        return dynamic
    if dynamic is None:
        return InteractiveMode.STATIC
    try:
        return InteractiveMode(dynamic)
    except ValueError:
        return InteractiveMode[dynamic]


def build_main_eval_configs(
    pending_datasets: Iterable[Any],
    *,
    target_llm_model: LLMModel,
) -> List[Dict[str, Any]]:
    """Generate grouped in-memory evaluation configs for pending datasets."""
    grouped: Dict[Tuple[RepresentationLevel, InteractiveMode, int], List[str]] = defaultdict(list)
    for dataset in pending_datasets:
        level = _normalize_textualization_level(getattr(dataset, "textualization_level", None))
        dynamic = _normalize_dynamic(getattr(dataset, "dynamic", None))
        num_samples = getattr(dataset, "num_samples", None)
        if num_samples is None:
            continue
        try:
            num_samples = int(num_samples)
        except (TypeError, ValueError):
            continue
        dataset_root = getattr(dataset, "dataset_root", None)
        if dataset_root is None:
            continue
        grouped[(level, dynamic, num_samples)].append(str(dataset_root))

    if not grouped:
        print("🟡 No pending datasets to build configs for.")
        return []

    generated_configs: List[Dict[str, Any]] = []
    for (level, dynamic, num_samples), paths in grouped.items():
        is_interactive = dynamic != InteractiveMode.STATIC
        if is_interactive:
            number_of_intro_samples = 10
            max_experiments = max(0, num_samples - number_of_intro_samples)
        else:
            number_of_intro_samples = num_samples
            max_experiments = 0

        cfg: Dict[str, Any] = {**evaluation_config.EVALUATION_CONFIG}
        cfg.update({
            "dataset_paths": sorted(paths),
            "num_iterations": 1,
            "max_workers": 5,
            "levels": [level],
            "model_names": [target_llm_model],
            "interactive_config": InteractiveConfig(
                mode=dynamic,
                max_experiments=max_experiments,
                number_of_intro_samples=number_of_intro_samples,
                oracle_config=OracleConfig(
                    llm_model=target_llm_model,
                ),
            ),
        })
        generated_configs.append(cfg)

    print(f"📦 Generated {len(grouped)} in-memory config(s).")
    return generated_configs


__all__ = ["build_main_eval_configs"]
