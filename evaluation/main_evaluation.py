#!/usr/bin/env python
"""
Discover paper datasets that match a target evaluation configuration.

The script searches for ``paper_dataset_instance.json`` files beneath a root
directory, parses their metadata, and filters the results against the target
criteria defined in a config dictionary (see ``configs.main_evaluation_config``).
Matching datasets are additionally checked for existing evaluation results so
completed runs can be reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import re

import importlib
import importlib.util

from configs.interactive_config import InteractiveMode
from data_management.dataset import RepresentationLevel
from evaluation.build_main_eval_configs import build_main_eval_configs

CONFIG_FILENAME = "paper_dataset_instance.json"


@dataclass
class DatasetConfig:
    """Lightweight view of a saved ``paper_dataset_instance.json``."""

    path: Path
    udd_complexity: int
    num_samples: int
    textualization_level: Optional[RepresentationLevel]
    dynamic: Optional[InteractiveMode]
    raw: Dict[str, Any]

    @property
    def dataset_root(self) -> Path:
        """Returns the dataset folder containing the config file."""
        return self.path.parent


def _load_env_if_available() -> None:
    """Load environment variables from .env when python-dotenv is installed."""
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        return
    load_dotenv = importlib.import_module("dotenv").load_dotenv
    load_dotenv()


_load_env_if_available()


def _parse_textualization_level(value: Any) -> Optional[RepresentationLevel]:
    if value is None:
        return None
    parsed = RepresentationLevel.parse(value)
    return None if parsed == RepresentationLevel.UNKNOWN else parsed


def _parse_dynamic(value: Any) -> Optional[InteractiveMode]:
    if isinstance(value, InteractiveMode):
        return value
    if isinstance(value, str):
        try:
            return InteractiveMode(value)
        except ValueError:
            # Support name-based lookup as a fallback.
            try:
                return InteractiveMode[value]
            except Exception:
                return None
    return None


def _normalize_target_levels(levels: Iterable[Any]) -> List[Optional[RepresentationLevel]]:
    normalized: List[Optional[RepresentationLevel]] = []
    for level in levels:
        normalized_level = _parse_textualization_level(level)
        if level is None and normalized_level is None:
            normalized.append(None)
        elif normalized_level is not None:
            normalized.append(normalized_level)
    return normalized


def _normalize_target_dynamics(dynamics: Iterable[Any]) -> List[InteractiveMode]:
    normalized: List[InteractiveMode] = []
    for dyn in dynamics:
        parsed = _parse_dynamic(dyn)
        if parsed is not None:
            normalized.append(parsed)
    return normalized


def _slug(value: str) -> str:
    """Match dataset slugging for eval result paths."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


def _normalize_model_names(target_llm: Any) -> List[str]:
    if target_llm is None:
        return []
    names = []
    name = getattr(target_llm, "name", None)
    value = getattr(target_llm, "value", None)
    if name:
        names.append(name)
    if value and value not in names:
        names.append(str(value))
    if not names:
        names.append(str(target_llm))
    return names


def _normalize_dynamic_names(dynamic: Optional[InteractiveMode]) -> List[str]:
    if dynamic is None:
        return []
    names = []
    if isinstance(dynamic, InteractiveMode):
        names.append(dynamic.name)
        if dynamic.value not in names:
            names.append(dynamic.value)
    else:
        names.append(str(dynamic))
    return names


def _normalize_level_values(level: Optional[RepresentationLevel]) -> List[str]:
    if level is None:
        return [RepresentationLevel.RAW.value, RepresentationLevel.UNKNOWN.value]
    if isinstance(level, RepresentationLevel):
        return [level.value]
    parsed = _parse_textualization_level(level)
    if parsed is None:
        return [RepresentationLevel.UNKNOWN.value]
    return [parsed.value]


def _has_existing_eval_results(
    dataset_root: Path,
    *,
    model_names: Iterable[str],
    interactive_names: Iterable[str],
    level_values: Iterable[str],
) -> bool:
    eval_root = dataset_root / "evaluation_results"
    if not eval_root.exists():
        return False

    model_dirs = [eval_root / _slug(name) for name in model_names]
    for model_dir in model_dirs:
        if not model_dir.exists():
            continue

        # New layout: evaluation_results/<model>/<interactive>/<level>/<instance>/*.json
        for interactive_name in interactive_names:
            interactive_dir = model_dir / _slug(interactive_name)
            for level_value in level_values:
                level_dir = interactive_dir / _slug(level_value)
                if level_dir.exists() and any(level_dir.rglob("*.json")):
                    return True

        # Legacy layout: evaluation_results/<model>/<level>/<instance>/*.json
        for level_value in level_values:
            level_dir = model_dir / _slug(level_value)
            if level_dir.exists() and any(level_dir.rglob("*.json")):
                return True

        # Older legacy layout: evaluation_results/<model>/<instance>/*.json
        if any(model_dir.rglob("*.json")):
            return True

    return False


def _load_dataset_config(config_path: Path) -> Optional[DatasetConfig]:
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"⚠️  Failed to read {config_path}: {exc}")
        return None

    textualization_level = _parse_textualization_level(data.get("textualization_level"))
    dynamic = _parse_dynamic(data.get("dynamic"))

    try:
        complexity = int(data["udd_complexity"])
        num_samples = int(data["num_samples"])
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"⚠️  Invalid numeric fields in {config_path}: {exc}")
        return None

    return DatasetConfig(
        path=config_path,
        udd_complexity=complexity,
        num_samples=num_samples,
        textualization_level=textualization_level,
        dynamic=dynamic,
        raw=data,
    )


def discover_dataset_configs(root: Path, filename: str = CONFIG_FILENAME) -> List[DatasetConfig]:
    """Recursively locate and parse dataset configs under ``root``."""
    if not root.exists():
        print(f"❌ Dataset root '{root}' does not exist.")
        return []

    configs: List[DatasetConfig] = []
    for path in root.rglob(filename):
        parsed = _load_dataset_config(path)
        if parsed:
            configs.append(parsed)
    print(f"🔍 Found {len(configs)} dataset config(s) under {root}.")
    return configs


def filter_matching_datasets(
    datasets: List[DatasetConfig],
    target_textualization_levels: Iterable[Any],
    target_dynamics_levels: Iterable[Any],
    target_complexity_levels: Iterable[int],
) -> Tuple[List[DatasetConfig], List[Tuple[DatasetConfig, List[str]]]]:
    """Partition datasets into matches and non-matches with reasons."""
    allowed_levels = _normalize_target_levels(target_textualization_levels)
    allowed_dynamics = _normalize_target_dynamics(target_dynamics_levels)
    allowed_complexities = {int(level) for level in target_complexity_levels}

    matches: List[DatasetConfig] = []
    rejections: List[Tuple[DatasetConfig, List[str]]] = []

    for ds in datasets:
        reasons: List[str] = []
        if allowed_levels and ds.textualization_level not in allowed_levels:
            reasons.append(f"textualization_level={ds.textualization_level}")
        if allowed_dynamics and ds.dynamic not in allowed_dynamics:
            reasons.append(f"dynamic={ds.dynamic}")
        if allowed_complexities and ds.udd_complexity not in allowed_complexities:
            reasons.append(f"udd_complexity={ds.udd_complexity}")

        if reasons:
            rejections.append((ds, reasons))
        else:
            matches.append(ds)
    return matches, rejections


def run_main_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point used by ``main_evaluation.py``."""
    dataset_root = Path(config.get("dataset_root", "../paper"))
    datasets = discover_dataset_configs(dataset_root)

    matches, rejections = filter_matching_datasets(
        datasets=datasets,
        target_textualization_levels=config.get("target_textualization_levels", []),
        target_dynamics_levels=config.get("target_dynamics_levels", []),
        target_complexity_levels=config.get("target_complexity_levels", []),
    )

    target_llm = config.get("target_llm_model")
    print("\n🎯 Target evaluation configuration:")
    print(f"   LLM model: {target_llm}")
    print(f"   Allowed textualization levels: {config.get('target_textualization_levels', [])}")
    print(f"   Allowed dynamics: {config.get('target_dynamics_levels', [])}")
    print(f"   Allowed complexity levels: {config.get('target_complexity_levels', [])}")

    model_names = _normalize_model_names(target_llm)
    pending: List[DatasetConfig] = []
    completed: List[DatasetConfig] = []

    for ds in matches:
        level_values = _normalize_level_values(ds.textualization_level)
        dynamic_names = _normalize_dynamic_names(ds.dynamic)
        if _has_existing_eval_results(
            ds.dataset_root,
            model_names=model_names,
            interactive_names=dynamic_names,
            level_values=level_values,
        ):
            completed.append(ds)
        else:
            pending.append(ds)

    generated_configs: List[Dict[str, Any]] = []
    if target_llm is None:
        print("⚠️  No target LLM model configured; skipping config generation.")
    elif pending:
        generated_configs = build_main_eval_configs(pending, target_llm_model=target_llm)

    print("\n✅ Matching datasets (pending evaluation):")
    for ds in pending:
        print(f" - {ds.dataset_root} (complexity={ds.udd_complexity}, "
              f"textualization={ds.textualization_level}, dynamic={ds.dynamic}, "
              f"samples={ds.num_samples})")

    if completed:
        print("\n📦 Matching datasets already evaluated:")
        for ds in completed:
            print(f" - {ds.dataset_root} (complexity={ds.udd_complexity}, "
                  f"textualization={ds.textualization_level}, dynamic={ds.dynamic}, "
                  f"samples={ds.num_samples})")

    if rejections:
        print("\n🚫 Non-matching datasets:")
        for ds, reasons in rejections:
            formatted = "; ".join(reasons)
            print(f" - {ds.dataset_root} [rejected: {formatted}]")

    print(
        f"\nSummary: {len(matches)} matched "
        f"({len(completed)} complete, {len(pending)} pending), "
        f"{len(rejections)} rejected."
    )
    return {
        "matches": matches,
        "pending": pending,
        "completed": completed,
        "rejections": rejections,
        "target_llm_model": target_llm,
        "generated_configs": generated_configs,
    }


if __name__ == "__main__":
    # Running directly without the wrapper defaults to the base config.
    from configs.main_evaluation_config import MAIN_EVALUATION_CONFIG

    run_main_evaluation(MAIN_EVALUATION_CONFIG)
