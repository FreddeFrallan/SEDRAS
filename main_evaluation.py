#!/usr/bin/env python
"""
Entry point for paper dataset evaluation discovery.

Usage:
    python main_evaluation.py SEDRAS_2026
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from configs.main_evaluation_config import MAIN_EVALUATION_CONFIG
from configs.instances.SEDRAS_2026MainEvaluation import SEDRAS_2026_MAIN_EVALUATION_CONFIG
from evaluation.textual_llm_evaluation import evaluate_parallel
from evaluation.main_evaluation import run_main_evaluation, _load_env_if_available

_load_env_if_available()

AVAILABLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "SEDRAS_2026": SEDRAS_2026_MAIN_EVALUATION_CONFIG,
    "Base": MAIN_EVALUATION_CONFIG,
}


def _load_config(name: str) -> Dict[str, Any]:
    if name not in AVAILABLE_CONFIGS:
        available = ", ".join(sorted(AVAILABLE_CONFIGS.keys()))
        print(f"Unknown config '{name}'.")
        print(f"Available options: {available}")
        sys.exit(1)
    return AVAILABLE_CONFIGS[name]


def main() -> None:
    if len(sys.argv) != 2:
        available = ", ".join(sorted(AVAILABLE_CONFIGS.keys()))
        print("Usage: python main_evaluation.py <PersonName>")
        print(f"Available PersonName options: {available}")
        sys.exit(1)

    person_name = sys.argv[1]
    config = _load_config(person_name)

    print(f"🚀 Running main evaluation discovery for {person_name}")
    summary = run_main_evaluation(config)
    generated_configs = summary.get("generated_configs", [])
    if not generated_configs:
        print("⚠️  No evaluation configs generated; skipping evaluation.")
        return

    print(f"\n🧪 Prepared {len(generated_configs)} in-memory evaluation job(s).")
    response = input("Proceed with evaluation? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        print("🛑 Evaluation aborted by user.")
        return

    evaluate_parallel(generated_configs, max_workers=config["max_workers"])


if __name__ == "__main__":
    main()
