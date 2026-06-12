from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_if_available() -> None:
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        print("[benchmark_backend_model] python-dotenv not installed; .env was not loaded.")
        print(
            "[benchmark_backend_model] GOOGLE_API_KEY present: "
            f"{bool(os.getenv('GOOGLE_API_KEY'))}"
        )
        return

    load_dotenv = importlib.import_module("dotenv").load_dotenv
    env_path = PROJECT_ROOT / ".env"
    loaded = load_dotenv(env_path)
    print(
        "[benchmark_backend_model] .env loaded: "
        f"{loaded} from {env_path}"
    )
    print(
        "[benchmark_backend_model] GOOGLE_API_KEY present: "
        f"{bool(os.getenv('GOOGLE_API_KEY'))}"
    )


_load_env_if_available()

from configs.interactive_config import InteractiveConfig, InteractiveMode, OracleConfig
from data_management.dataset import RepresentationLevel
from inference import LLMBackend, LLMModel
from server_interface.host.server import REMOTE_HTTP_EVAL_DATASET_PATH


BACKEND_MODEL_ENV_VAR = "REMOTE_EVAL_BACKEND_MODEL"
DEFAULT_BACKEND_MODEL = LLMModel.GEMINI_3_FLASH_PREVIEW
RUN_BENCHMARK_ENV_VAR = "RUN_BACKEND_MODEL_BENCHMARK"


def resolve_backend_model() -> LLMModel:
    configured = os.getenv(BACKEND_MODEL_ENV_VAR)
    if not configured:
        return DEFAULT_BACKEND_MODEL

    normalized = configured.strip()
    for model in LLMModel:
        if normalized in {
            model.name,
            model.native_id,
            model.litellm_id,
            str(model),
        }:
            return model

    valid_names = ", ".join(model.name for model in LLMModel)
    raise ValueError(
        f"Unknown {BACKEND_MODEL_ENV_VAR}={configured!r}. Use one of: {valid_names}"
    )


BENCHMARK_MODEL = resolve_backend_model()


def backend_api_key_is_available(model: LLMModel) -> bool:
    if model.name.startswith("GEMINI"):
        return bool(os.getenv("GOOGLE_API_KEY"))
    if model.name.startswith("GPT"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if model.name.startswith("CLAUDE"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if model.name.startswith("GROK"):
        return bool(os.getenv("XAI_API_KEY"))
    if model.name.startswith("DEEPSEEK"):
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    return True


def build_backend_model_evaluation_configs() -> list[dict[str, Any]]:
    """Match the remote evaluation flow config, but use the selected model directly."""

    return [
        {
            "dataset_paths": [str(REMOTE_HTTP_EVAL_DATASET_PATH)],
            "instance_names": ["raw_samples"],
            "model_names": [BENCHMARK_MODEL],
            "levels": [RepresentationLevel.RAW],
            "num_iterations": 1,
            "return_per_sample": False,
            "max_workers": 1,
            "timeout_per_task": None,
            "debug": None,
            "interactive_config": InteractiveConfig(
                mode=InteractiveMode.STATIC,
                verbose=False,
                max_experiments=0,
                number_of_intro_samples=10,
                oracle_config=OracleConfig(
                    llm_model=BENCHMARK_MODEL,
                    backend=LLMBackend.NATIVE,
                    verbose=False,
                    kwargs={},
                ),
            ),
        }
    ]


def run_benchmark(verbose: bool = False) -> dict[str, Any]:
    from evaluation.textual_llm_evaluation import evaluate_parallel

    started_at = time.perf_counter()
    result = evaluate_parallel(
        build_backend_model_evaluation_configs(),
        max_workers=1,
        verbose=verbose,
    )
    elapsed_seconds = time.perf_counter() - started_at

    successes = result.get("successes", [])
    failures = result.get("failures", [])
    first_success = successes[0] if successes else {}
    first_failure = failures[0] if failures else {}
    report = first_success.get("report") or {}

    return {
        "model": BENCHMARK_MODEL.name,
        "model_native_id": BENCHMARK_MODEL.native_id,
        "dataset_path": str(REMOTE_HTTP_EVAL_DATASET_PATH),
        "successes": len(successes),
        "failures": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "task_total_seconds": (first_success.get("timings") or {}).get("total_seconds"),
        "accuracy": report.get("accuracy"),
        "full_accuracy": report.get("full_accuracy"),
        "first_failure": first_failure.get("error"),
    }


@unittest.skipUnless(
    os.getenv(RUN_BENCHMARK_ENV_VAR) == "1"
    and backend_api_key_is_available(BENCHMARK_MODEL)
    and importlib.util.find_spec("sklearn") is not None,
    f"Set {RUN_BENCHMARK_ENV_VAR}=1, {BACKEND_MODEL_ENV_VAR} if needed, and the backend model API key, with sklearn installed, to run this benchmark.",
)
class TestBackendModelBenchmark(unittest.TestCase):
    def test_direct_backend_model_benchmark(self):
        summary = run_benchmark(verbose=False)
        print("\n[benchmark_backend_model] benchmark summary")
        print(json.dumps(summary, indent=2, default=str))

        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["failures"], 0)


if __name__ == "__main__":
    if os.getenv(RUN_BENCHMARK_ENV_VAR) == "1":
        unittest.main()
    else:
        summary = run_benchmark(verbose=True)
        print("\n[benchmark_backend_model] benchmark summary")
        print(json.dumps(summary, indent=2, default=str))
