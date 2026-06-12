# init_oracle.py
from __future__ import annotations

import random
from typing import List, Any, Tuple

from configs.interactive_config import InteractiveMode, InteractiveConfig
from data_management.dataset import RepresentationLevel
from inference.enums import LLMBackend

from interactive.oracles.oracle_workers import OracleWorker
from interactive.oracles.direct_experiment_worker import DirectExperimentWorker, DirectExperimentAndExplicitFinish
from interactive.oracles.direct_experiment_and_evaluation_oracle import (
    DirectExperimentAndEvaluationOracle,
    TextualDirectExperimentAndEvaluationOracle,
)
from interactive.oracles.evaluate_theory_oracle import EvaluateTheoryOracle
from interactive.oracles.symbolic_regression_oracle import (
    SymbolicRegressionOracle,
    TextualSymbolicRegressionOracle,
)
from interactive.oracles.conversational_worker import ConversationalWorker
from interactive.oracles.retrieval_conversational_worker import (
    RetrievalConversationalWorker,
)
from prompts.oracle_workers import (
    build_retrieval_conversational_system_prompt,
    build_sample_augmented_system_prompt,
)
from data_management.utils import get_output_label_names


def _select_oracle_samples(
    dataset_instance,
    requested_samples: int,
    *,
    rng_seed: int = 42,
) -> List[Tuple[int, Any]]:
    """
    Select a deterministic subset of samples from the dataset instance.

    Returns:
        A list of (index, sample) tuples.

    Raises:
        ValueError: If the dataset has no samples or if more samples are
                    requested than available.
    """
    if not getattr(dataset_instance, "samples", None):
        raise ValueError("Dataset instance does not contain any samples to build a prompt.")

    total = len(dataset_instance.samples)
    if requested_samples > total:
        raise ValueError(
            "Requested more oracle samples than available in the dataset: "
            f"requested={requested_samples}, available={total}"
        )

    if requested_samples <= 0:
        return []

    rng = random.Random(rng_seed)
    selected_indices = rng.sample(range(total), requested_samples)

    # Keep the prompt order deterministic for readability and reproducibility
    selected_indices.sort()

    return [(idx, dataset_instance.samples[idx]) for idx in selected_indices]


def select_backend_for_representation(
    representation_level: RepresentationLevel,
    backend: LLMBackend | None,
) -> LLMBackend | None:
    if representation_level == RepresentationLevel.FILES and backend != LLMBackend.NATIVE:
        return LLMBackend.NATIVE
    return backend


def init_oracle_worker(dataset_instance, interactive_config: InteractiveConfig,
                       llm_model,
                       ) -> OracleWorker:
    """
    Initialize the appropriate OracleWorker subclass based on the given config.
    """
    oracle_config = interactive_config.oracle_config
    representation_level = RepresentationLevel.parse(
        getattr(dataset_instance, "representation_level", RepresentationLevel.UNKNOWN)
    )
    backend = select_backend_for_representation(representation_level, oracle_config.backend)

    if interactive_config.mode == InteractiveMode.DIRECT_EXPERIMENT:
        raise NotImplementedError("DirectExperiment mode is deprecated")
        # Direct experiment oracle
        return DirectExperimentWorker(
            dataset_instance,
            llm_model=oracle_config.llm_model,
            verbose=oracle_config.verbose,
            # --- FIX: Pass max_experiments here ---
            max_experiments=interactive_config.max_experiments,
            **oracle_config.kwargs,
        )

    if (
        interactive_config.mode
        == InteractiveMode.DIRECT_EXPERIMENT_AND_EXPLICIT_FINISH
    ):
        raise NotImplementedError("DirectExperimentAndExplicitFinish mode is deprecated")
        return DirectExperimentAndExplicitFinish(
            dataset_instance,
            llm_model=oracle_config.llm_model,
            verbose=oracle_config.verbose,
            # --- FIX: Pass max_experiments here ---
            max_experiments=interactive_config.max_experiments,
            minimum_number_of_experiments=interactive_config.minimum_number_of_experiments,
            **oracle_config.kwargs,
        )

    if (
        interactive_config.mode
        == InteractiveMode.DIRECT_EXPERIMENT_AND_EVALUATION
    ):
        # Use textual tools whenever the dataset is textualized
        if representation_level != RepresentationLevel.RAW:
            return TextualDirectExperimentAndEvaluationOracle(
                dataset_instance,
                llm_model=llm_model,
                backend=backend,
                verbose=oracle_config.verbose,
                max_experiments=interactive_config.max_experiments,
                **oracle_config.kwargs,
            )
        else:
            return DirectExperimentAndEvaluationOracle(
                dataset_instance,
                # llm_model=oracle_config.llm_model,
                llm_model=llm_model,
                backend=backend,
                verbose=oracle_config.verbose,
                max_experiments=interactive_config.max_experiments,
                **oracle_config.kwargs,
            )

    if interactive_config.mode == InteractiveMode.SYMBOLIC_REGRESSION:
        if representation_level != RepresentationLevel.RAW:
            return TextualSymbolicRegressionOracle(
                dataset_instance,
                # llm_model=oracle_config.llm_model,
                llm_model=llm_model,
                backend=backend,
                verbose=oracle_config.verbose,
                max_experiments=interactive_config.max_experiments,
                **oracle_config.kwargs,
            )
        else:
            return SymbolicRegressionOracle(
                dataset_instance,
                # llm_model=oracle_config.llm_model,
                llm_model=llm_model,
                backend=backend,
                verbose=oracle_config.verbose,
                max_experiments=interactive_config.max_experiments,
                **oracle_config.kwargs,
            )

    # ... (Rest of the function remains the same)
    if interactive_config.mode == InteractiveMode.SINGLE_CONVERSATIONAL:
        raise NotImplementedError("SingleConversational mode is deprecated")
        requested_samples = max(0, int(interactive_config.number_of_oracle_samples))

        selected_samples = _select_oracle_samples(
            dataset_instance,
            requested_samples,
        )

        system_prompt_text = build_sample_augmented_system_prompt(
            selected_samples,
            dataset_instance=dataset_instance,
        )

        return ConversationalWorker(
            dataset_instance,
            llm_model=oracle_config.llm_model,
            verbose=oracle_config.verbose,
            llm_system_prompt_text=system_prompt_text,
            **oracle_config.kwargs,
        )

    if interactive_config.mode == InteractiveMode.RETRIEVAL_CONVERSATIONAL:
        raise NotImplementedError("RetrievalConversational mode is deprecated")
        requested_samples = max(0, int(interactive_config.number_of_oracle_samples))

        selected_samples = _select_oracle_samples(
            dataset_instance,
            requested_samples,
        )

        label_names = get_output_label_names(dataset_instance)
        system_prompt_text = build_retrieval_conversational_system_prompt(
            dataset_instance,
            interactive_config,
            selected_samples,
            label_names=label_names,
        )

        return RetrievalConversationalWorker(
            dataset_instance,
            llm_model=oracle_config.llm_model,
            verbose=oracle_config.verbose,
            llm_system_prompt_text=system_prompt_text,
            **oracle_config.kwargs,
        )

    raise ValueError(f"Unsupported interactive mode: {interactive_config.mode!r}")
