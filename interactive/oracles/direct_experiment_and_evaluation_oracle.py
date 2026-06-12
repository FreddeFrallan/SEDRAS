"""Combined oracle for running experiments and evaluating theories."""
from __future__ import annotations

from typing import Any, Callable, List

from interactive.oracles.combination_oracle import CombinationOracle
from interactive.oracles.direct_experiment_worker import DirectExperimentWorker
from interactive.oracles.evaluate_theory_oracle import EvaluateTheoryOracle
from interactive.oracles.textual_direct_experiment_worker import TextualDirectExperimentWorker


class DirectExperimentAndEvaluationOracle(CombinationOracle):
    """Expose both experiment execution and theory evaluation tools.

    This oracle wraps a :class:`DirectExperimentWorker` together with an
    :class:`EvaluateTheoryOracle`, surfacing both of their callable tools. It
    leverages :class:`CombinationOracle` to delegate tool execution to the
    appropriate underlying worker while presenting them as a single oracle to
    the interactive loop.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        *,
        verbose: bool = False,
        max_experiments: int | None = None,
        **kwargs: Any,
    ) -> None:
        direct_oracle = DirectExperimentWorker(
            dataset_instance,
            llm_model,
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )
        evaluate_oracle = EvaluateTheoryOracle(
            dataset_instance,
            llm_model,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(
            [direct_oracle, evaluate_oracle],
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        return super().get_callable_functions()


class TextualDirectExperimentAndEvaluationOracle(CombinationOracle):
    """Expose both experiment execution and theory evaluation tools.

    This oracle wraps a :class:`DirectExperimentWorker` together with an
    :class:`EvaluateTheoryOracle`, surfacing both of their callable tools. It
    leverages :class:`CombinationOracle` to delegate tool execution to the
    appropriate underlying worker while presenting them as a single oracle to
    the interactive loop.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        *,
        verbose: bool = False,
        max_experiments: int | None = None,
        **kwargs: Any,
    ) -> None:
        direct_oracle = TextualDirectExperimentWorker(
            dataset_instance,
            llm_model,
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )
        evaluate_oracle = EvaluateTheoryOracle(
            dataset_instance,
            llm_model,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(
            [direct_oracle, evaluate_oracle],
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        return super().get_callable_functions()