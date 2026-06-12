"""Abstract oracle that combines multiple oracle workers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from data_management.utils import get_output_label_names

from interactive.oracles.oracle_workers import OracleWorker


class CombinationOracle(OracleWorker, ABC):
    """Wrapper oracle that delegates to multiple other oracle instances.

    This abstract class is initialized with a list of existing oracle workers and
    exposes the union of their callable functions. Subclasses should override
    :meth:`get_callable_functions` and can call ``super()`` to surface all wrapped
    oracle tools unchanged or extend them with additional behaviors.
    """

    def __init__(
        self,
        oracles: List[OracleWorker],
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        if not oracles:
            raise ValueError("CombinationOracle requires at least one oracle instance.")

        self._oracles = list(oracles)
        self.dataset_instance = self._oracles[0].dataset_instance
        self.llm = self._oracles[0].llm
        self._experiment_counter = 0
        self.verbose = verbose
        self.extra_kwargs: Dict[str, Any] = kwargs
        self.label_names = get_output_label_names(self.dataset_instance)

    @property
    def oracles(self) -> List[OracleWorker]:
        """Return the wrapped oracle instances."""

        return self._oracles

    @property
    def experiment_counter(self) -> int:
        """Return the total number of experiments executed by child oracles."""

        counters = [
            getattr(oracle, "experiment_counter", None) for oracle in self._oracles
        ]
        numeric_counters = [count for count in counters if isinstance(count, int)]
        if numeric_counters:
            return sum(numeric_counters)
        return self._experiment_counter

    @experiment_counter.setter
    def experiment_counter(self, value: int) -> None:
        self._experiment_counter = int(value) if value is not None else 0

    @abstractmethod
    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """Return tool callables from all wrapped oracles.

        Subclasses can override this to modify or filter the exposed tools while
        reusing the default aggregation behavior via ``super()``.
        """

        combined: List[Callable[..., Any]] = []
        for oracle in self._oracles:
            combined.extend(oracle.get_callable_functions())
        return combined

    def get_session_data(self) -> Dict[str, Any]:
        """Merge session data collected by each wrapped oracle."""

        return {
            "oracles": [
                {
                    "type": oracle.__class__.__name__,
                    "session_data": oracle.get_session_data(),
                }
                for oracle in self._oracles
            ]
        }
