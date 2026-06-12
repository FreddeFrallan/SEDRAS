# oracle_workers.py
from __future__ import annotations

from typing import Dict, Any, List, Callable
from abc import ABC, abstractmethod

from inference.model_wrappers import llm_wrapper
from data_management.utils import get_output_label_names


class OracleWorker(ABC):
    """
    Abstract base class for oracle-style workers.

    Responsibilities:
      - Hold a reference to the DatasetInstance and an LLM wrapper.
      - Optionally store extra configuration via **kwargs.
      - Expose a list of callable functions that can be presented as tools
        to an LLM via `get_callable_functions`.

    Subclasses should implement concrete tool functions (e.g.,
    `get_variable_mapping`, `run_experiment`, retrieval/query methods, etc.)
    and return them from `get_callable_functions`.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model: llm_wrapper.LLMModel,
        *,
        backend: llm_wrapper.LLMBackend | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            dataset_instance: A DatasetInstance with an attached UDD as `dataset_instance.udd`.
            llm_model: The LLMModel enum to use for any internal LLM calls.
            verbose: If True, print debug information during tool execution.
            **kwargs: Extra configuration or options that subclasses may use.
        """
        self.dataset_instance = dataset_instance
        # Backend selection (native vs LiteLLM) is handled internally by get_llm_wrapper,
        # using env var LLM_BACKEND unless callers explicitly pass a backend.
        self.llm = llm_wrapper.get_llm_wrapper(llm_model, backend=backend)
        self.experiment_counter = 0
        self.verbose = verbose
        # Store any extra options for future use by subclasses
        self.extra_kwargs: Dict[str, Any] = kwargs
        self.label_names = get_output_label_names(dataset_instance)

    @abstractmethod
    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """
        Return the list of callable functions that should be exposed as tools
        to the LLM. Each callable should have a proper docstring and type
        annotations so that tool definitions can be generated automatically.
        """
        raise NotImplementedError

    def get_session_data(self) -> Dict[str, Any]:
        """
        Return any JSON-serializable session metadata that should be persisted
        once the interactive session finishes.

        Subclasses can override this to expose additional information (e.g.,
        experiment traces). By default, no extra metadata is returned.
        """
        return {}

    def get_session_settings(self) -> Dict[str, Any]:
        """Return per-session settings for the interactive loop."""

        return {"end_on_no_tool_calls": True}
