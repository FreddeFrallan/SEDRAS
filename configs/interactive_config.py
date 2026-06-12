from inference.enums import LLMModel, LLMBackend
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
import json


class InteractiveMode(Enum):
    STATIC = "static"
    DIRECT_EXPERIMENT = "direct_experiment"
    DIRECT_EXPERIMENT_AND_EXPLICIT_FINISH = "direct_experiment_and_explicit_finish"
    DIRECT_EXPERIMENT_AND_EVALUATION = "direct_experiment_and_evaluation"
    SINGLE_CONVERSATIONAL = "single_conversational"
    RETRIEVAL_CONVERSATIONAL = "retrieval_conversational"
    SYMBOLIC_REGRESSION = "symbolic_regression"


@dataclass
class OracleConfig:
    """
    Configuration object describing how to initialize an OracleWorker.

    Attributes:
        oracle_type: Which concrete oracle implementation to use.
        llm_model:   The LLM model used internally by the oracle worker.
        backend:     Which backend implementation to use (native vs litellm).
        verbose:     If True, the worker may print debug / trace info.
        kwargs:      Additional keyword arguments forwarded to the worker
                     constructor (e.g., `llm_system_prompt` for
                     ConversationalRetrievalWorker).
    """

    llm_model: LLMModel = LLMModel.GEMINI_2_5_FLASH
    backend: LLMBackend = LLMBackend.NATIVE
    verbose: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""

        def _enum_or_value(x: Any) -> Any:
            # If it's an Enum, use its name for compactness
            return getattr(x, "name", x)

        return {
            "llm_model": _enum_or_value(self.llm_model),
            "backend": _enum_or_value(self.backend),
            "verbose": self.verbose,
            "kwargs": self.kwargs,
        }

    def __str__(self) -> str:
        """Human-readable string representation, e.g. for logs."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class InteractiveConfig:
    mode: InteractiveMode = InteractiveMode.STATIC
    oracle_config: OracleConfig = None

    number_of_oracle_samples: int = 20
    max_experiments: int = 50
    minimum_number_of_experiments: Optional[int] = None
    number_of_intro_samples: int = 10
    max_retrievals_per_question: int = 3
    max_turns: int = 20
    verbose: bool = False


    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""

        def _enum_or_value(x: Any) -> Any:
            # If it's an Enum, use its name for compactness
            return getattr(x, "name", x)

        return {
            "mode": _enum_or_value(self.mode),
            "oracle_config": self.oracle_config.to_dict() if self.oracle_config else None,
            "max_experiments": self.max_experiments,
            "minimum_number_of_experiments": self.minimum_number_of_experiments,
            "max_turns": self.max_turns,
            "number_of_intro_samples": self.number_of_intro_samples,
            "verbose": self.verbose,
        }
