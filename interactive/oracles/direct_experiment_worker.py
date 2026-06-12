# oracle_workers.py
from __future__ import annotations

from typing import Dict, Any, List, Callable, Optional
import numpy as np

from inference.model_wrappers import llm_wrapper
from interactive.oracles.oracle_workers import OracleWorker

from prompts.task_descriptions.utils import (
    _get_property_label_names,
    _render_from_assignment,
    _render_label_and_properties,
    _render_raw_samples,
)
from data_management.creation.create_balanced_dataset import _label_from_scores
from types import SimpleNamespace


class DirectExperimentWorker(OracleWorker):
    """
    Concrete oracle worker that exposes one tool:

      - run_experiment(assignment: dict[str, int]) -> dict

    It keeps an internal experiment counter.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model: llm_wrapper.LLMModel,
        verbose: bool = False,
        **kwargs: Any,
    ):
        # Let the base class store dataset_instance, llm, verbose, extra_kwargs
        super().__init__(dataset_instance, llm_model, verbose=verbose, **kwargs)

        # We now store dicts like {"input_values": {...}, "label": int | None}
        self._assignment_history: List[Dict[str, Any]] = []

    def _json_safe_assignment(self, assignment: dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable copy of the assignment."""

        def _convert(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            return value

        if not isinstance(assignment, dict):
            return {"value": _convert(assignment)}

        return {str(k): _convert(v) for k, v in assignment.items()}

    def _record_assignment(
        self,
        assignment: dict[str, Any],
        label: Optional[int],
        property_labels: Optional[Dict[str, Optional[int]]] = None,
    ) -> None:
        """
        Record one experiment as:
            {
              "input_values": { "V0": ..., "V1": ..., ... },
              "label": 0 | 1 | None
            }
        """
        json_safe = self._json_safe_assignment(assignment)
        label_name = None if label is None else self.label_names.get(label, f"Label {label}")
        self._assignment_history.append(
            {
                "input_values": json_safe,
                "label": label,
                "label_name": label_name,
                "property_labels": property_labels or {},
            }
        )

    def _evaluate_output_properties(
        self,
        assignment: Dict[str, int],
        *,
        label: int,
        sample_labels: bool,
    ) -> Dict[str, Optional[int]]:
        """
        Evaluate any additional output properties tied to this dataset instance.

        Properties are only computed when the current main label is supported
        for that property (based on property_label_support).
        """

        property_labels: Dict[str, Optional[int]] = {}
        prop_label_support = getattr(self.dataset_instance, "property_label_support", {}) or {}
        for prop_name, prop_udd in getattr(self.dataset_instance, "output_properties", {}).items():
            supported_labels = set(prop_label_support.get(prop_name, []))
            if supported_labels and label not in supported_labels:
                property_labels[prop_name] = None
                continue

            prop_res = prop_udd.evaluate(assignment)
            prop_norm_scores = prop_res.get(
                "normalized_scores",
                {prop_res.get("predicted_label", 0): prop_res.get("normalized_score", 0.0)},
            )
            prop_label = _label_from_scores(prop_norm_scores, sample_labels=sample_labels)
            property_labels[prop_name] = prop_label

        return property_labels

    def _format_result_sample(
        self,
        *,
        assignment: Dict[str, int],
        label: int,
        property_labels: Dict[str, Optional[int]],
    ) -> Dict[str, Any]:
        """Render a human-friendly sample string aligned with intro examples."""

        tm = self.dataset_instance.text_mapping or {}
        tm_labels = tm.get("output_labels", {}) if tm else {}
        use_textual_only = isinstance(tm_labels, dict) and len(tm_labels) > 0
        property_label_names = _get_property_label_names(self.dataset_instance)

        text = _render_from_assignment(tm, assignment) if tm else None
        if not text:
            text = _render_raw_samples(SimpleNamespace(assignment=assignment))
        text = (text or "").replace("\n", " ")

        label_block = _render_label_and_properties(
            label_idx=label,
            label_names=self.label_names,
            property_labels=property_labels,
            property_label_names=property_label_names,
            use_textual_only=use_textual_only,
        )

        return {
            "rendered_sample": f"Labels: {label_block} | {text if text else '(no text)'}",
            "property_label_names": property_label_names,
        }

    def run_experiment(self, assignment: dict[str, int]) -> dict:
        """
        Run an experiment with a variable assignment and return outcome info as a JSON-safe dict.

        Expected 'assignment' shape: { "V0": int, "V1": int, ... } matching dataset categories.

        """

        self.experiment_counter += 1
        if self.verbose:
            print(f"Running experiment #{self.experiment_counter} with assignment: {assignment}")

        try:
            res = self.dataset_instance.udd.evaluate(assignment)
            normalized_scores = res.get(
                "normalized_scores",
                {res.get("predicted_label", 0): res.get("normalized_score", 0.0)},
            )
            sample_labels = bool(self.dataset_instance.metadata.get("sample_labels", False))
            label = _label_from_scores(normalized_scores, sample_labels=sample_labels)
            score = float(normalized_scores.get(label, 0.0))
            label_name = self.label_names.get(label, f"Label {label}")

            property_labels = self._evaluate_output_properties(
                assignment, label=label, sample_labels=sample_labels
            )
            formatted_sample = self._format_result_sample(
                assignment=assignment,
                label=label,
                property_labels=property_labels,
            )

            if self.verbose:
                print(
                    f"Experiment #{self.experiment_counter} result: "
                    f"score={score}, label={label_name} (id={label}), "
                    f"properties={property_labels}"
                )

            # Record with both inputs and label
            self._record_assignment(assignment, label, property_labels)

            return {
                "experiment status": "Success",
                "label": label,
                "label_name": label_name,
                "property_labels": property_labels,
                **formatted_sample,
                "score": score,
                "experiment_index": self.experiment_counter,
            }
        except Exception as e:
            if self.verbose:
                print(f"Experiment #{self.experiment_counter} failed with error: {e}")

            # Record the attempted assignment with unknown label
            self._record_assignment(assignment, None, None)

            return {
                "experiment status": "Error",
                "message": str(e),
                "label": None,
                "experiment_index": self.experiment_counter,
            }

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """
        Return the list of callables to expose as tools to the LLM.
        """
        return [self.run_experiment]

    def get_session_data(self) -> Dict[str, Any]:
        # Now returns:
        # {
        #   "assignments": [
        #     { "input_values": {...}, "label": 0/1/None },
        #     ...
        #   ]
        # }
        return {"assignments": list(self._assignment_history)}


class DirectExperimentAndExplicitFinish(DirectExperimentWorker):
    """Direct experiment oracle with an explicit submit_theory tool."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._final_theory: Optional[str] = None
        min_exp = kwargs.get("minimum_number_of_experiments")
        self.minimum_number_of_experiments: Optional[int] = (
            int(min_exp) if min_exp is not None else None
        )

    def submit_theory(self, theory: str) -> Dict[str, Any]:
        """Submit a final theory string and signal whether it was accepted."""

        if not isinstance(theory, str):
            return {"accepted": False, "message": "Theory must be provided as a string."}

        if (
            self.minimum_number_of_experiments is not None
            and self.experiment_counter < self.minimum_number_of_experiments
        ):
            if self.verbose:
                print(
                    "Rejecting submit_theory: "
                    f"{self.experiment_counter}/{self.minimum_number_of_experiments} "
                    "experiments completed."
                )

            remaining = self.minimum_number_of_experiments - self.experiment_counter
            return {
                "accepted": False,
                "message": (
                    "Minimum number of experiments not yet reached; "
                    f"run {remaining} more experiment(s) before submitting a final theory."
                ),
                "completed_experiments": self.experiment_counter,
                "required_experiments": self.minimum_number_of_experiments,
            }

        self._final_theory = theory
        return {"accepted": True, "final_theory": theory}

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        return [*super().get_callable_functions(), self.submit_theory]

    def get_session_data(self) -> Dict[str, Any]:
        session_data = super().get_session_data()
        session_data["final_theory"] = self._final_theory
        return session_data

    def get_session_settings(self) -> Dict[str, Any]:
        return {
            "end_on_no_tool_calls": False,
            "no_tool_calls_continue_message": f"Let's continue until more experiments are conducted. You must perform at least {self.minimum_number_of_experiments} experiments before submitting a final theory. Then use the submit_theory tool to finish.",
        }
