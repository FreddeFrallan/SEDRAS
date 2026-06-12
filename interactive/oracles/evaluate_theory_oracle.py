"""Oracle worker that evaluates textual theories on the dataset."""
from __future__ import annotations

from typing import Any, Dict, List

from data_management.utils import _infer_variable_order, _schema_from_dataset
from evaluation.backbone.compile_and_evaluate_textual_theory import (
    compile_and_evaluate_textual_theory,
)
from interactive.oracles.oracle_workers import OracleWorker


class EvaluateTheoryOracle(OracleWorker):
    """
    Oracle worker exposing a single tool:

      - evaluate_theory(theory_text: str) -> float

    The tool compiles the provided textual theory into a classifier using the
    existing compilation pipeline and evaluates it against every sample in the
    underlying dataset distribution (UDD). It returns the aggregate accuracy
    over all outputs.
    """

    def __init__(self, dataset_instance, llm_model, verbose: bool = False, **kwargs: Any):
        super().__init__(dataset_instance, llm_model, verbose=verbose, **kwargs)

        self.variable_order: List[str] = _infer_variable_order(dataset_instance)
        self.schema: Dict[str, Dict[str, Any]] = _schema_from_dataset(
            dataset_instance, self.variable_order
        )
        self._theory_evaluation_history: List[Dict[str, Any]] = []

    def evaluate_theory(self, theory_text: str) -> float:
        """
        Compile and evaluate a textual theory on all available samples.

        Args:
            theory_text: Natural-language description of the decision rule.

        Returns:
            The total accuracy over the dataset as a floating point value.
        """

        if self.verbose:
            print("[EvaluateTheoryOracle] Evaluating provided textual theory.")

        _, report, _ = compile_and_evaluate_textual_theory(
            theory_txt=theory_text,
            variable_order=self.variable_order,
            schema=self.schema,
            dataset_samples=self.dataset_instance.raw_samples,
            llm=self.llm,
            num_output_labels=int(getattr(self.dataset_instance.udd, "num_output_labels", 2)),
            label_names=self.label_names,
            output_properties=getattr(self.dataset_instance, "output_properties", None),
            return_per_sample=False,
        )

        accuracy = float(report.get("accuracy", 0.0))
        self._theory_evaluation_history.append(
            {
                "theory_text": theory_text,
                "theory_code": report.get("classifier_code", ""),
                "theory_accuracy": accuracy,
            }
        )

        return accuracy

    def get_callable_functions(self):
        return [self.evaluate_theory]

    def get_session_data(self) -> Dict[str, Any]:
        session_data = super().get_session_data()
        session_data["theory_evaluations"] = list(self._theory_evaluation_history)
        return session_data
