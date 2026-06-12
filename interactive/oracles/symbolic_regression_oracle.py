"""Oracle worker that derives symbolic numerical and categorical models as Python code."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

from interactive.oracles.combination_oracle import CombinationOracle
from interactive.oracles.direct_experiment_and_evaluation_oracle import (
    DirectExperimentAndEvaluationOracle,
    TextualDirectExperimentAndEvaluationOracle,
)
from interactive.oracles.oracle_workers import OracleWorker


class SymbolicReasoningOracle(OracleWorker):
    """
    Oracle that fits interpretable models over sample lists provided by the agent.

    Exposes two tools:
      - derive_numerical_regression: Returns a Python function for continuous trends.
      - derive_categorical_logic: Returns a Python function with IF/ELSE logic for classes.

    All outputs use raw indices (x0, x1, ...) to maintain mapping with the UDD space.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        *,
        polynomial_degree: int = 2,
        max_tree_depth: int = 3,
        regularization_strength: float = 1e-3,
        verbose: bool = False,
        **kwargs: Any,
    ):
        # super().__init__(dataset_instance, llm_model, verbose=verbose, **kwargs)
        self.polynomial_degree = polynomial_degree
        self.max_tree_depth = max_tree_depth
        self.regularization_strength = regularization_strength

    # ----------------------------
    # Internal Helpers
    # ----------------------------

    def _prepare_data(self, samples: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Extracts features and targets from the raw sample dictionaries."""
        if not samples:
            raise ValueError("No samples provided to the oracle.")

        # Identify numerical keys (indices) based on the first sample
        indices = sorted([k for k in samples[0].keys() if k.isdigit()], key=int)
        feature_names = [f"x{i}" for i in indices]

        X_rows, y_num, y_cat = [], [], []

        for s in samples:
            # Features: map index strings "0", "1" to floats
            X_rows.append([float(s.get(idx, 0.0)) for idx in indices])
            # Targets: score for regression, label for classification
            y_num.append(float(s.get("score", 0.0)))
            y_cat.append(str(s.get("label", "unknown")))

        return np.array(X_rows), np.array(y_num), np.array(y_cat), feature_names

    def _tree_to_python(self, tree, feature_names, classes, node=0, indent="    ") -> str:
        """Recursively converts the sklearn decision tree into a Python IF-ELSE block."""
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]

        # Check if leaf node
        if left_child == right_child:
            class_idx = np.argmax(tree.value[node])
            return f"{indent}return '{classes[class_idx]}'\n"

        # Decision node
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        code = f"{indent}if {feature} <= {threshold:.4f}:\n"
        code += self._tree_to_python(tree, feature_names, classes, left_child, indent + "    ")
        code += f"{indent}else:\n"
        code += self._tree_to_python(tree, feature_names, classes, right_child, indent + "    ")
        return code

    # ----------------------------
    # Tool Functions
    # ----------------------------

    # def derive_numerical_regression(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Fits a polynomial regression and returns a Python function string."""
    #     try:
    #         X, y, _, f_names = self._prepare_data(samples)
    #
    #         poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
    #         X_poly = poly.fit_transform(X)
    #         poly_names = poly.get_feature_names_out(f_names)
    #
    #         model = Ridge(alpha=self.regularization_strength)
    #         model.fit(X_poly, y)
    #
    #         # Build the return statement
    #         terms = [f"({c:.6f} * {n})" for c, n in zip(model.coef_, poly_names) if abs(c) > 1e-6]
    #         equation_str = f"return {model.intercept_:.6f} + " + " + ".join(terms)
    #         python_func = f"def predict(x):\n    # x keys: {', '.join(f_names)}\n    {equation_str}"
    #
    #         return {
    #             "status": "ok",
    #             "python_function": python_func,
    #             "r2_score": float(r2_score(y, model.predict(X_poly))),
    #             "sample_count": len(samples)
    #         }
    #     except Exception as e:
    #         return {"status": "error", "message": str(e)}

    def derive_categorical_logic(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fits a Decision Tree and returns a Python function with IF-ELSE logic."""
        try:
            X, _, y, f_names = self._prepare_data(samples)

            clf = DecisionTreeClassifier(max_depth=self.max_tree_depth)
            clf.fit(X, y)

            classes = [str(c) for c in clf.classes_]
            python_func = "def predict(x):\n"
            python_func += f"    # x keys: {', '.join(f_names)}\n"
            python_func += self._tree_to_python(clf.tree_, f_names, classes)

            return {
                "status": "ok",
                "python_function": python_func,
                "accuracy": float(accuracy_score(y, clf.predict(X))),
                "detected_classes": classes,
                "sample_count": len(samples)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_callable_functions(self):
        return [
            # self.derive_numerical_regression,
                self.derive_categorical_logic]


class SymbolicRegressionOracle(CombinationOracle):
    """
    Oracle that combines direct experimentation, theory evaluation, and
    symbolic reasoning helpers.

    This surfaces the full toolset from direct experimentation plus the
    symbolic regression-specific tools so the agent can both gather data
    and induce equations.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        *,
        verbose: bool = False,
        max_experiments: Optional[int] = None,
        polynomial_degree: int = 2,
        max_tree_depth: int = 3,
        regularization_strength: float = 1e-3,
        **kwargs: Any,
    ):
        direct_and_eval_oracle = DirectExperimentAndEvaluationOracle(
            dataset_instance,
            llm_model,
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )
        symbolic_oracle = SymbolicReasoningOracle(
            dataset_instance,
            llm_model,
            polynomial_degree=polynomial_degree,
            max_tree_depth=max_tree_depth,
            regularization_strength=regularization_strength,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(
            [direct_and_eval_oracle, symbolic_oracle],
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )

    def get_callable_functions(self):
        return super().get_callable_functions()


class TextualSymbolicRegressionOracle(CombinationOracle):
    """
    Textual variant that combines direct experimentation, theory evaluation, and
    symbolic reasoning helpers.

    Uses :class:`TextualDirectExperimentAndEvaluationOracle` to ensure the underlying
    experiment worker operates on textualized datasets.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        *,
        verbose: bool = False,
        max_experiments: Optional[int] = None,
        polynomial_degree: int = 2,
        max_tree_depth: int = 3,
        regularization_strength: float = 1e-3,
        **kwargs: Any,
    ):
        direct_and_eval_oracle = TextualDirectExperimentAndEvaluationOracle(
            dataset_instance,
            llm_model,
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )
        symbolic_oracle = SymbolicReasoningOracle(
            dataset_instance,
            llm_model,
            polynomial_degree=polynomial_degree,
            max_tree_depth=max_tree_depth,
            regularization_strength=regularization_strength,
            verbose=verbose,
            **kwargs,
        )
        super().__init__(
            [direct_and_eval_oracle, symbolic_oracle],
            verbose=verbose,
            max_experiments=max_experiments,
            **kwargs,
        )

    def get_callable_functions(self):
        return super().get_callable_functions()
