from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any
from data_management.underlying_data.rule import Rule
from data_management.underlying_data.variables import Variable, VariableType
import json
import os


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))




@dataclass
class UnderlyingDataDistribution:
    """
    Holds variables and rules.
    Given an assignment {var_name: category}, it:
      - Adds each variable's selected category (weight, score)
      - Adds each rule's (weight, score) for every rule that matches
      - Does NOT consume variables; all matches are independent
    Returns aggregates + per-entity info.
    """
    num_output_labels: int = 2
    variables: Dict[str, Variable] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    def add_variable(self, var: Variable) -> None:
        if var.num_output_labels != self.num_output_labels:
            var.num_output_labels = self.num_output_labels
            var.__post_init__()
        self.variables[var.name] = var

    def add_rule(self, rule: Rule) -> None:
        var_a = self.variables.get(rule.var_a)
        var_b = self.variables.get(rule.var_b)
        if var_a is None or var_b is None:
            raise ValueError(f"Rule {rule.name} references unknown variables.")

        def _validate(var: Variable, value: int, which: str) -> None:
            if value < 0 or value >= var.categories:
                raise ValueError(
                    f"Rule {rule.name} value {which}={value} out of range for {var.name}"
                )

        def _normalize_value(var: Variable, raw_value: object, which: str) -> int:
            try:
                return var.category_from_assignment(raw_value)
            except Exception as exc:
                raise ValueError(
                    f"Rule {rule.name} invalid {which} value {raw_value!r} for variable {var.name}"
                ) from exc

        rule.val_a = _normalize_value(var_a, rule.val_a, "val_a")
        rule.val_b = _normalize_value(var_b, rule.val_b, "val_b")
        _validate(var_a, rule.val_a, "val_a")
        _validate(var_b, rule.val_b, "val_b")

        self.rules.append(rule)

    def evaluate(
        self,
        assignment: Dict[str, Any],
        numerical_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        entity_weights: Dict[int, Dict[str, float]] = {
            lbl: {} for lbl in range(self.num_output_labels)
        }
        entity_scores: Dict[int, Dict[str, float]] = {
            lbl: {} for lbl in range(self.num_output_labels)
        }
        numerical_values = numerical_values or {}

        normalized_assignment: Dict[str, int] = {}
        resolved_numerical_values: Dict[str, float] = {}
        for name, raw_value in assignment.items():
            var = self.variables.get(name)
            if var is None:
                continue
            normalized_assignment[name] = var.category_from_assignment(raw_value)
            if var.variable_type is VariableType.NUMERICAL:
                if name in numerical_values:
                    resolved_numerical_values[name] = float(numerical_values[name])
                else:
                    try:
                        val = float(raw_value)
                        if not (isinstance(raw_value, bool) or float(val).is_integer()):
                            resolved_numerical_values[name] = val
                    except Exception:
                        pass

        # Variables contribute their selected category's (w, s) per label
        for name, var in self.variables.items():
            if name in normalized_assignment:
                cat = normalized_assignment[name]
                if var.variable_type is VariableType.NUMERICAL:
                    numeric_value = resolved_numerical_values.get(name)
                    if numeric_value is None:
                        start, end = var.span_for(cat)
                        numeric_value = (start + end) / 2.0
                    for lbl in range(self.num_output_labels):
                        entity_weights[lbl][name] = var.weight_for(lbl, cat, numeric_value)
                        entity_scores[lbl][name] = var.score_for(lbl, cat)
                else:
                    for lbl in range(self.num_output_labels):
                        entity_weights[lbl][name] = var.weight_for(lbl, cat)
                        entity_scores[lbl][name] = var.score_for(lbl, cat)

        # Rules contribute independently if their pattern matches (shared across labels)
        applied_rules: List[str] = []
        for rule in self.rules:
            if (
                rule.var_a in normalized_assignment
                and rule.var_b in normalized_assignment
                and normalized_assignment[rule.var_a] == rule.val_a
                and normalized_assignment[rule.var_b] == rule.val_b
            ):
                for lbl in range(self.num_output_labels):
                    entity_weights[lbl][rule.name] = (
                        entity_weights[lbl].get(rule.name, 0.0) + rule.weight
                    )
                    entity_scores[lbl][rule.name] = rule.score
                applied_rules.append(rule.name)

        weighted_sums: Dict[int, float] = {}
        sum_weights: Dict[int, float] = {}
        normalized_scores: Dict[int, float] = {}
        for lbl in range(self.num_output_labels):
            weighted_sums[lbl] = sum(
                entity_weights[lbl][k] * entity_scores[lbl][k] for k in entity_weights[lbl]
            )
            sum_weights[lbl] = sum(entity_weights[lbl].values())
            normalized_scores[lbl] = (
                (weighted_sums[lbl] / sum_weights[lbl]) if sum_weights[lbl] > 0 else 0.0
            )

        best_label = max(normalized_scores, key=lambda lbl: (normalized_scores[lbl], -lbl))
        best_score = normalized_scores[best_label]

        return {
            "weighted_sums": weighted_sums,
            "sum_weights": sum_weights,
            "normalized_scores": normalized_scores,
            "predicted_label": best_label,
            "entity_weights": {lbl: dict(w) for lbl, w in entity_weights.items()},
            "entity_scores": {lbl: dict(s) for lbl, s in entity_scores.items()},
            "applied_rules": applied_rules,
            # Backwards-compatible top-label aliases
            "weighted_sum": weighted_sums[best_label],
            "sum_weights": sum_weights[best_label],
            "normalized_score": best_score,
        }

    def make_classifier_fn(
        self,
        variable_order: Optional[List[str]] = None,
    ):
        """
        Create a callable classifier function equivalent to an LLM-produced one.

        The returned function:
            classify(x: list[float]) -> int

        - x[i] corresponds to the category for variable_order[i] (categorical
          variables use integer categories; numerical variables use floating
          values).
        - It computes the UDD's per-label normalized scores for that assignment.
        - Returns the label with the highest normalized score.
        """

        # Determine a consistent variable order
        if variable_order is None:
            variable_order = sorted(self.variables.keys())

        def classify(x: List[float]) -> int:
            if len(x) != len(variable_order):
                raise ValueError(
                    f"Expected {len(variable_order)} inputs, got {len(x)}"
                )

            assignment: Dict[str, Any] = {}
            numerical_values: Dict[str, float] = {}

            for var_name, raw_value in zip(variable_order, x):
                var = self.variables.get(var_name)
                if var is None:
                    continue
                if var.variable_type is VariableType.NUMERICAL:
                    numeric_val = float(raw_value)
                    assignment[var_name] = numeric_val
                    numerical_values[var_name] = numeric_val
                else:
                    assignment[var_name] = int(raw_value)

            # Evaluate the UDD
            result = self.evaluate(
                assignment, numerical_values=numerical_values or None
            )
            label = int(result.get("predicted_label", 0))

            return label

        return classify

    def export_genetic_dna(self) -> Dict[str, Any]:
        """
        Exports the mutable parameters of the UDD into a structured 'DNA' dictionary.
        This DNA contains only the values that a Genetic Algorithm would want to
        mutate or crossover.
        """
        dna = {
            "num_output_labels": self.num_output_labels,
            "variables": {},
            "rules": []
        }

        # 1. Variable DNA
        for name, var in self.variables.items():
            v_dna = {
                "type": var.variable_type.value,
                "categories": var.categories,
            }

            if var.variable_type == VariableType.CATEGORICAL:
                v_dna["weights"] = var.weights
                v_dna["scores"] = var.scores
            else:
                # Numerical parameters
                v_dna["min_weight"] = var.min_weight
                v_dna["max_weight"] = var.max_weight
                v_dna["direction"] = var.direction
                v_dna["variable_score"] = var.variable_score
                v_dna["spans"] = var.spans

            dna["variables"][name] = v_dna

        # 2. Rule DNA
        for rule in self.rules:
            dna["rules"].append({
                "name": rule.name,
                "var_a": rule.var_a,
                "val_a": rule.val_a,
                "var_b": rule.var_b,
                "val_b": rule.val_b,
                "weight": rule.weight,
                "score": rule.score
            })

        return dna

    @classmethod
    def from_genetic_dna(cls, dna: Dict[str, Any]) -> UnderlyingDataDistribution:
        """
        Factory method to create a new UDD instance from a DNA dictionary.
        """
        udd = cls(num_output_labels=dna["num_output_labels"])

        # Reconstruct Variables
        for name, v_dna in dna["variables"].items():
            var = Variable(
                name=name,
                categories=v_dna["categories"],
                num_output_labels=udd.num_output_labels,
                variable_type=VariableType(v_dna["type"]),
                # Use .get() to handle the conditional existence of fields
                weights=v_dna.get("weights", {}),
                scores=v_dna.get("scores", {}),
                min_weight=v_dna.get("min_weight"),
                max_weight=v_dna.get("max_weight"),
                direction=v_dna.get("direction"),
                variable_score=v_dna.get("variable_score"),
                spans=v_dna.get("spans", [])
            )
            udd.add_variable(var)

        # Reconstruct Rules
        for r_dna in dna["rules"]:
            rule = Rule(
                name=r_dna["name"],
                var_a=r_dna["var_a"],
                val_a=r_dna["val_a"],
                var_b=r_dna["var_b"],
                val_b=r_dna["val_b"],
                weight=r_dna["weight"],
                score=r_dna["score"]
            )
            udd.add_rule(rule)

        return udd

    # ---------------------------
    # Save / Load
    # ---------------------------

    def save(self, json_path: str) -> None:
        """Save the UDD (variables + rules) as a JSON file."""
        data = {
            "num_output_labels": self.num_output_labels,
            "variables": {name: var.to_dict() for name, var in self.variables.items()},
            "rules": [r.to_dict() for r in self.rules],
        }
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved UnderlyingDataDistribution to {json_path}")

    @classmethod
    def load(cls, json_path: str) -> UnderlyingDataDistribution:
        """Load a UDD object from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        num_output_labels = int(data.get("num_output_labels", 2))
        variables = {
            name: Variable.from_dict(vdict) for name, vdict in data["variables"].items()
        }
        for var in variables.values():
            if var.num_output_labels != num_output_labels:
                var.num_output_labels = num_output_labels
                var.__post_init__()
        rules = [Rule.from_dict(rdict) for rdict in data["rules"]]

        udd = cls(num_output_labels=num_output_labels, variables=variables, rules=rules)
        print(f"✅ Loaded UnderlyingDataDistribution from {json_path}")
        return udd


