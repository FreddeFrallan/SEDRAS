from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterable, Any
import random
import json
import os

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

class VariableType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


@dataclass
class Variable:
    """
    A variable with a fixed number of categories.

    For categorical variables, weights/scores are defined PER OUTPUT LABEL:
        weights[label][category] ∈ [0, 1]
        scores[label][category]  ∈ [0, 1]

    For numerical variables, the interpolation parameters are also per label:
        min_weight[label], max_weight[label] ∈ [0, 1]
        direction[label] ∈ {True, False}
        variable_score[label] ∈ [0, 1]

    Missing categories/labels are randomized once at init.
    """
    name: str
    categories: int = 3
    num_output_labels: int = 2
    weights: Dict[int, Dict[int, float]] = field(default_factory=dict)
    scores: Dict[int, Dict[int, float]] = field(default_factory=dict)
    variable_type: VariableType = VariableType.CATEGORICAL
    spans: List[Tuple[float, float]] = field(default_factory=list)
    min_weight: Optional[Dict[int, float]] = None
    max_weight: Optional[Dict[int, float]] = None
    direction: Optional[Dict[int, bool]] = None
    variable_score: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if isinstance(self.variable_type, str):
            self.variable_type = VariableType(self.variable_type)

        self.categories = int(self.categories)
        self.num_output_labels = max(1, int(self.num_output_labels))
        if self.categories < 1:
            raise ValueError("Variable must have at least one category/span.")

        if self.variable_type is VariableType.NUMERICAL:
            if not self.spans:
                self.spans = self._generate_equal_spans(self.categories)
            else:
                self.spans = [self._normalize_span(span) for span in self.spans]
                self.categories = len(self.spans)
        else:
            self.spans = []

        def _normalize_category_table(table: Dict[int, Dict[int, float]]) -> Dict[int, Dict[int, float]]:
            # Backward compatibility: if table maps category->value, broadcast to all labels
            normalized: Dict[int, Dict[int, float]] = {
                lbl: {} for lbl in range(self.num_output_labels)
            }
            if table:
                sample_val = next(iter(table.values()))
                if isinstance(sample_val, dict):
                    for raw_lbl, cat_map in table.items():
                        lbl = int(raw_lbl)
                        normalized[lbl] = {int(k): clamp01(v) for k, v in cat_map.items()}
                else:
                    base = {int(k): clamp01(v) for k, v in table.items()}
                    for lbl in normalized:
                        normalized[lbl].update(base)
            for lbl in range(self.num_output_labels):
                for k in range(self.categories):
                    if k not in normalized[lbl]:
                        normalized[lbl][k] = random.random()
            return normalized

        def _normalize_scalar_by_label(
            raw_value: Optional[Dict[int, Any] | float | bool],
            *,
            clamp_fn,
            random_fn,
        ) -> Dict[int, Any]:
            normalized: Dict[int, Any] = {}
            if isinstance(raw_value, dict):
                normalized = {int(k): clamp_fn(v) for k, v in raw_value.items()}
            elif raw_value is not None:
                v = clamp_fn(raw_value)
                normalized = {lbl: v for lbl in range(self.num_output_labels)}
            for lbl in range(self.num_output_labels):
                if lbl not in normalized:
                    normalized[lbl] = clamp_fn(random_fn(lbl))
            return normalized

        if self.variable_type is VariableType.NUMERICAL:
            self.variable_score = _normalize_scalar_by_label(
                self.variable_score,
                clamp_fn=clamp01,
                random_fn=lambda _: random.random(),
            )
            self.min_weight = _normalize_scalar_by_label(
                self.min_weight,
                clamp_fn=clamp01,
                random_fn=lambda _: random.random(),
            )
            self.max_weight = _normalize_scalar_by_label(
                self.max_weight,
                clamp_fn=clamp01,
                random_fn=lambda _: random.random(),
            )
            self.direction = _normalize_scalar_by_label(
                self.direction,
                clamp_fn=lambda v: bool(v),
                random_fn=lambda _: bool(random.getrandbits(1)),
            )

            for lbl in range(self.num_output_labels):
                if self.min_weight[lbl] > self.max_weight[lbl]:
                    self.min_weight[lbl], self.max_weight[lbl] = (
                        self.max_weight[lbl],
                        self.min_weight[lbl],
                    )

            # Preserve any explicitly provided weights/scores but don't auto-populate
            self.weights = _normalize_category_table(self.weights)
            self.scores = _normalize_category_table(self.scores)
        else:
            self.weights = _normalize_category_table(self.weights)
            self.scores = _normalize_category_table(self.scores)

    @staticmethod
    def _normalize_span(span: Iterable[float]) -> Tuple[float, float]:
        try:
            start, end = span
        except Exception as exc:
            raise ValueError(f"Invalid span definition: {span!r}") from exc
        start = clamp01(start)
        end = clamp01(end)
        if end < start:
            start, end = end, start
        if end == start:
            end = min(1.0, start + 1e-6)
        return (float(start), float(end))

    @staticmethod
    def _generate_equal_spans(num_spans: int) -> List[Tuple[float, float]]:
        if num_spans < 1:
            raise ValueError("num_spans must be >= 1")
        spans: List[Tuple[float, float]] = []
        for idx in range(num_spans):
            start = idx / num_spans
            end = (idx + 1) / num_spans
            if idx == num_spans - 1:
                end = 1.0
            spans.append((start, end))
        return spans

    def span_for(self, value: int) -> Tuple[float, float]:
        if self.variable_type is not VariableType.NUMERICAL:
            raise ValueError("span_for is only valid for numerical variables")
        if value not in range(self.categories):
            raise KeyError(f"{self.name} has no span for category {value}")
        return self.spans[value]

    def span_index_for_value(self, numeric_value: float) -> int:
        """Return the span index that contains the provided numeric value."""
        if self.variable_type is not VariableType.NUMERICAL:
            raise ValueError("span_index_for_value is only valid for numerical variables")
        if not self.spans:
            raise ValueError(f"{self.name} does not define any spans")

        x = float(numeric_value)
        for idx, (start, end) in enumerate(self.spans):
            if idx == len(self.spans) - 1:
                if x < start:
                    continue
                if x <= end:
                    return idx
            if start <= x < end:
                return idx

        # If the value lies slightly outside due to floating point noise, clamp
        if x < self.spans[0][0]:
            return 0
        return len(self.spans) - 1

    def category_from_assignment(self, raw_value: object) -> int:
        """
        Interpret an assignment entry and return the resolved category index.

        For numerical variables we allow either the explicit span index or a
        concrete numeric value in [0, 1]; the latter is mapped to the
        corresponding span index.
        """

        def _parse_value(value: object) -> object:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    raise ValueError(f"Empty assignment value for {self.name}")
                try:
                    return int(stripped)
                except ValueError:
                    return float(stripped)
            raise TypeError(f"Unsupported assignment value type for {self.name}: {type(value)!r}")

        value = _parse_value(raw_value)
        if isinstance(value, int):
            idx = value
        elif isinstance(value, float):
            if value.is_integer():
                idx = int(value)
            elif self.variable_type is VariableType.NUMERICAL:
                idx = self.span_index_for_value(value)
            else:
                raise ValueError(
                    f"Non-integer assignment {value} provided for categorical variable {self.name}"
                )
        else:
            raise TypeError(f"Unhandled assignment type for {self.name}: {type(value)!r}")

        if idx < 0 or idx >= self.categories:
            raise ValueError(f"Assignment value {idx} out of range for {self.name}")
        return idx

    def sample_value_for_category(self, value: int) -> float:
        if self.variable_type is not VariableType.NUMERICAL:
            raise ValueError("sample_value_for_category is only valid for numerical variables")
        start, end = self.span_for(value)
        return random.uniform(start, end)

    def _interpolated_weight(self, label: int, numeric_value: float) -> float:
        if (
            self.min_weight is None
            or self.max_weight is None
            or self.direction is None
            or label not in self.min_weight
            or label not in self.max_weight
            or label not in self.direction
        ):
            raise KeyError(f"{self.name} is missing interpolation bounds for label {label}")
        t = clamp01(float(numeric_value))
        if not self.direction[label]:
            t = 1.0 - t
        w = self.min_weight[label] + t * (self.max_weight[label] - self.min_weight[label])
        return clamp01(w)

    def weight_for(self, label: int, value: int, numeric_value: Optional[float] = None) -> float:
        if self.variable_type is VariableType.NUMERICAL:
            if numeric_value is None:
                start, end = self.span_for(value)
                numeric_value = (start + end) / 2.0
            return self._interpolated_weight(label, numeric_value)
        if label not in self.weights or value not in self.weights[label]:
            raise KeyError(f"{self.name} has no weight for label {label}, category {value}")
        return self.weights[label][value]

    def score_for(self, label: int, value: int) -> float:
        if self.variable_type is VariableType.NUMERICAL:
            if self.variable_score is None or label not in self.variable_score:
                raise KeyError(f"{self.name} has no score configured for label {label}")
            return self.variable_score[label]
        if label not in self.scores or value not in self.scores[label]:
            raise KeyError(f"{self.name} has no score for label {label}, category {value}")
        return self.scores[label][value]

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "categories": self.categories,
            "num_output_labels": self.num_output_labels,
            "weights": self.weights,
            "scores": self.scores,
            "variable_type": self.variable_type.value,
            "spans": [[s, e] for s, e in self.spans],
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "direction": self.direction,
            "variable_score": self.variable_score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Variable:
        raw_type = data.get("variable_type", VariableType.CATEGORICAL)
        variable_type = raw_type if isinstance(raw_type, VariableType) else VariableType(raw_type)
        return cls(
            name=data["name"],
            categories=int(data["categories"]),
            num_output_labels=int(data.get("num_output_labels", 2)),
            weights={int(lbl): {int(k): float(v) for k, v in inner.items()} for lbl, inner in (data.get("weights", {}) or {}).items()},
            scores={int(lbl): {int(k): float(v) for k, v in inner.items()} for lbl, inner in (data.get("scores", {}) or {}).items()},
            variable_type=variable_type,
            spans=[tuple(map(float, span)) for span in data.get("spans", [])],
            min_weight={int(k): float(v) for k, v in (data.get("min_weight", {}) or {}).items()},
            max_weight={int(k): float(v) for k, v in (data.get("max_weight", {}) or {}).items()},
            direction={int(k): bool(v) for k, v in (data.get("direction", {}) or {}).items()},
            variable_score={int(k): float(v) for k, v in (data.get("variable_score", {}) or {}).items()},
        )

