from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import random

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class Rule:
    """
    Independent rule: if (var_a == val_a) and (var_b == val_b), this rule
    contributes its own (weight, score) as an additional entity.
    The (weight, score) are either provided or randomized once at init.

    """
    name: str
    var_a: str
    val_a: object
    var_b: str
    val_b: object
    weight: Optional[float] = None
    score: Optional[float] = None

    def __post_init__(self):
        if self.weight is None:
            self.weight = random.random()
        if self.score is None:
            self.score = random.random()
        self.weight = clamp01(self.weight)
        self.score = clamp01(self.score)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "var_a": self.var_a,
            "val_a": self.val_a,
            "var_b": self.var_b,
            "val_b": self.val_b,
            "weight": self.weight,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Rule:
        return cls(
            name=data["name"],
            var_a=data["var_a"],
            val_a=data["val_a"],
            var_b=data["var_b"],
            val_b=data["val_b"],
            weight=float(data["weight"]),
            score=float(data["score"]),
        )