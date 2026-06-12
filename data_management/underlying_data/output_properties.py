from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class OutputPropertyType(str, Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"


@dataclass
class OutputProperty(ABC):
    """
    Base class for output properties.
    """

    name: str
    property_type: OutputPropertyType

    @abstractmethod
    def to_metadata(self) -> Dict[str, Any]:
        """
        Serialize the property definition so it can be attached to dataset metadata.
        """
        raise NotImplementedError


@dataclass
class CategoricalOutputProperty(OutputProperty):
    """
    Categorical output property with a fixed number of categories.
    """

    num_categories: int

    def __init__(self, name: str, num_categories: int):
        if num_categories < 1:
            raise ValueError("num_categories must be at least 1")
        super().__init__(name=name, property_type=OutputPropertyType.CATEGORICAL)
        self.num_categories = int(num_categories)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.property_type.value,
            "num_categories": self.num_categories,
        }


@dataclass
class NumericalOutputProperty(OutputProperty):
    """
    Numerical/continuous output property.
    """

    min_value: float = 0.0
    max_value: float = 1.0
    unit: Optional[str] = None

    def __init__(self, name: str, min_value: float = 0.0, max_value: float = 1.0, unit: Optional[str] = None):
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value")
        super().__init__(name=name, property_type=OutputPropertyType.NUMERICAL)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.unit = unit

    def to_metadata(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "type": self.property_type.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
        if self.unit:
            data["unit"] = self.unit
        return data
