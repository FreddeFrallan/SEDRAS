from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import json
import os


DATASET_CREATION_STATISTICS_FILENAME = "dataset_creation_statistics.json"


@dataclass
class DatasetCreationStatistics:
    """
    Lightweight container for tracking metrics during dataset creation.

    Currently records:
      - Number of iterations required to find a suitable UDD.
      - Number of samples per label in the final dataset.
      - Exhaustive analysis metrics from ``analyze_distribution_instance``.

    This object can be saved alongside a dataset and later reloaded to
    inspect how it was generated.
    """

    udd_search_iterations: int = 0
    samples_per_label: Dict[str, int] = field(default_factory=dict)
    total_number_of_assignments: int = 0
    total_samples_per_label: Dict[int, int] = field(default_factory=dict)
    analysis_results: Dict[str, float] = field(default_factory=dict)

    # ------------------ mutation helpers ------------------
    def record_udd_search_iterations(self, iterations: int) -> None:
        self.udd_search_iterations = int(iterations)

    def record_samples_per_label(self, counts: Dict[int | str, int]) -> None:
        self.samples_per_label = {str(label): int(count) for label, count in counts.items()}

    def record_total_samples_per_label(self, counts: Dict[int | int, int]) -> None:
        self.total_samples_per_label = {int(label): int(count) for label, count in counts.items()}

    def record_total_number_of_assignments(self, total: int) -> None:
        self.total_number_of_assignments = int(total)

    def record_analysis_results(self, analysis)-> None:
        self.analysis_results = {k: v for k, v in analysis.items()}

    # ------------------ serialization helpers ------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "udd_search_iterations": self.udd_search_iterations,
            "samples_per_label": dict(self.samples_per_label),
            "total_samples_per_label": dict(self.total_samples_per_label),
            "total_number_of_assignments": self.total_number_of_assignments,
            "analysis_results": dict(self.analysis_results),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetCreationStatistics":
        stats = cls()
        stats.udd_search_iterations = int(data.get("udd_search_iterations", 0))
        samples_raw = data.get("samples_per_label", {}) or {}
        if isinstance(samples_raw, dict):
            stats.samples_per_label = {str(k): int(v) for k, v in samples_raw.items()}
        total_samples_raw = data.get("total_samples_per_label", {}) or {}
        if isinstance(total_samples_raw, dict):
            stats.total_samples_per_label = {int(k): int(v) for k, v in total_samples_raw.items()}
        stats.total_number_of_assignments = int(data.get("total_number_of_assignments", 0))
        analysis_raw = data.get("analysis_results", {}) or {}
        if isinstance(analysis_raw, dict):
            stats.analysis_results = {str(k): float(v) for k, v in analysis_raw.items()}
        return stats

    def save(self, folder_path: str) -> str:
        """
        Persist the statistics alongside a dataset folder.

        Returns the path to the written file.
        """
        folder_path = os.path.abspath(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        fpath = os.path.join(folder_path, DATASET_CREATION_STATISTICS_FILENAME)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return fpath

    @classmethod
    def load(cls, folder_path: str) -> "DatasetCreationStatistics":
        fpath = os.path.join(os.path.abspath(folder_path), DATASET_CREATION_STATISTICS_FILENAME)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
