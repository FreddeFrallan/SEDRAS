from typing import Any, Dict, List, Optional, Set, Callable


class SamplesTracker:
    """
    A tracking wrapper around the dataset samples.

    This class is passed into the LLM-defined block functions. It allows
    the LLM to query the data flexibly while forcing it to use a specific
    method to retrieve data for rendering, which allows us to track coverage.
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self._samples = samples
        self._used_indices: Set[int] = set()

    # --- Query API (Does not trigger tracking) ---

    def get_total_count(self) -> int:
        """Returns total number of samples in the dataset."""
        return len(self._samples)

    def get_all_indices(self) -> List[int]:
        """Returns a list of all integer indices [0, 1, 2, ...]."""
        return list(range(len(self._samples)))

    def get_indices_by_label(self, label: int) -> List[int]:
        """Returns indices where the sample label matches the input."""
        return [i for i, s in enumerate(self._samples) if s.get("label") == label]

    def get_label_for_index(self, index: int) -> Optional[int]:
        """Returns the label for a given sample index, or None if index is invalid."""
        if 0 <= index < len(self._samples):
            return self._samples[index].get("label")
        return None

    def get_indices_matching(self, condition: Callable[[Dict[str, Any]], bool]) -> List[int]:
        """
        Generic filter.
        Example: tracker.get_indices_matching(lambda s: len(s['instance_text']) > 100)
        """
        return [i for i, s in enumerate(self._samples) if condition(s)]

    def get_unused_indices(self) -> List[int]:
        """Returns indices that have not been retrieved via get_samples_for_rendering yet."""
        return [i for i in range(len(self._samples)) if i not in self._used_indices]

    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        """
        Returns a sample's data for analysis/logic without marking it as 'rendered'.
        Useful for blocks that need to check content before deciding how to layout.
        """
        if 0 <= index < len(self._samples):
            return self._samples[index]
        return {}

    # --- Tracking API (Retrieval for Rendering) ---

    def get_samples_for_rendering(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        THE MANDATORY RETRIEVAL FUNCTION.

        This method returns the actual sample data to be appended to the ReportLab story.
        Calling this function marks the provided indices as 'tracked'.
        """
        out = []
        for i in indices:
            if 0 <= i < len(self._samples):
                self._used_indices.add(i)
                out.append(self._samples[i])
        return out

    # --- Validation API (Used by the Engine) ---

    def get_missing_indices(self) -> List[int]:
        """Returns indices that were never requested for rendering."""
        return sorted(list(set(range(len(self._samples))) - self._used_indices))

    def is_fully_covered(self) -> bool:
        """Checks if all samples have been tracked."""
        return len(self._used_indices) == len(self._samples)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Returns stats for logging."""
        return {
            "total": len(self._samples),
            "used": len(self._used_indices),
            "missing": self.get_missing_indices()
        }