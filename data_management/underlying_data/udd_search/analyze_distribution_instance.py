# analyze_distribution_instance.py
from __future__ import annotations

from typing import Dict, Optional, Iterable, List, Tuple
from itertools import product
import math
import random

from data_management.underlying_data.underlying_data import UnderlyingDataDistribution

try:
    from build_data_distribution import build_data_distribution, _summarize  # demo only
except Exception:
    build_data_distribution = None
    _summarize = None


def analyze_distribution_instance(
    udd: UnderlyingDataDistribution,
    threshold: float = 0.5,
    max_configurations: Optional[int] = None,
) -> Dict[str, float]:
    """
    Exhaustively enumerate all combinations of variable assignments and analyze:
      - counts per predicted label
      - decisiveness: margin between the best and second-best normalized scores

    Returns a dict with counts, fractions, and average decisiveness metrics.
    The ``threshold`` argument is kept for backward compatibility but is ignored
    in the multi-label setup. If ``max_configurations`` is set, the analysis
    samples at most that many variable assignments instead of exhaustive search.
    """
    var_names = list(udd.variables.keys())

    def _decisiveness(scores: Dict[int, float]) -> float:
        if not scores:
            return 0.0
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        second_best = ordered[1] if len(ordered) > 1 else 0.0
        margin = best - second_best
        return 0.0 if margin < 0.0 else (1.0 if margin > 1.0 else margin)

    # Edge case: no variables — evaluate empty assignment
    if not var_names:
        res = udd.evaluate({})
        scores = res.get("normalized_scores", {res.get("predicted_label", 0): res.get("normalized_score", 0.0)})
        dec = _decisiveness(scores)
        best_label = int(res.get("predicted_label", 0))
        counts = {lbl: (1 if lbl == best_label else 0) for lbl in range(udd.num_output_labels)}
        return {
            "total_combinations": 1,
            "label_counts": counts,
            "label_fractions": {lbl: float(counts.get(lbl, 0)) for lbl in counts},
            "avg_decisiveness": dec,
            "avg_top_score": float(scores.get(best_label, 0.0)),
        }

    domain_sizes = [udd.variables[name].categories for name in var_names]
    total_combinations = math.prod(domain_sizes)
    # print(f"Total combinations to analyze: {total_combinations}")

    label_counts: Dict[int, int] = {lbl: 0 for lbl in range(udd.num_output_labels)}

    def _combo_from_index(index: int, sizes: List[int]) -> List[int]:
        combo: List[int] = []
        for size in reversed(sizes):
            index, rem = divmod(index, size)
            combo.append(rem)
        combo.reverse()
        return combo

    def _iter_combinations() -> Iterable[Tuple[int, ...]]:
        if max_configurations is None or max_configurations <= 0 or total_combinations <= max_configurations:
            return product(*[range(n) for n in domain_sizes])
        indices = random.sample(range(total_combinations), k=max_configurations)
        return (tuple(_combo_from_index(idx, domain_sizes)) for idx in indices)

    total = min(total_combinations, max_configurations) if max_configurations and max_configurations > 0 else total_combinations

    sum_dec_all = 0.0
    sum_top_score = 0.0

    for combo in _iter_combinations():
        assignment = {name: cat for name, cat in zip(var_names, combo)}
        res = udd.evaluate(assignment)
        scores = res.get("normalized_scores", {res.get("predicted_label", 0): res.get("normalized_score", 0.0)})
        dec = _decisiveness(scores)
        best_label = int(res.get("predicted_label", 0))

        label_counts[best_label] = label_counts.get(best_label, 0) + 1

        sum_dec_all += dec
        sum_top_score += float(scores.get(best_label, 0.0))

    return {
        "total_combinations": total,
        "label_counts": dict(label_counts),
        "label_fractions": {
            lbl: (count / total) if total else 0.0 for lbl, count in label_counts.items()
        },
        "avg_decisiveness": (sum_dec_all / total) if total else 0.0,
        "avg_top_score": (sum_top_score / total) if total else 0.0,
        "max_fraction_gap_from_uniform": max(
            abs((label_counts[lbl] / total) - (1.0 / udd.num_output_labels))
            if total else 0.0
            for lbl in label_counts
        ),
    }
