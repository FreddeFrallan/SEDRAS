from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import tqdm


from data_management.underlying_data.build_data_distribution import build_data_distribution
from data_management.underlying_data.underlying_data import UnderlyingDataDistribution
from data_management.underlying_data.udd_search.utility_function import get_udd_value_score
from data_management.underlying_data.output_properties import OutputProperty

def find_balanced_decisive_udd_random(
    *,
    num_variables: int = 3,
    min_categories: int = 2,
    max_categories: int = 3,
    num_numerical_variables: int = 0,
    numerical_min_spans: int = 2,
    numerical_max_spans: int = 4,
    num_rules: int = 2,
    num_output_labels: int = 2,
    max_ratio_diff: float = 0.25,
    min_decisive: float = 0.25,
    max_iterations: int = 10_000,
    allow_multi_use: bool = False,
    seed_start: Optional[int] = None,
    verbose: bool = True,
    output_properties: Optional[List[OutputProperty]] = None,
) -> Tuple[Optional[UnderlyingDataDistribution], Optional[Dict[str, float]], int, int]:
    """
    Randomly tries to find a UDD that satisfies BOTH:
      1) label distribution is close to uniform (max_fraction_gap_from_uniform <= max_ratio_diff)
      2) min_decisiveness_overall >= min_decisive
         (i.e., EVERY combination's decisiveness margin >= min_decisive)

    Returns:
        (udd, analysis_augmented, iterations_used)
        If not found, returns (best_udd, best_analysis_augmented, max_iterations)
        where penalty prefers smaller imbalance and larger min decisiveness.
    """



    best: Tuple[Optional[UnderlyingDataDistribution], Optional[Dict[str, float]], float] = (
        None,
        None,
        math.inf,
    )

    for i in tqdm.tqdm(range(max_iterations), disable=not verbose, desc="Searching for balanced decisive UDD"):
        seed = (seed_start + i) if seed_start is not None else None

        udd = build_data_distribution(
            num_variables=num_variables,
            min_categories=min_categories,
            max_categories=max_categories,
            num_numerical_variables=num_numerical_variables,
            numerical_min_spans=numerical_min_spans,
            numerical_max_spans=numerical_max_spans,
            num_rules=num_rules,
            num_output_labels=num_output_labels,
            allow_multi_use=allow_multi_use,
            seed=seed,
        )

        penalty, passes_contraints, analysis_aug = get_udd_value_score(
            udd,
            max_ratio_diff,
            min_decisive,
            output_properties=output_properties,
        )

        if penalty < best[2]:
            best = (udd, analysis_aug, penalty)
        if passes_contraints:
            return udd, analysis_aug, i + 1, i + 1

    # Not found — return best candidate
    return best[0], best[1], max_iterations, i
