from __future__ import annotations

from typing import Dict, Optional, Tuple, List

from data_management.underlying_data.build_data_distribution import _summarize
from data_management.underlying_data.udd_search import UDDSearchMethod
from data_management.underlying_data.udd_search.genetic_search import find_balanced_decisive_udd_genetic
from data_management.underlying_data.udd_search.random_search import find_balanced_decisive_udd_random
from data_management.underlying_data.underlying_data import UnderlyingDataDistribution
from data_management.underlying_data.output_properties import OutputProperty


def find_balanced_decisive_udd(
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
    search_method: UDDSearchMethod = UDDSearchMethod.REANDOM,
    max_genetic_search_configurations: Optional[int] = None,
    output_properties: Optional[List[OutputProperty]] = None,
) -> Tuple[Optional[UnderlyingDataDistribution], Optional[Dict[str, float]], int, int]:
    """
    Dispatch UDD search to the configured strategy.
    """
    if search_method == UDDSearchMethod.REANDOM:
        return find_balanced_decisive_udd_random(
            num_variables=num_variables,
            min_categories=min_categories,
            max_categories=max_categories,
            num_numerical_variables=num_numerical_variables,
            numerical_min_spans=numerical_min_spans,
            numerical_max_spans=numerical_max_spans,
            num_rules=num_rules,
            num_output_labels=num_output_labels,
            max_ratio_diff=max_ratio_diff,
            min_decisive=min_decisive,
            max_iterations=max_iterations,
            allow_multi_use=allow_multi_use,
            seed_start=seed_start,
            verbose=verbose,
            output_properties=output_properties,
        )
    if search_method == UDDSearchMethod.GENETIC:
        return find_balanced_decisive_udd_genetic(
            num_variables=num_variables,
            min_categories=min_categories,
            max_categories=max_categories,
            num_numerical_variables=num_numerical_variables,
            numerical_min_spans=numerical_min_spans,
            numerical_max_spans=numerical_max_spans,
            num_rules=num_rules,
            max_iterations=max_iterations,
            num_output_labels=num_output_labels,
            max_ratio_diff=max_ratio_diff,
            min_decisive=min_decisive,
            seed_start=seed_start,
            verbose=verbose,
            max_genetic_search_configurations=max_genetic_search_configurations,
            output_properties=output_properties,
        )
    raise ValueError(f"Unsupported UDD search method: {search_method}")


def main():
    udd, analysis, iters, num_iters = find_balanced_decisive_udd(
        num_variables=3,
        min_categories=2,
        max_categories=3,
        num_rules=2,
        max_ratio_diff=0.25,
        min_decisive=0.75,
        max_iterations=10_000,
        allow_multi_use=False,
        seed_start=123,  # set to None for fully random each run
    )

    if udd is None or analysis is None:
        print("No suitable UDD found.")
        return

    print(f"Found in {iters} iterations:")
    print(_summarize(udd))
    print("-" * 60)
    for k, v in analysis.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
