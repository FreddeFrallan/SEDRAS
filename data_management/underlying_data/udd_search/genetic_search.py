from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

from data_management.underlying_data.build_data_distribution import build_data_distribution
from data_management.underlying_data.underlying_data import UnderlyingDataDistribution
from data_management.underlying_data.udd_search.utility_function import get_udd_value_score
from data_management.underlying_data.output_properties import OutputProperty

import random
import copy
import math
import tqdm

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def mutate_dna(dna: Dict[str, Any], mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> Dict[str, Any]:
    """Randomly perturbs the DNA values."""
    child_dna = copy.deepcopy(dna)

    # Mutate Variables
    for var_name, v_dna in child_dna["variables"].items():
        if v_dna["type"] == "categorical":
            for lbl in v_dna["weights"]:
                for cat in v_dna["weights"][lbl]:
                    if random.random() < mutation_rate:
                        v_dna["weights"][lbl][cat] = clamp01(
                            v_dna["weights"][lbl][cat] + random.uniform(-mutation_strength, mutation_strength))
                        v_dna["scores"][lbl][cat] = clamp01(
                            v_dna["scores"][lbl][cat] + random.uniform(-mutation_strength, mutation_strength))
        else:
            # Numerical mutation
            for lbl in range(child_dna["num_output_labels"]):
                if random.random() < mutation_rate:
                    v_dna["min_weight"][lbl] = clamp01(
                        v_dna["min_weight"][lbl] + random.uniform(-mutation_strength, mutation_strength))
                    v_dna["max_weight"][lbl] = clamp01(
                        v_dna["max_weight"][lbl] + random.uniform(-mutation_strength, mutation_strength))
                    v_dna["variable_score"][lbl] = clamp01(
                        v_dna["variable_score"][lbl] + random.uniform(-mutation_strength, mutation_strength))

    # Mutate Rules
    for rule in child_dna["rules"]:
        if random.random() < mutation_rate:
            rule["weight"] = clamp01(rule["weight"] + random.uniform(-mutation_strength, mutation_strength))
            rule["score"] = clamp01(rule["score"] + random.uniform(-mutation_strength, mutation_strength))
        # Occasionally swap rule targets
        if random.random() < (mutation_rate * 0.5):
            var_a = child_dna["variables"][rule["var_a"]]
            rule["val_a"] = random.randint(0, var_a["categories"] - 1)

    return child_dna


def create_initial_population_dna_from_noise(
        population_size: int,
        num_variables: int,
        min_categories: int,
        max_categories: int,
        num_numerical_variables: int,
        numerical_min_spans: int,
        numerical_max_spans: int,
        num_rules: int,
        num_output_labels: int,
        allow_multi_use: bool,
        mutation_rate: float,
        seed_start: Optional[int] = None
) -> List[Any]:
    """
    Creates an initial population by generating one base UDD and
    applying random mutation noise to create variations.
    """
    # 1. Create the single "Seed" UDD
    # This UDD defines the structure (counts) for the entire population
    base_udd = build_data_distribution(
        num_variables=num_variables,
        min_categories=min_categories,
        max_categories=max_categories,
        num_numerical_variables=num_numerical_variables,
        numerical_min_spans=numerical_min_spans,
        numerical_max_spans=numerical_max_spans,
        num_rules=num_rules,
        num_output_labels=num_output_labels,
        seed=seed_start,
        allow_multi_use=allow_multi_use
    )

    base_dna = base_udd.export_genetic_dna()
    population_dna = [base_dna]  # Keep the original as the first member

    # 2. Generate the rest of the population by adding noise
    # We use a loop to create (size - 1) mutated versions
    for i in range(1, population_size):
        # Apply mutation to the base DNA to create a "noisy" neighbor
        noisy_dna = mutate_dna(base_dna, mutation_rate=mutation_rate)
        population_dna.append(noisy_dna)

    return population_dna


def find_balanced_decisive_udd_genetic(
        *,
        num_variables: int = 3,
        min_categories: int = 2,
        max_categories: int = 3,
        num_numerical_variables: int = 0,
        numerical_min_spans: int = 2,
        numerical_max_spans: int = 4,
        num_rules: int = 2,
        num_output_labels: int = 2,
        max_ratio_diff: float = 0.2,
        min_decisive: float = 0.2,
        max_iterations: int = 2_000,  # Global evaluation budget
        allow_multi_use: bool = False,
        seed_start: Optional[int] = None,
        verbose: bool = True,
        max_genetic_search_configurations: Optional[int] = None,
        output_properties: Optional[List[OutputProperty]] = None,
        # GA Specific Hyperparameters
        population_size: int = 100,
        mutation_rate: float = 0.2,
) -> Tuple[Optional[UnderlyingDataDistribution], Optional[Dict[str, float]], int, int]:
    # 1. Initialize Population
    population_dna = create_initial_population_dna_from_noise(
        population_size=population_size,
        num_variables=num_variables,
        min_categories=min_categories,
        max_categories=max_categories,
        num_numerical_variables=num_numerical_variables,
        numerical_min_spans=numerical_min_spans,
        numerical_max_spans=numerical_max_spans,
        num_rules=num_rules,
        num_output_labels=num_output_labels,
        allow_multi_use=allow_multi_use,
        seed_start=seed_start,
        mutation_rate=mutation_rate,
    )

    best_udd: Optional[UnderlyingDataDistribution] = None
    best_analysis: Optional[Dict[str, float]] = None
    min_penalty = math.inf
    total_evals = 0
    generation = 0

    # We use a while loop to respect the max_iterations budget
    pbar = tqdm.tqdm(total=max_iterations, disable=not verbose, desc="Evolving UDDs")

    while total_evals < max_iterations:
        scored_population = []

        for dna in population_dna:
            if total_evals >= max_iterations:
                break

            total_evals += 1
            pbar.update(1)

            udd = UnderlyingDataDistribution.from_genetic_dna(dna)
            penalty, passes, analysis = get_udd_value_score(
                udd,
                max_ratio_diff,
                min_decisive,
                max_configurations=max_genetic_search_configurations,
                output_properties=output_properties,
            )

            # Check for immediate success
            if passes:
                pbar.close()
                if verbose:
                    print(f"\n✨ Solution found in Gen {generation} (Eval {total_evals})!")
                return udd, analysis, total_evals, generation

            scored_population.append((penalty, dna, udd, analysis))

            # Track global best-so-far
            if penalty < min_penalty:
                min_penalty = penalty
                best_udd = udd
                best_analysis = analysis

        # 2. Selection & Reproduction
        scored_population.sort(key=lambda x: x[0])

        # Print top score of this generation
        # if verbose:
        # top_penalty = scored_population[0][0]
        # print(f"Gen {generation} - Top Penalty: {top_penalty:.6f}")

        # Elitism: top 25%
        num_winners = max(1, population_size // 4)
        winners = scored_population[:num_winners]

        new_population = []
        new_population.extend([w[1] for w in winners])

        while len(new_population) < population_size:
            parent_dna = random.choice(winners)[1]
            child_dna = mutate_dna(parent_dna, mutation_rate=mutation_rate)
            new_population.append(child_dna)

        population_dna = new_population
        generation += 1

    pbar.close()
    return best_udd, best_analysis, total_evals, generation
