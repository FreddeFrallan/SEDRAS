from __future__ import annotations

from itertools import product
from typing import Dict, Tuple, Optional, Iterable, List
import math
import random

from data_management.underlying_data.udd_search.analyze_distribution_instance import analyze_distribution_instance
from data_management.underlying_data.output_properties import CategoricalOutputProperty, OutputProperty
from data_management.underlying_data.underlying_data import UnderlyingDataDistribution
from data_management.underlying_data.variables import Variable
from data_management.underlying_data.rule import Rule


def _decisiveness(scores: Dict[int, float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(scores.values(), reverse=True)
    best = ordered[0]
    second_best = ordered[1] if len(ordered) > 1 else 0.0
    margin = best - second_best
    return 0.0 if margin < 0.0 else (1.0 if margin > 1.0 else margin)


def _iter_assignments(
    udd: UnderlyingDataDistribution,
    max_configurations: Optional[int],
) -> Iterable[Dict[str, int]]:
    var_names = list(udd.variables.keys())
    if not var_names:
        return iter([{}])

    domain_sizes = [udd.variables[name].categories for name in var_names]
    total_combinations = math.prod(domain_sizes)

    def _combo_from_index(index: int, sizes: List[int]) -> List[int]:
        combo: List[int] = []
        for size in reversed(sizes):
            index, rem = divmod(index, size)
            combo.append(rem)
        combo.reverse()
        return combo

    if max_configurations is None or max_configurations <= 0 or total_combinations <= max_configurations:
        return ({n: c for n, c in zip(var_names, combo)} for combo in product(*[range(n) for n in domain_sizes]))

    indices = random.sample(range(total_combinations), k=max_configurations)
    return ({n: c for n, c in zip(var_names, _combo_from_index(idx, domain_sizes))} for idx in indices)


def _clone_variable_structure(var: Variable, num_labels: int) -> Variable:
    spans_copy = list(var.spans)
    return Variable(
        name=var.name,
        categories=var.categories,
        num_output_labels=num_labels,
        variable_type=var.variable_type,
        spans=spans_copy,
    )


def _build_property_udd(
    udd: UnderlyingDataDistribution,
    num_output_labels: int,
) -> UnderlyingDataDistribution:
    prop_udd = UnderlyingDataDistribution(num_output_labels=num_output_labels)
    for var in udd.variables.values():
        prop_udd.add_variable(_clone_variable_structure(var, num_output_labels))
    for idx, rule in enumerate(udd.rules):
        prop_rule = Rule(
            name=f"{rule.name}__prop_{num_output_labels}_{idx}",
            var_a=rule.var_a,
            val_a=rule.val_a,
            var_b=rule.var_b,
            val_b=rule.val_b,
        )
        prop_udd.add_rule(prop_rule)
    return prop_udd


def _property_balance_penalty(
    udd: UnderlyingDataDistribution,
    output_properties: Optional[Iterable[OutputProperty]],
    max_configurations: Optional[int],
) -> Tuple[float, Dict[str, float]]:
    if not output_properties:
        return 0.0, {}

    prop_imbalances: Dict[str, float] = {}
    for prop in output_properties:
        if not isinstance(prop, CategoricalOutputProperty):
            continue
        prop_udd = _build_property_udd(udd, prop.num_categories)
        prop_analysis = analyze_distribution_instance(
            prop_udd, threshold=0.0, max_configurations=max_configurations
        )
        prop_imbalances[prop.name] = float(
            prop_analysis.get("max_fraction_gap_from_uniform", 1.0)
        )

    if not prop_imbalances:
        return 0.0, {}

    avg_penalty = sum(prop_imbalances.values()) / len(prop_imbalances)
    return avg_penalty, prop_imbalances


def get_udd_value_score(udd: UnderlyingDataDistribution,
                        max_ratio_diff: float,
                        min_decisive: float,
                        max_configurations: Optional[int] = None,
                        output_properties: Optional[Iterable[OutputProperty]] = None,
                        property_weight: float = 0.5) -> Tuple[bool, float, Dict[str, float]]:
    analysis = analyze_distribution_instance(udd, threshold=0.0, max_configurations=max_configurations)

    # Compute MIN decisiveness across ALL combinations
    min_dec_seen = 1.0
    for assignment in _iter_assignments(udd, max_configurations):
        res = udd.evaluate(assignment)
        scores = res.get(
            "normalized_scores", {res.get("predicted_label", 0): res.get("normalized_score", 0.0)}
        )
        dec = _decisiveness(scores)
        if dec < min_dec_seen:
            min_dec_seen = dec
            if min_dec_seen < min_decisive:  # early break if already below required min
                break

    # Augment analysis dict with the min decisiveness we just computed
    analysis_aug = dict(analysis)
    analysis_aug["min_decisiveness_overall"] = min_dec_seen

    ratio_diff = float(analysis.get("max_fraction_gap_from_uniform", 1.0))
    prop_penalty, prop_imbalances = _property_balance_penalty(
        udd, output_properties, max_configurations
    )
    analysis_aug["property_imbalance_avg"] = prop_penalty
    analysis_aug["property_imbalances"] = prop_imbalances

    # Hard constraints
    if ratio_diff <= max_ratio_diff and min_dec_seen >= min_decisive:
        passes_constraints = True
    else:
        passes_constraints = False

    # Best-so-far by penalty:
    #  - Imbalance beyond limit
    #  - Shortfall of min decisiveness (most important part of the new constraint)
    # pen_imbalance = max(0.0, ratio_diff - max_ratio_diff)
    pen_imbalance = ratio_diff
    pen_min_dec = max(0.0, min_decisive - min_dec_seen)
    # penalty = pen_min_dec * 10.0 + pen_imbalance  # weight min decisiveness more heavily
    penalty = pen_min_dec * 1 + pen_imbalance + (property_weight * prop_penalty)

    return penalty, passes_constraints, analysis_aug
