from typing import Dict, List, Tuple, Optional, Any
from itertools import product
import random
from collections import Counter

from data_management.underlying_data.udd_search.find_balanced_udd import find_balanced_decisive_udd
from data_management.underlying_data.udd_search import UDDSearchMethod
from data_management.underlying_data.underlying_data import (
    Rule,
    UnderlyingDataDistribution,
    Variable,
    VariableType,
)
from data_management.dataset import DatasetSample, AbstractDataset, RepresentationLevel
from data_management.creation.dataset_creation_statistics import DatasetCreationStatistics
from data_management.underlying_data.output_properties import (
    OutputProperty,
    CategoricalOutputProperty,
    NumericalOutputProperty,
)


def _var_num_categories(var: Any) -> int:
    """
    Robustly infer the number of categories for a Variable.
    Supports:
      - int at var.categories
      - list/dict at var.categories
      - num_categories / n_categories / n_cats
      - to_dict() fallback with 'categories' as list/dict
    """
    cats = getattr(var, "categories", None)
    if isinstance(cats, int):
        return cats
    if isinstance(cats, (list, dict)):
        return len(cats)
    for attr in ("num_categories", "n_categories", "n_cats"):
        n = getattr(var, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    try:
        td = var.to_dict()
        if isinstance(td, dict) and "categories" in td:
            c = td["categories"]
            if isinstance(c, (list, dict)):
                return len(c)
    except Exception:
        pass
    # Fallback: assume binary
    return 2


def _sample_numerical_values(
    udd: UnderlyingDataDistribution, assignment: Dict[str, int]
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, cat in assignment.items():
        var = udd.variables.get(name)
        if var and var.variable_type is VariableType.NUMERICAL:
            values[name] = var.sample_value_for_category(int(cat))
    return values


def _label_from_scores(normalized_scores: Dict[int, float], sample_labels: bool = True) -> int:
    if not normalized_scores:
        return 0
    if not sample_labels:
        return max(normalized_scores, key=lambda lbl: (normalized_scores[lbl], -lbl))

    # Sample label proportional to normalized scores (fallback to uniform if all zeros)
    total = sum(max(v, 0.0) for v in normalized_scores.values())
    if total <= 0:
        return random.choice(list(normalized_scores.keys()))
    r = random.random() * total
    cumulative = 0.0
    for lbl, score in normalized_scores.items():
        cumulative += max(score, 0.0)
        if r <= cumulative:
            return lbl
    return max(normalized_scores, key=lambda lbl: (normalized_scores[lbl], -lbl))



def _enumerate_assignments(udd: UnderlyingDataDistribution) -> List[Dict[str, int]]:
    var_names = list(udd.variables.keys())
    if not var_names:
        return [{}]
    domains = []
    for n in var_names:
        k = _var_num_categories(udd.variables[n])
        domains.append(range(k))
    return [{n: v for n, v in zip(var_names, combo)} for combo in product(*domains)]


def create_balanced_dataset(
    main_name: str,
    save_path: str,
    *,
    num_samples: int,
    # UDD search constraints (defaults match your example)
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
    sample_labels: bool = False,
    seed: Optional[int] = None,
    search_method: UDDSearchMethod = UDDSearchMethod.REANDOM,
    max_genetic_search_configurations: Optional[int] = None,
    output_properties: Optional[List[OutputProperty]] = None,
) -> Tuple[AbstractDataset, Dict[str, float]]:
    """
    Finds a balanced & decisive UDD, then creates an AbstractDataset with:
      - raw_samples: all enumerated assignments scored & labeled
      - dataset_instances[instance_name]: a stratified-balanced subset of size ~num_samples

    Optional output properties:
      - Provide `CategoricalOutputProperty` instances to create an additional UDD per
        property with the same input variables but independent weights/rules. Each main
        output label randomly enables or disables each property; disabled properties
        yield None for that sample.
      - `NumericalOutputProperty` instances are recorded in metadata but not yet used
        for score generation; evaluation support will be added later.

    Args control both categorical and numerical variables. Numerical variables use
    equally-sized spans inside [0, 1], and their span selections are treated as
    categories during combinatorial enumeration. When DatasetSamples are created,
    each numerical variable receives a concrete value uniformly sampled within its
    assigned span, stored in DatasetSample.numerical_values.

    The actual save_path is derived from the provided `main_name` and key parameters.
    """
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)

    # 1) Find a UDD that meets balance/decisiveness constraints
    udd, analysis, _, num_iters = find_balanced_decisive_udd(
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
        seed_start=seed,
        search_method=search_method,
        max_genetic_search_configurations=max_genetic_search_configurations,
        output_properties=output_properties,
    )
    if udd is None or analysis is None:
        raise RuntimeError("Failed to find a suitable UDD under given constraints.")
    print("✅ Found UDD after", num_iters, "iterations")
    print("💾 Save path:", save_path)

    # 1b) Build optional output property UDDs that share the same input variables
    output_properties = output_properties or []
    categorical_props: Dict[str, int] = {}
    numerical_props: List[NumericalOutputProperty] = []
    for prop in output_properties:
        if isinstance(prop, CategoricalOutputProperty):
            if prop.name in categorical_props:
                raise ValueError(f"Duplicate output property name: {prop.name}")
            categorical_props[prop.name] = prop.num_categories
        elif isinstance(prop, NumericalOutputProperty):
            numerical_props.append(prop)
        else:
            raise TypeError(f"Unsupported output property type: {type(prop)!r}")

    property_udds: Dict[str, UnderlyingDataDistribution] = {}
    property_label_support: Dict[str, List[int]] = {}

    def _clone_variable_structure(var: Variable, num_labels: int) -> Variable:
        spans_copy = list(var.spans)
        return Variable(
            name=var.name,
            categories=var.categories,
            num_output_labels=num_labels,
            variable_type=var.variable_type,
            spans=spans_copy,
        )

    for prop_name, prop_categories in categorical_props.items():
        if prop_categories < 1:
            raise ValueError(f"Property {prop_name} must have at least one category")

        prop_udd = UnderlyingDataDistribution(num_output_labels=prop_categories)
        for var in udd.variables.values():
            prop_udd.add_variable(_clone_variable_structure(var, prop_categories))

        for idx, rule in enumerate(udd.rules):
            prop_rule = Rule(
                name=f"{rule.name}__{prop_name}_{idx}",
                var_a=rule.var_a,
                val_a=rule.val_a,
                var_b=rule.var_b,
                val_b=rule.val_b,
            )
            prop_udd.add_rule(prop_rule)

        property_udds[prop_name] = prop_udd

        availability = [bool(random.getrandbits(1)) for _ in range(num_output_labels)]
        if not any(availability):
            availability[random.randrange(num_output_labels)] = True
        property_label_support[prop_name] = [idx for idx, flag in enumerate(availability) if flag]

    # Numerical properties: track availability per label, even though they don't have a UDD.
    for prop in numerical_props:
        availability = [bool(random.getrandbits(1)) for _ in range(num_output_labels)]
        if not any(availability):
            availability[random.randrange(num_output_labels)] = True
        property_label_support[prop.name] = [idx for idx, flag in enumerate(availability) if flag]

    # 2) Enumerate all assignments and compute scores/labels
    assignments = _enumerate_assignments(udd)

    # 3) Randomly select a subset of samples with scores and labels, trying to balance the labels
    selected_assignments = []
    label_to_samples: Dict[int, List[Dict[str, object]]] = {
        lbl: [] for lbl in range(num_output_labels)
    }

    def _evaluate_properties(
        assignment: Dict[str, int],
        numerical_values: Dict[str, float],
        label: int,
    ) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[float]]]:
        prop_labels: Dict[str, Optional[int]] = {}
        prop_scores: Dict[str, Optional[float]] = {}
        for prop_name, prop_udd in property_udds.items():
            supported_labels = set(property_label_support.get(prop_name, []))
            if label not in supported_labels:
                prop_labels[prop_name] = None
                prop_scores[prop_name] = None
                continue

            prop_res = prop_udd.evaluate(assignment, numerical_values=numerical_values)
            prop_norm_scores = prop_res.get(
                "normalized_scores",
                {prop_res.get("predicted_label", 0): prop_res.get("normalized_score", 0.0)},
            )
            prop_label = _label_from_scores(prop_norm_scores, sample_labels=sample_labels)
            prop_labels[prop_name] = prop_label
            prop_scores[prop_name] = float(prop_norm_scores.get(prop_label, 0.0))

        # Numerical properties: generate a value when available for this label.
        for prop in numerical_props:
            supported_labels = set(property_label_support.get(prop.name, []))
            if label not in supported_labels:
                prop_labels[prop.name] = None
                prop_scores[prop.name] = None
                continue

            value = random.uniform(prop.min_value, prop.max_value)
            prop_labels[prop.name] = None
            prop_scores[prop.name] = float(value)

        return prop_labels, prop_scores
    for a in assignments:
        numerical_values = _sample_numerical_values(udd, a)
        res = udd.evaluate(a, numerical_values=numerical_values)
        normalized_scores = res.get("normalized_scores", {res.get("predicted_label", 0): res.get("normalized_score", 0.0)})
        label = _label_from_scores(normalized_scores, sample_labels=sample_labels)
        score = float(normalized_scores.get(label, 0.0))
        property_labels, property_scores = _evaluate_properties(a, numerical_values, label)
        info = {
            "assignment": a,
            "numerical_values": numerical_values,
            "score": score,
            "label": label,
            "property_labels": property_labels,
            "property_scores": property_scores,
        }
        label_to_samples.setdefault(label, []).append(info)

    base_take = num_samples // num_output_labels

    def _sample(pool: List[Dict[str, object]], k: int) -> List[Dict[str, object]]:
        if k <= len(pool):
            return random.sample(pool, k)
        out = pool.copy()
        while len(out) < k and pool:
            out.append(random.choice(pool))
        return out[:k]

    for lbl in range(num_output_labels):
        selected_assignments += _sample(label_to_samples.get(lbl, []), base_take)

    # Distribute remaining slots by drawing from the richest pools
    remaining = num_samples - len(selected_assignments)
    while remaining > 0:
        lbl = max(label_to_samples, key=lambda l: len(label_to_samples[l]))
        if not label_to_samples[lbl]:
            break
        selected_assignments.append(random.choice(label_to_samples[lbl]))
        remaining -= 1

    # Shuffle the selected assignments
    random.shuffle(selected_assignments)

    samples_all: List[DatasetSample] = []
    for info in selected_assignments:
        a = info["assignment"]
        numerical_values = info["numerical_values"]
        score = info["score"]
        label = info["label"]
        property_labels = info.get("property_labels", {})
        property_scores = info.get("property_scores", {})
        samples_all.append(
            DatasetSample(
                assignment=a,
                score=score,
                label=label,
                instance_text=None,
                instance_type=None,
                numerical_values=numerical_values,
                property_labels=property_labels,
                property_scores=property_scores,
            )
        )

    # 4) Stratified sampling to build a balanced dataset
    balanced_samples: List[DatasetSample] = samples_all[:num_samples]

    # 6) Build AbstractDataset with global metadata and one named instance
    property_metadata = {
        name: {
            "num_output_labels": prop_udd.num_output_labels,
            "available_for_labels": property_label_support.get(name, []),
            "type": "categorical",
        }
        for name, prop_udd in property_udds.items()
    }
    if numerical_props:
        for prop in numerical_props:
            property_metadata[prop.name] = {
                "type": "numerical",
                "min_value": prop.min_value,
                "max_value": prop.max_value,
                "available_for_labels": property_label_support.get(prop.name, list(range(num_output_labels))),
                **({"unit": prop.unit} if prop.unit else {}),
            }

    metadata = {
        "analysis": analysis,
        "save_path": save_path,
        "main_name": main_name,
        "num_output_labels": num_output_labels,
    }
    if property_metadata:
        metadata["output_properties"] = property_metadata

    abstract_ds = AbstractDataset(
        udd=udd,
        raw_samples=samples_all,                 # keep the full enumeration as raw
        metadata=metadata,         # global metadata
        output_properties=property_udds,
        property_label_support=property_label_support,
    )

    # Track basic creation metrics
    stats = DatasetCreationStatistics()
    stats.record_udd_search_iterations(num_iters)
    label_counts = Counter(s.label for s in balanced_samples)
    stats.record_samples_per_label(dict(label_counts))
    stats.record_total_number_of_assignments(len(assignments))
    stats.record_analysis_results(analysis)
    abstract_ds.creation_statistics = stats

    # Submit the number of samples per label
    num_samples_per_label = {lbl: len(samps) for lbl, samps in label_to_samples.items()}
    stats.record_total_samples_per_label(num_samples_per_label)
    print("Samples available per label:", num_samples_per_label)

    # Save dataset
    abstract_ds.save(save_path)

    abstract_ds.add_instance(
        "raw_samples",
        balanced_samples,
        level=RepresentationLevel.RAW,
        metadata={
            "note": "Stratified-balanced subset",
            "num_output_labels": num_output_labels,
            "size": len(balanced_samples),
        },
    )

    return abstract_ds, analysis
