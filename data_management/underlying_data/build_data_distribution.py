# build_data_distribution.py
from data_management.underlying_data.underlying_data import Variable, Rule, UnderlyingDataDistribution, VariableType
from typing import Optional
import random
import logging


def build_data_distribution(
    num_variables: int,
    min_categories: int,
    max_categories: int,
    num_rules: int,
    allow_multi_use: bool = False,
    seed: Optional[int] = None,
    num_numerical_variables: int = 0,
    numerical_min_spans: int = 2,
    numerical_max_spans: int = 4,
    num_output_labels: int = 2,
) -> UnderlyingDataDistribution:
    if seed is not None:
        random.seed(seed)

    if num_variables < 1:
        raise ValueError("num_variables must be >= 1")
    if min_categories < 1:
        raise ValueError("min_categories must be >= 1")
    if max_categories < min_categories:
        raise ValueError("max_categories must be >= min_categories")
    if num_rules < 0:
        raise ValueError("num_rules must be >= 0")

    if num_numerical_variables < 0:
        raise ValueError("num_numerical_variables must be >= 0")
    if num_numerical_variables > num_variables:
        raise ValueError("num_numerical_variables cannot exceed num_variables")
    if numerical_min_spans < 1 or numerical_max_spans < numerical_min_spans:
        raise ValueError("Invalid numerical span bounds")
    if num_output_labels < 1:
        raise ValueError("num_output_labels must be >= 1")

    udd = UnderlyingDataDistribution(num_output_labels=num_output_labels)

    # Create variables
    var_names = []
    var_cat_counts = {}
    types = (
        [VariableType.NUMERICAL] * num_numerical_variables
        + [VariableType.CATEGORICAL] * (num_variables - num_numerical_variables)
    )
    random.shuffle(types)

    for i, vtype in enumerate(types):
        name = f"V{i}"
        if vtype is VariableType.NUMERICAL:
            spans = random.randint(numerical_min_spans, numerical_max_spans)
            var = Variable(
                name=name,
                categories=spans,
                num_output_labels=num_output_labels,
                variable_type=VariableType.NUMERICAL,
            )
        else:
            cats = random.randint(min_categories, max_categories)
            var = Variable(name=name, categories=cats, num_output_labels=num_output_labels)
            spans = cats
        udd.add_variable(var)
        var_names.append(name)
        var_cat_counts[name] = spans

    # Create rules
    used = set()
    rules_created = 0
    attempts = 0
    max_attempts = max(1000, num_rules * 20)

    def random_rule_pair():
        a, b = random.sample(var_names, 2)
        ca = random.randrange(var_cat_counts[a])
        cb = random.randrange(var_cat_counts[b])
        return a, ca, b, cb

    while rules_created < num_rules and attempts < max_attempts:
        attempts += 1
        va, ca, vb, cb = random_rule_pair()
        if not allow_multi_use:
            if (va, ca) in used or (vb, cb) in used:
                continue
        rname = f"R_{rules_created}"
        rule = Rule(name=rname, var_a=va, val_a=ca, var_b=vb, val_b=cb)
        try:
            udd.add_rule(rule)
        except ValueError:
            continue
        if not allow_multi_use:
            used.add((va, ca))
            used.add((vb, cb))
        rules_created += 1

    if rules_created < num_rules:
        logging.warning(
            "Requested %d rules, but only created %d (allow_multi_use=%s)",
            num_rules, rules_created, allow_multi_use
        )

    return udd


def _summarize(udd: UnderlyingDataDistribution) -> str:
    lines = []
    lines.append(f"Output labels: {udd.num_output_labels}")
    lines.append(f"Variables: {len(udd.variables)}")
    for name, var in udd.variables.items():
        if var.variable_type is VariableType.NUMERICAL:
            span_str = ", ".join(f"[{s:.2f}, {e:.2f}]" for s, e in var.spans)
            lines.append(
                f"  - {name}: type=numerical spans={span_str}"
            )
        else:
            lines.append(f"  - {name}: type=categorical categories={var.categories}")
    lines.append(f"Rules: {len(udd.rules)}")
    for r in udd.rules:
        def _describe(var_name: str, value: int) -> str:
            var = udd.variables.get(var_name)
            if var and var.variable_type is VariableType.NUMERICAL:
                span = var.span_for(value)
                start, end = span
                return f"{var_name} in [{start:.2f}, {end:.2f}]"
            return f"{var_name}=={value}"

        cond_a = _describe(r.var_a, r.val_a)
        cond_b = _describe(r.var_b, r.val_b)
        lines.append(
            f"  - {r.name}: IF {cond_a} AND {cond_b} "
            f"-> (w={r.weight:.3f}, s={r.score:.3f})"
        )
    return "\n".join(lines)


def main():
    # --- Default parameters ---
    num_variables = 3
    min_categories = 2
    max_categories = 3
    num_rules = 2
    allow_multi_use = False
    seed = 42

    udd = build_data_distribution(
        num_variables=num_variables,
        min_categories=min_categories,
        max_categories=max_categories,
        num_rules=num_rules,
        allow_multi_use=allow_multi_use,
        seed=seed,
    )

    print(_summarize(udd))

    return udd


if __name__ == "__main__":
    main()
