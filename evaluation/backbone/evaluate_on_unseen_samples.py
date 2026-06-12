from data_management.utils import _schema_from_dataset, _infer_variable_order
from evaluation.backbone import compile_and_evaluate_textual_theory
from data_management.underlying_data.underlying_data import UnderlyingDataDistribution, VariableType
from data_management import dataset
from data_management.creation.create_balanced_dataset import _enumerate_assignments
from inference.model_wrappers import llm_wrapper
from typing import Dict, List
import numpy as np


def generate_unseen_samples(
    udd: UnderlyingDataDistribution,
    samples: List[dataset.DatasetSample],
    max_new_samples: int = 1000,
    seed: int = 42,
) -> List[dataset.DatasetSample]:
    # Enumerate all assignments and compute scores/labels
    all_assignments = _enumerate_assignments(udd)

    def key_func(a: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in sorted(a.items()))

    existing_keys = set(key_func(s.assignment) for s in samples)

    new_samples = []
    for assignment in all_assignments:
        k = key_func(assignment)
        if k not in existing_keys:
            new_samples.append(assignment)

    if len(new_samples) > max_new_samples:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(new_samples), size=max_new_samples, replace=False)
        new_samples = [new_samples[i] for i in selected_indices]

    output_samples = []
    for assignment in new_samples:
        numerical_values = {
            name: udd.variables[name].sample_value_for_category(cat)
            for name, cat in assignment.items()
            if name in udd.variables and udd.variables[name].variable_type is VariableType.NUMERICAL
        }
        res = udd.evaluate(assignment, numerical_values=numerical_values)
        normalized_scores = res.get("normalized_scores", {res.get("predicted_label", 0): res.get("normalized_score", 0.0)})
        label = int(res.get("predicted_label", 0))
        score = float(normalized_scores.get(label, res.get("normalized_score", 0.0)))
        s = dataset.DatasetSample(
            assignment=assignment,
            score=score,
            label=label,
            instance_text=None,
            instance_type=None,
            numerical_values=numerical_values,
        )

        output_samples.append(s)

    return output_samples


def evaluate_on_unseen_samples(
    evaluation_func,
    llm: llm_wrapper.LLMWrapper,
    dataset_instance: dataset.DatasetInstance,
    existing_samples: List[dataset.DatasetSample],
    num_output_labels: int,
    label_names: Dict[int, str],
    max_new_samples: int = 1000,
    seed: int = 42,
):
    unseen_samples = generate_unseen_samples(
        udd=dataset_instance.udd,
        samples=existing_samples,
        max_new_samples=max_new_samples,
        seed=seed,
    )

    variable_order = _infer_variable_order(dataset_instance)
    schema = _schema_from_dataset(dataset_instance, variable_order)

    result, report, classifier_fn = compile_and_evaluate_textual_theory.evaluate_compiled_theory(
        classifier_fn=evaluation_func,
        theory_txt="",
        classifier_txt="",
        variable_order=variable_order,
        schema=schema,
        dataset_samples=unseen_samples,
        llm=llm,
        num_output_labels=num_output_labels,
        label_names=label_names,
        return_per_sample=True,
    )

    return result, report
