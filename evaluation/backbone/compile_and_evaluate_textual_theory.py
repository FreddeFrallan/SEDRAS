# evaluation/compile_and_evaluate_textual_theory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

from inference.model_wrappers.llm_wrapper import LLMWrapper
from evaluation.backbone.evaluate_on_samples import hard_evaluation

from data_management.utils import _normalize_samples


@dataclass
class InductionResult:
    theory_text: str
    classifier_fn: Any
    variable_order: List[str]
    schema: Dict[str, Dict[str, Any]]


def _zero_evaluation_report(
    *,
    num_output_labels: int,
    label_names: Dict[int, str],
    variable_order: List[str],
    output_properties: Optional[Dict[str, Any]],
    return_per_sample: bool,
) -> Dict[str, Any]:
    labels = list(range(num_output_labels))
    confusion_matrix = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    per_label_metrics = {
        lbl: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tnr": 0.0,
            "support": 0,
        }
        for lbl in labels
    }

    report: Dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "accuracy": 0.0,
        "full_accuracy": 0,
        "confusion_matrix": confusion_matrix,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "per_label": per_label_metrics,
        "balanced_accuracy": 0.0,
        "variable_order": variable_order,
    }

    if label_names is not None:
        report["label_names"] = label_names

    if return_per_sample:
        report["per_sample"] = []
        report["all_output_accuracy"] = 0.0
    else:
        report["all_output_accuracy"] = None

    return report


def _is_missing_code_block_error(error: Exception) -> bool:
    return "No code block found in LLM response." in str(error)



def _compile_theory_to_classifier(
    *,
    theory_text: str,
    variable_order: List[str],
    schema: Dict[str, Dict[str, Any]],
    num_output_labels: int,
    label_names: Dict[int, str],
    output_properties: Optional[Dict[str, Any]],
    llm: LLMWrapper,
) -> Tuple[Any, str]:
    """
    Ask the LLM to convert the textual theory into a deterministic Python function.

    Returns (callable_fn, code_text).
    """
    # Build a small recap that helps codegen but avoids leaking examples again.
    def _schema_lines() -> str:
        lines = ["Variable schema (integers are authoritative):"]
        for v in variable_order:
            entry = schema.get(v, {}) if isinstance(schema, dict) else {}
            values = entry.get("values", [])
            role = entry.get("role")
            labels = entry.get("labels")
            role_str = f" ({role})" if role else ""
            if labels:
                name_map = ", ".join(f"{k}:{labels.get(k, '')}" for k in values)
                lines.append(f"- {v}{role_str}: values={values}; names={{ {name_map} }}")
            else:
                lines.append(f"- {v}{role_str}: values={values}")
        return "\n".join(lines)

    label_options = ", ".join(
        f"{idx}:{label_names.get(idx, idx)}" for idx in range(num_output_labels)
    )

    return_signature = "def classify(x: list[float]) -> dict:\n"

    properties_lines: List[str] = [
        "- Return a dictionary with an integer entry label under the key \"label\".",
    ]
    if output_properties:
        prop_names = ", ".join(sorted(output_properties.keys()))
        properties_lines.extend([
            "- Also include integer predictions for these additional output properties (if any apply):",
            f"  {prop_names if prop_names else 'none'}.",
            "  Example: {'label': 0, 'prop_a': 1}",
        ])

    guidance = "".join([
        "Convert the following THEORY description into a deterministic Python classifier.\n",
        "Emit ONLY one function with EXACTLY this signature:\n",
        f"{return_signature}",
        "- x[i] is the value for the i-th variable in this fixed order:\n",
        f"  {variable_order}\n",
        "  - Categorical variables use integer category IDs.\n",
        "  - Numerical variables use floating-point values (e.g., spans normalized to [0, 1]).\n",
        f"- Return exactly one of these integer labels: {list(range(num_output_labels))} (label meanings: {label_options}).\n",
        ("\n".join(properties_lines) + "\n" if properties_lines else ""),
        "- No prints, no imports, no randomness.\n",
        "- Implement the logic faithfully; do not add heuristics not stated in the theory.\n\n",
        "- Make sure you interpret variables in the textual theory, and convert them to the correct variables and order.\n",
        f"{_schema_lines()}\n\n",
        "TEXTUAL THEORY START\n",
        f"{theory_text}\n",
        "TEXTUAL THEORY END\n\n",
        f"Now output ONLY valid Python code for {return_signature.strip()}: (no extra text).",
    ])
    # print(f"GUIDANCE for compiling theory to code:\n{guidance}\n")
    classifier_fn, classifier_txt = llm.make_call_to_python_code(guidance)
    return classifier_fn, classifier_txt


def _infer_output_properties(dataset_samples: List[Any]) -> Dict[str, Any]:
    """Infer the presence of extra output properties from the samples themselves."""

    property_names: set[str] = set()
    for sample in dataset_samples:
        if isinstance(sample, dict):
            labels = sample.get("property_labels") or {}
            scores = sample.get("property_scores") or {}
        else:
            labels = getattr(sample, "property_labels", {}) or {}
            scores = getattr(sample, "property_scores", {}) or {}

        property_names.update(labels.keys())
        property_names.update(scores.keys())

    return {name: None for name in sorted(property_names)} if property_names else {}


def compile_and_evaluate_theory(
    *,
    theory,  # InducedTheory from the caller (has theory_text, variable_order, schema, guidance_prompt)
    dataset_samples: List[Any],
    llm: LLMWrapper,
    num_output_labels: int,
    label_names: Dict[int, str],
    output_properties: Optional[Dict[str, Any]] = None,
    return_per_sample: bool = True,
) -> Tuple[InductionResult, Dict[str, Any], Callable[[List[int]], Any]]:
    """
    Compile a textual theory into a Python classifier and evaluate on provided samples.

      Inputs:
        - theory: an object with .theory_text, .variable_order, .schema
        - dataset_samples: list of DatasetSample objects or dict-like samples
        - llm: wrapper to compile theory to code
        - return_per_sample: include per-sample breakdown

    Returns:
      (InductionResult, report_dict)
      report_dict contains keys used elsewhere in the pipeline (accuracy, confusion, code, timings, etc.)
    """
    return compile_and_evaluate_textual_theory(
        theory_txt=theory.theory_text,
        variable_order=theory.variable_order,
        schema=theory.schema,
        dataset_samples=dataset_samples,
        llm=llm,
        num_output_labels=num_output_labels,
        label_names=label_names,
        output_properties=output_properties,
        return_per_sample=return_per_sample,
    )


def evaluate_compiled_theory(*,
    classifier_fn,
    theory_txt: str,
    classifier_txt: str,
    variable_order: List[str],
    schema: Dict[str, Dict[str, Any]],
    dataset_samples: List[Any],
    llm: LLMWrapper,
    num_output_labels: int,
    label_names: Dict[int, str],
    output_properties: Optional[Dict[str, Any]] = None,
    return_per_sample: bool = True,
):
    effective_output_properties = output_properties or _infer_output_properties(dataset_samples)

    if classifier_fn is None:
        report = _zero_evaluation_report(
            num_output_labels=num_output_labels,
            label_names=label_names,
            variable_order=variable_order,
            output_properties=effective_output_properties,
            return_per_sample=return_per_sample,
        )

        report["model"] = getattr(llm, "model", "unknown")
        report["variable_order"] = variable_order
        report["mode"] = "theory_then_code"
        report["theory_text"] = theory_txt
        report["classifier_code"] = None
        report["num_output_labels"] = num_output_labels
        report["label_names"] = label_names
        report["output_properties"] = (
            sorted(effective_output_properties.keys())
            if effective_output_properties
            else []
        )

        result = InductionResult(
            theory_text=theory_txt,
            classifier_fn=None,
            variable_order=variable_order,
            schema=schema,
        )
        return result, report, None

    # 2) Normalize samples to dicts
    norm_samples = _normalize_samples(dataset_samples)
    effective_output_properties = output_properties or _infer_output_properties(norm_samples)

    # 3) Evaluate
    report = hard_evaluation(
        samples=norm_samples,
        classifier=classifier_fn,
        variable_order=variable_order,
        num_output_labels=num_output_labels,
        label_names=label_names,
        return_per_sample=return_per_sample,
        output_properties=sorted(effective_output_properties.keys())
        if effective_output_properties
        else None,
    )

    # 4) Breadcrumbs/metadata
    report["model"] = getattr(llm, "model", "unknown")
    report["variable_order"] = variable_order
    report["mode"] = "theory_then_code"
    report["theory_text"] = theory_txt
    report["classifier_code"] = classifier_txt
    report["num_output_labels"] = num_output_labels
    report["label_names"] = label_names
    report["output_properties"] = sorted(effective_output_properties.keys()) if effective_output_properties else []
    # timing fields (theory_time / compilation_time) can be added by the caller

    result = InductionResult(
        theory_text=theory_txt,
        classifier_fn=classifier_fn,
        variable_order=variable_order,
        schema=schema,
    )
    return result, report, classifier_fn

def compile_and_evaluate_textual_theory(
    *,
    theory_txt: str,  # InducedTheory from the caller (has theory_text, variable_order, schema, guidance_prompt)
    variable_order: List[str],
    schema: Dict[str, Dict[str, Any]],
    dataset_samples: List[Any],
    llm: LLMWrapper,
    num_output_labels: int,
    label_names: Dict[int, str],
    output_properties: Optional[Dict[str, Any]] = None,
    return_per_sample: bool = True,
) -> Tuple[InductionResult, Dict[str, Any], Callable[[List[int]], Any]]:
    """
    Compile a textual theory into a Python classifier and evaluate on provided samples.

      Inputs:
        - theory: an object with .theory_text, .variable_order, .schema
        - dataset_samples: list of DatasetSample objects or dict-like samples
        - llm: wrapper to compile theory to code
        - return_per_sample: include per-sample breakdown

    Returns:
      (InductionResult, report_dict)
      report_dict contains keys used elsewhere in the pipeline (accuracy, confusion, code, timings, etc.)
    """
    # 1) Compile textual theory -> classifier code
    effective_output_properties = output_properties or _infer_output_properties(dataset_samples)

    try:
        classifier_fn, classifier_txt = _compile_theory_to_classifier(
            theory_text=theory_txt,
            variable_order=variable_order,
            schema=schema,
            num_output_labels=num_output_labels,
            label_names=label_names,
            output_properties=effective_output_properties,
            llm=llm,
        )
    except ValueError as exc:
        if _is_missing_code_block_error(exc):
            return evaluate_compiled_theory(
                classifier_fn=None,
                theory_txt=theory_txt,
                classifier_txt="",
                variable_order=variable_order,
                schema=schema,
                dataset_samples=dataset_samples,
                llm=llm,
                num_output_labels=num_output_labels,
                label_names=label_names,
                output_properties=effective_output_properties,
                return_per_sample=return_per_sample,
            )
        raise

    return evaluate_compiled_theory(
        classifier_fn=classifier_fn,
        theory_txt=theory_txt,
        classifier_txt=classifier_txt,
        variable_order=variable_order,
        schema=schema,
        dataset_samples=dataset_samples,
        llm=llm,
        num_output_labels=num_output_labels,
        label_names=label_names,
        output_properties=effective_output_properties,
        return_per_sample=return_per_sample,
    )

