# evaluate_on_samples.py
from __future__ import annotations

from typing import Callable, Dict, List, Any, Iterable


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _compute_all_output_accuracy(
    per_sample: List[Dict[str, Any]], output_properties: List[str] | None
) -> float:
    """
    Computes accuracy over all outputs (main label + optional property labels).

    For each sample, we independently compare the predicted label and each
    property prediction against their corresponding ground truth values. When a
    ground truth property label is absent, the correct behavior is to avoid
    predicting that property (or set it to None).
    """

    if not per_sample:
        return 0.0

    correct = 0
    total = 0

    for sample in per_sample:
        property_labels = sample.get("property_labels") or {}
        raw_output = sample.get("raw_output") or {}

        # Main label accuracy (already tracked via "correct" flag)
        if "correct" in sample:
            total += 1
            correct += int(bool(sample["correct"]))

        # Property-level accuracy
        prop_keys = set(output_properties or [])
        prop_keys.update(property_labels.keys())
        prop_keys.update(k for k in raw_output.keys() if k != "label")

        for prop in prop_keys:
            total += 1
            label_val = property_labels.get(prop)
            pred_val = raw_output.get(prop) if isinstance(raw_output, dict) else None

            if label_val is None:
                # Correct if the model abstains (missing or explicit None)
                correct += int(prop not in raw_output or pred_val is None)
            else:
                correct += int(pred_val == label_val)

    return _safe_div(correct, total)


def hard_evaluation(
    samples: List[Dict[str, Any]],
    classifier: Callable[[List[int]], Any],
    *,
    variable_order: List[str] | None = None,
    num_output_labels: int | None = None,
    label_names: Dict[int, str] | None = None,
    return_per_sample: bool = False,
    output_properties: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a classifier on a list of samples with arbitrary categorical labels.

    Each sample must contain:
      - "assignment": Dict[str, int]   # variable -> category index
      - "label": int                   # ground truth label (0..N-1)

    The classifier must be a function:
        classifier(x: List[int]) -> dict
    where the list of ints corresponds to variable categories in `variable_order`.
    The integer label MUST be stored under the key "label"; any additional keys
    are ignored for the main accuracy calculation but are preserved in
    per-sample outputs when requested.

    Args:
        samples: list of samples (each with assignment + label)
        classifier: user-supplied function that maps [int,...] -> label
        variable_order: optional fixed ordering of variable names.
            If None, inferred from the first sample's assignment (sorted alphabetically).
        num_output_labels: optional number of labels. If provided, metrics are
            computed over the fixed label set range(num_output_labels) even if
            some labels are absent from the samples.
        label_names: optional mapping from label index to a human-readable name.
            This is stored in the returned report for convenience.
        return_per_sample: include per-sample outputs for inspection

    Returns:
        dict with accuracy, macro precision/recall/F1, balanced accuracy,
        per-label metrics, confusion matrix, and optionally per-sample results.
    """
    if not samples:
        raise ValueError("No samples provided for evaluation.")

    if variable_order is None:
        variable_order = sorted(samples[0]["assignment"].keys())

    label_set = set(range(num_output_labels)) if num_output_labels else set()
    total = 0
    correct = 0
    per_sample_out: List[Dict[str, Any]] = []

    # Pre-seed confusion matrix if label_set is known
    confusion_matrix: Dict[int, Dict[int, int]] = {
        true_label: {pred_label: 0 for pred_label in label_set}
        for true_label in label_set
    }

    INVALID_PRED_LABEL = -1

    for i, s in enumerate(samples):
        a = s.get("assignment")
        label = int(s.get("label", 0))

        if not isinstance(a, dict):
            raise TypeError(f"Sample {i} assignment must be dict, got {type(a)}")

        x = [a[v] for v in variable_order]
        raw_answer: Dict[str, Any] | None = None
        pred: int | None = None
        error: str | None = None

        try:
            raw_answer = classifier(x)
            if not isinstance(raw_answer, dict):
                raw_answer = {"label": raw_answer}

            if "label" not in raw_answer:
                raise ValueError(
                    "Classifier returned a dict but did not include a 'label' entry."
                )

            pred_val = raw_answer["label"]
            pred = int(pred_val)
        except Exception as exc:
            error = str(exc)
            pred = INVALID_PRED_LABEL
            raw_answer = None

        if label_set is not None:
            label_set.update([label, pred])
        if label not in confusion_matrix:
            confusion_matrix[label] = {}
        if pred not in confusion_matrix[label]:
            confusion_matrix[label][pred] = 0
        confusion_matrix[label][pred] += 1

        is_correct = int(pred == label)
        correct += is_correct
        total += 1

        if return_per_sample:
            per_sample_entry: Dict[str, Any] = {
                "x": x,
                "label": label,
                "pred": None if pred == INVALID_PRED_LABEL else pred,
                "correct": bool(is_correct),
                "property_labels": s.get("property_labels"),
            }

            per_sample_entry["raw_output"] = (
                raw_answer if raw_answer is not None else {"error": error}
            )

            per_sample_out.append(per_sample_entry)

    labels = sorted(label_set)

    # Ensure confusion matrix is dense for all observed/potential labels
    for true_label in labels:
        confusion_matrix.setdefault(true_label, {})
        for pred_label in labels:
            confusion_matrix[true_label].setdefault(pred_label, 0)

    per_label_metrics: Dict[int, Dict[str, float]] = {}
    recalls: List[float] = []
    precisions: List[float] = []
    f1s: List[float] = []
    supports: Dict[int, int] = {}

    for lbl in labels:
        tp = confusion_matrix[lbl][lbl]
        fp = sum(confusion_matrix[true][lbl] for true in labels if true != lbl)
        fn = sum(confusion_matrix[lbl][pred] for pred in labels if pred != lbl)
        tn = total - tp - fp - fn

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        tnr = _safe_div(tn, tn + fp)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports[lbl] = tp + fn

        per_label_metrics[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tnr": tnr,
            "support": supports[lbl],
        }

    accuracy = _safe_div(correct, total)
    macro_precision = _safe_mean(precisions)
    macro_recall = _safe_mean(recalls)
    macro_f1 = _safe_mean(f1s)
    balanced_accuracy = _safe_mean(recalls)
    full_accuracy = 1 if accuracy == 1.0 else 0
    # print(correct, total, accuracy)

    report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "full_accuracy": full_accuracy,
        "confusion_matrix": confusion_matrix,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label_metrics,
        "balanced_accuracy": balanced_accuracy,
        "variable_order": variable_order,
    }

    if label_names is not None:
        report["label_names"] = label_names

    if return_per_sample:
        report["per_sample"] = per_sample_out
        report["all_output_accuracy"] = _compute_all_output_accuracy(
            per_sample_out, output_properties
        )
    else:
        report["all_output_accuracy"] = None

    return report


# --- Example usage ---
if __name__ == "__main__":
    # Demo samples
    samples = [
        {"assignment": {"V0": 0, "V1": 2, "V2": 1}, "label": 1},
        {"assignment": {"V0": 1, "V1": 1, "V2": 0}, "label": 0},
        {"assignment": {"V0": 0, "V1": 0, "V2": 2}, "label": 1},
        {"assignment": {"V0": 2, "V1": 1, "V2": 0}, "label": 0},
    ]

    # Example classifier: return 1 if sum(x) > 2
    def simple_classifier(x: List[int]) -> int:
        return 1 if sum(x) > 2 else 0

    report = hard_evaluation(samples, simple_classifier, return_per_sample=True)
    from pprint import pprint
    pprint(report)
