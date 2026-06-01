"""
causalpol.evaluation.metrics
=============================
Evaluation metrics for benchmarking CausalPol against annotated corpora.

Implements the evaluation protocol used in Sections 5 and 6 of the paper:

* **Span-level F1** (Section 5.1) — evaluates extraction against gold
  annotations using partial-overlap and exact-match criteria.
* **Causal type macro-F1** (Section 5.3) — classifies claims into the
  four-way causal type taxonomy.
* **Epistemic status macro-F1** — classifies claims by evidential status.
* **Cohen's κ** (Section 6.1) — inter-rater agreement between GPT-4o
  plausibility assessments and expert panel labels.
* **Inconsistency detection precision/recall** (Section 6.2).

Functions
---------
span_f1(predictions, gold, overlap_threshold)
    Compute precision, recall, and F1 for causal span extraction.

cohens_kappa(labels_a, labels_b)
    Compute Cohen's κ for ordinal or nominal labels.

classification_report(predictions, gold, label_field)
    Compute per-class and macro-averaged precision, recall, F1 for
    causal type or epistemic status classification.

inconsistency_metrics(predicted_pairs, gold_pairs)
    Compute precision, recall, F1 for inconsistency detection.

compute_all_metrics(predictions, gold_annotations)
    Compute all metrics in one call for a held-out evaluation partition.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Span overlap utilities
# ---------------------------------------------------------------------------

def _span_overlap_ratio(pred_start: int, pred_end: int,
                         gold_start: int, gold_end: int) -> float:
    """Compute the overlap ratio between two character spans."""
    overlap_start = max(pred_start, gold_start)
    overlap_end = min(pred_end, gold_end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_len = overlap_end - overlap_start
    pred_len = pred_end - pred_start
    gold_len = gold_end - gold_start
    return overlap_len / max(pred_len, gold_len)


def span_f1(
    predictions: list[dict],
    gold: list[dict],
    overlap_threshold: float = 0.5,
) -> dict:
    """
    Compute span-level precision, recall, and F1 for causal extraction.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must have keys ``source_char_start``, ``source_char_end``,
        and optionally ``span_type`` (``"CAUSE"``, ``"EFFECT"``, ``"MECH"``).
    gold : list[dict]
        Gold annotations with the same structure.
    overlap_threshold : float
        Minimum character overlap ratio to count as a match (default 0.5,
        following the partial-overlap SemEval convention).

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``, ``n_predicted``,
        ``n_gold``, ``n_matched``.

    Notes
    -----
    The overall F1 reported in Table 1 of the paper uses span-level
    evaluation with overlap_threshold=0.5.  Exact-match F1 (threshold=1.0)
    is approximately 5–8 points lower across domains.
    """
    matched_pred = set()
    matched_gold = set()

    for i, pred in enumerate(predictions):
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            ratio = _span_overlap_ratio(
                pred["source_char_start"], pred["source_char_end"],
                g["source_char_start"], g["source_char_end"],
            )
            if ratio >= overlap_threshold:
                matched_pred.add(i)
                matched_gold.add(j)
                break

    tp = len(matched_pred)
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": len(predictions),
        "n_gold": len(gold),
        "n_matched": tp,
    }


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """
    Compute Cohen's κ for two sequences of categorical labels.

    Used to measure agreement between GPT-4o plausibility scores and expert
    panel labels (Section 6.1, κ = 0.68) and between human annotators
    (inter-rater agreement: causal type κ = 0.71, epistemic status κ = 0.68).

    Parameters
    ----------
    labels_a : list
        Labels from rater A.
    labels_b : list
        Labels from rater B.

    Returns
    -------
    float
        Cohen's κ in [-1, 1].  Values above 0.6 indicate substantial
        agreement (Landis & Koch, 1977).

    Raises
    ------
    ValueError
        If the two sequences have different lengths or are empty.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length.")
    if not labels_a:
        raise ValueError("Label sequences must not be empty.")

    n = len(labels_a)
    categories = list(set(labels_a) | set(labels_b))
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        conf[cat_index[a]][cat_index[b]] += 1

    # Observed agreement
    p_o = sum(conf[i][i] for i in range(k)) / n

    # Expected agreement
    row_sums = [sum(conf[i]) for i in range(k)]
    col_sums = [sum(conf[r][c] for r in range(k)) for c in range(k)]
    p_e = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(k))

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def classification_report(
    predictions: list[Any],
    gold: list[Any],
    label_field: str = "causal_type",
) -> dict:
    """
    Compute per-class and macro-averaged precision, recall, and F1.

    Parameters
    ----------
    predictions : list
        Predicted CausalClaim objects or dicts.
    gold : list
        Gold CausalClaim objects or dicts (same order as predictions).
    label_field : str
        Field to evaluate: ``"causal_type"`` or ``"epistemic_status"``.

    Returns
    -------
    dict
        Keys: per-class metrics dict, ``"macro_f1"``, ``"macro_precision"``,
        ``"macro_recall"``, ``"accuracy"``.
    """
    def get_label(item):
        if isinstance(item, dict):
            val = item.get(label_field)
        else:
            val = getattr(item, label_field, None)
        return str(val.value if hasattr(val, "value") else val)

    pred_labels = [get_label(p) for p in predictions]
    gold_labels = [get_label(g) for g in gold]

    classes = sorted(set(pred_labels) | set(gold_labels))
    per_class = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != cls and g == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    macro_p = sum(v["precision"] for v in per_class.values()) / max(len(per_class), 1)
    macro_r = sum(v["recall"] for v in per_class.values()) / max(len(per_class), 1)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)
    accuracy = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g) / max(len(gold_labels), 1)

    return {
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Inconsistency detection metrics
# ---------------------------------------------------------------------------

def inconsistency_metrics(
    predicted_pairs: list,
    gold_pairs: list,
    match_fn=None,
) -> dict:
    """
    Compute precision, recall, and F1 for inconsistency detection.

    Parameters
    ----------
    predicted_pairs : list[InconsistencyPair]
        Detected inconsistency pairs (both genuine and contextual).
    gold_pairs : list
        Gold inconsistency annotations.
    match_fn : callable, optional
        Function(pred, gold) → bool for determining if a predicted pair
        matches a gold annotation.  Default uses claim_id overlap.

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1``, ``n_genuine_predicted``,
        ``n_gold``.
    """
    genuine_pred = [p for p in predicted_pairs if p.is_genuine]

    if match_fn is None:
        def match_fn(pred, gold):
            pred_ids = {pred.claim_a.claim_id, pred.claim_b.claim_id}
            gold_ids = {gold.claim_a.claim_id, gold.claim_b.claim_id}
            return pred_ids == gold_ids

    matched = 0
    for pred in genuine_pred:
        for gold in gold_pairs:
            if match_fn(pred, gold):
                matched += 1
                break

    precision = matched / len(genuine_pred) if genuine_pred else 0.0
    recall = matched / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_genuine_predicted": len(genuine_pred),
        "n_gold": len(gold_pairs),
        "n_matched": matched,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pipeline_results: list,
    gold_annotations: list,
) -> dict:
    """
    Compute all evaluation metrics for a set of pipeline results against gold
    annotations.

    Parameters
    ----------
    pipeline_results : list[PipelineResult]
        Pipeline outputs for a held-out evaluation set.
    gold_annotations : list[dict]
        Gold annotations in PolCausal-50K format.  Each dict should have
        keys ``document_id``, ``claims`` (list of gold CausalClaim-like
        dicts), and ``inconsistency_pairs``.

    Returns
    -------
    dict
        Nested dict with keys ``extraction``, ``causal_type``,
        ``epistemic_status``, ``consistency``, and ``validation_kappa``
        (if expert labels are present).
    """
    all_pred_spans = []
    all_gold_spans = []
    all_pred_claims = []
    all_gold_claims = []
    all_pred_pairs = []
    all_gold_pairs = []

    gold_by_doc = {g["document_id"]: g for g in gold_annotations}

    for result in pipeline_results:
        gold = gold_by_doc.get(result.document_id, {})

        # Spans
        pred_spans = [
            {"source_char_start": c.source_char_start,
             "source_char_end": c.source_char_end}
            for c in result.claims
        ]
        gold_spans = gold.get("claims", [])
        all_pred_spans.extend(pred_spans)
        all_gold_spans.extend(gold_spans)

        # Claims for type classification
        all_pred_claims.extend(result.claims)
        all_gold_claims.extend(gold.get("claims", []))

        # Inconsistency pairs
        all_pred_pairs.extend(result.inconsistency_pairs)
        all_gold_pairs.extend(gold.get("inconsistency_pairs", []))

    metrics = {
        "extraction": span_f1(all_pred_spans, all_gold_spans),
    }

    if all_pred_claims and all_gold_claims:
        metrics["causal_type"] = classification_report(
            all_pred_claims, all_gold_claims, "causal_type"
        )
        metrics["epistemic_status"] = classification_report(
            all_pred_claims, all_gold_claims, "epistemic_status"
        )

    if all_pred_pairs:
        metrics["consistency"] = inconsistency_metrics(all_pred_pairs, all_gold_pairs)

    # Cohen's kappa for validation (if expert labels available)
    validated_with_expert = [
        c for result in pipeline_results for c in result.claims
        if c.validation_score is not None and c.expert_label is not None
    ]
    if validated_with_expert:
        llm_scores = [str(c.validation_score) for c in validated_with_expert]
        expert_labels = [c.expert_label for c in validated_with_expert]
        metrics["validation_kappa"] = cohens_kappa(llm_scores, expert_labels)

    return metrics
