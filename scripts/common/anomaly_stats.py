"""Metrics for window-level anomaly detection (standard library only).

Scores are per-window anomaly scores (higher = more anomalous), labels are
booleans (True = the window touches a labelled anomaly). Everything here is
deterministic and uses exact or ``math.fsum`` arithmetic, so the published
numbers can be recomputed anywhere from the score arrays.

- ``roc_auc``: the Mann-Whitney statistic, ties counted as one half (average
  ranks); equals the area under the ROC curve with ties handled the usual way.
- ``average_precision``: area under the precision-recall curve as the step sum
  over distinct score thresholds, ``sum((R_k - R_{k-1}) * P_k)`` (no
  interpolation; the definition scikit-learn uses).
- ``confusion`` / ``precision_recall_f1``: at a threshold, a window is flagged
  when its score is strictly greater than the threshold.
- ``spearman``: Pearson correlation of average ranks.
- ``events``: event-level detection of labelled anomaly ranges by flagged
  windows, and how early each was flagged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

THRESHOLD_RULE = "max_validation_score"
FLAG_RULE = "score > threshold"
ROC_METHOD = "mann_whitney_average_ranks"
PR_METHOD = "average_precision_step"
CORRELATION_METHOD = "spearman_average_ranks"


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, tied values getting the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = sum(1 for v in labels if v)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC needs both positive and negative windows")
    ranks = average_ranks(scores)
    rank_sum = math.fsum(r for r, v in zip(ranks, labels) if v)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = sum(1 for v in labels if v)
    if positives == 0:
        raise ValueError("average precision needs positive windows")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    terms, tp, fp, last_recall, i = [], 0, 0, 0.0, 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            if labels[order[j]]:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / positives
        terms.append((recall - last_recall) * (tp / (tp + fp)))
        last_recall, i = recall, j
    return math.fsum(terms)


def confusion(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for s, v in zip(scores, labels):
        flagged = s > threshold
        if flagged and v:
            tp += 1
        elif flagged:
            fp += 1
        elif v:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(c: dict[str, int]) -> dict[str, float]:
    """Precision, recall and F1 of confusion counts (precision 0 when nothing is flagged)."""
    flagged, positives = c["tp"] + c["fp"], c["tp"] + c["fn"]
    precision = c["tp"] / flagged if flagged else 0.0
    recall = c["tp"] / positives if positives else 0.0
    f1 = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) if c["tp"] else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("spearman needs two sequences of the same length (at least 2)")
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = math.fsum(ra) / len(ra), math.fsum(rb) / len(rb)
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.fsum((x - ma) ** 2 for x in ra)
    vb = math.fsum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb)


def events(starts: Sequence[int], flags: Sequence[bool], ranges: Sequence[tuple[int, int]],
           window: int) -> list[dict]:
    """Per labelled range [start, stop): the windows overlapping it, and the first one flagged.

    A window starting at ``s`` covers points ``s .. s + window - 1`` and can only
    be scored once its last point exists, so a flag on it is a detection at
    point ``s + window - 1``. ``delay_points`` is that point minus the start of
    the labelled range (0 = flagged as soon as the range began); the fraction
    is the delay over the length of the labelled range.
    """
    out = []
    for start, stop in ranges:
        overlapping = [k for k, s in enumerate(starts) if s + window > start and s < stop]
        flagged = [k for k in overlapping if flags[k]]
        entry = {"start": start, "stop": stop, "windows": len(overlapping),
                 "flagged_windows": len(flagged), "detected": bool(flagged),
                 "first_flagged_window_start": None, "detection_point": None,
                 "delay_points": None, "delay_fraction_of_range": None}
        if flagged:
            first = starts[flagged[0]]
            point = first + window - 1
            entry.update(first_flagged_window_start=first, detection_point=point,
                         delay_points=point - start,
                         delay_fraction_of_range=(point - start) / (stop - start))
        out.append(entry)
    return out
