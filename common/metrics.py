"""Metrics used in both assignments, so all systems are scored the same way."""
from collections import Counter

from .task import CATEGORIES, FIELDS


def macro_f1(gold, pred, labels=CATEGORIES):
    scores = []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def score(gold_rows, preds, statuses=None, latencies=None):
    """gold_rows: dicts with category/urgency/escalate. preds: dicts or None (None = invalid output).

    exact_match = all three fields right. An invalid output counts as wrong on every field.
    """
    n = len(gold_rows)
    out = {"n": n}
    for f in FIELDS:
        out[f"{f}_acc"] = sum(p is not None and p[f] == g[f] for g, p in zip(gold_rows, preds)) / n
    out["exact_match"] = sum(p is not None and all(p[f] == g[f] for f in FIELDS) for g, p in zip(gold_rows, preds)) / n
    out["category_macro_f1"] = macro_f1([g["category"] for g in gold_rows], [(p or {}).get("category") for p in preds])
    tp = sum(g["escalate"] and p is not None and p["escalate"] for g, p in zip(gold_rows, preds))
    pos = sum(g["escalate"] for g in gold_rows)
    out["escalate_recall"] = tp / pos if pos else float("nan")
    out["invalid_rate"] = sum(p is None for p in preds) / n
    if statuses is not None:
        c = Counter(statuses)
        out["strict_json_rate"] = c.get("ok", 0) / n
    if latencies:
        out["avg_latency_s"] = sum(latencies) / len(latencies)
    return out


def auto_share_at(confidences, correct, target=0.95):
    """Decide automatically only when confidence >= a threshold and send the rest to a human.

    Returns the threshold with the largest automatic share whose accuracy is still >= target, as
    {threshold, n_auto, auto_share, auto_accuracy, human_share}, or None if no threshold reaches the target.
    Decisions with equal confidence cannot be separated by a threshold, so ties stay together.
    """
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    best, ok = None, 0
    for k, i in enumerate(order, 1):
        ok += bool(correct[i])
        if k < len(order) and confidences[order[k]] == confidences[i]:
            continue
        if ok / k >= target:
            best = {"threshold": confidences[i], "n_auto": k, "auto_share": k / len(order),
                    "auto_accuracy": ok / k, "human_share": 1 - k / len(order)}
    return best
