"""Offline visual-KL on the Exp2 greedy companion cells.

Only the stored top-k prompt distributions are compared, so this is a
renormalized top-k divergence, not a full-vocabulary KL.  Rows are paired on
(sample_id, repeat, template, budget) between visual_kv=full and masked.
"""
from __future__ import annotations

import math
import statistics


def _probs(entry: dict) -> dict:
    values = {str(k): float(v) for k, v in (entry or {}).items()}
    total = sum(math.exp(v) for v in values.values())
    if total <= 0:
        return {}
    return {k: math.exp(v) / total for k, v in values.items()}


def kl_divergence(clean: dict, masked: dict) -> float:
    """KL(clean || masked) over the union, after per-side renormalization."""
    if not clean or not masked:
        return None
    p = _probs(clean)
    q = _probs(masked)
    total = 0.0
    for token, value in p.items():
        other = q.get(token)
        if other is not None and other > 0 and value > 0:
            total += value * math.log(value / other)
    return total


def kl_pair_metrics(clean: dict, masked: dict) -> dict:
    forward = kl_divergence(clean, masked)
    reverse = kl_divergence(masked, clean)
    if forward is None or reverse is None:
        return None
    return {"kl_clean_masked": forward, "kl_masked_clean": reverse,
            "js_proxy": 0.5 * (forward + reverse)}


def _key(row):
    return (row["sample_id"], row["repeat"], row.get("template"), row.get("budget"))


def summarize_visual_kl(rows: list, target_samples: int = 64) -> dict:
    full, masked = {}, {}
    for row in rows:
        topk = row.get("next_token_topk")
        if not topk:
            continue
        key = _key(row)
        if row.get("visual_kv") == "full":
            full[key] = topk
        elif row.get("visual_kv") == "masked":
            masked[key] = topk
    pairs = []
    per_budget = {}
    per_template = {}
    for key, clean in full.items():
        other = masked.get(key)
        if other is None:
            continue
        metrics = kl_pair_metrics(clean, other)
        if metrics is None:
            continue
        pairs.append(metrics)
        per_budget.setdefault(key[3], []).append(metrics["kl_clean_masked"])
        per_template.setdefault(key[2], []).append(metrics["kl_clean_masked"])
    counts = {}
    for sample_id, repeat, _template, _budget in full:
        counts.setdefault((sample_id, repeat), 0)
        counts[(sample_id, repeat)] += 1
    values = sorted(item["kl_clean_masked"] for item in pairs)
    summary = {
        "definition": "KL(clean||masked) over renormalized top-k prompt logprobs",
        "pairs": len(pairs),
        "unique_prompts": len(counts),
        "target_samples": int(target_samples),
        "median_kl": statistics.median(values) if values else None,
        "mean_kl": (sum(values) / len(values)) if values else None,
        "p90_kl": values[int(0.9 * (len(values) - 1))] if values else None,
        "by_budget": {str(k): _stats(v) for k, v in sorted(per_budget.items())},
        "by_template": {str(k): _stats(v) for k, v in sorted(per_template.items())},
    }
    return summary


def _stats(values):
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered),
            "mean": sum(ordered) / len(ordered)}
