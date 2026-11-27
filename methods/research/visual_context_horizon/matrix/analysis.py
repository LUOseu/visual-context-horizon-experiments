"""Paired speedups, bootstrap intervals, token match and the run summary."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from collections import OrderedDict
from pathlib import Path

from . import lane as lane_mod
from . import runtime
from . import schedule
from . import visual_kl

DIMENSIONS = ("experiment", "target", "task", "track", "dataset", "budget",
              "frames", "template", "block_width", "batch_size", "apc",
              "visual_kv", "grouping", "tp", "method")


def _rows(run_dir, cell):
    return schedule.load_rows(run_dir, cell)


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def bootstrap_median_ci(values, samples=2000, seed=42):
    if len(values) < 2:
        return (values[0] if values else None, values[0] if values else None)
    rng = random.Random(seed)
    medians = []
    for _ in range(int(samples)):
        medians.append(statistics.median(rng.choices(values, k=len(values))))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _ar_index(rows):
    """AR rows keyed by the paired slice; shared across drafter KV modes."""
    index = {}
    for row in rows:
        if row.get("status") != "complete" or row.get("arm") != "ar":
            continue
        index[(row.get("pair_key"), row.get("sample_id"), row.get("repeat"))] = row
    return index


def speedup_rows(rows, config):
    protocol = config["protocol"]
    index = _ar_index(rows)
    grouped = OrderedDict()
    for row in rows:
        if row.get("status") != "complete" or row.get("arm") != "sd":
            continue
        ar = index.get((row.get("pair_key"), row.get("sample_id"), row.get("repeat")))
        if not ar:
            continue
        ar_ms, sd_ms = ar.get("e2e_ms"), row.get("e2e_ms")
        if ar_ms is None or sd_ms is None or ar_ms <= 0:
            continue
        temperatures = "%s|%s" % (ar.get("temperature"), row.get("temperature"))
        key = tuple(row.get(dimension) for dimension in DIMENSIONS) + (temperatures,)
        grouped.setdefault(key, []).append(sd_ms / ar_ms)
    out = []
    for key, ratios in grouped.items():
        record = dict(zip(DIMENSIONS + ("temperatures",), key))
        low, high = bootstrap_median_ci(
            ratios, protocol.get("bootstrap_ci_samples", 2000),
            protocol.get("bootstrap_seed", 42))
        median = statistics.median(ratios)
        record.update({
            "n_pairs": len(ratios),
            "sd_over_ar_median": median,
            "sd_over_ar_mean": statistics.mean(ratios),
            "sd_over_ar_geomean": math.exp(statistics.mean(math.log(r) for r in ratios)),
            "p10": _percentile(ratios, 0.1),
            "p90": _percentile(ratios, 0.9),
            "ci_low": low, "ci_high": high,
            "speedup_median": (1.0 / median) if median else None,
            "break_even": bool(high is not None and high <= 1.0),
        })
        out.append(record)
    out.sort(key=lambda row: (row["experiment"], row["target"], row["method"],
                              row["task"] or "",
                              row["budget"] if row["budget"] is not None else -1,
                              row["batch_size"] or 1,
                              row["visual_kv"] or "", row["block_width"] or 0))
    return out


def token_match_rows(rows):
    index = _ar_index(rows)
    buckets = OrderedDict()
    for row in rows:
        if row.get("status") != "complete" or row.get("arm") != "sd":
            continue
        if float(row.get("temperature") or 0) != 0.0:
            continue
        ar = index.get((row.get("pair_key"), row.get("sample_id"), row.get("repeat")))
        if not ar:
            continue
        buckets.setdefault((row.get("pair_key"), row.get("method")), []).append(
            (ar, row))
    out = []
    for (pair_key, method), pairs in buckets.items():
        matched = compared = 0
        first_diff = None
        for ar, sd in pairs:
            compared += 1
            if ar.get("token_ids") == sd.get("token_ids"):
                matched += 1
            elif first_diff is None:
                first_diff = {"sample_id": sd.get("sample_id"),
                              "repeat": sd.get("repeat")}
            sd["token_match"] = bool(ar.get("token_ids") == sd.get("token_ids"))
        out.append({"pair_key": pair_key, "method": method,
                    "visual_kv": pairs[0][1].get("visual_kv"),
                    "batch_size": pairs[0][1].get("batch_size"),
                    "block_width": pairs[0][1].get("block_width"),
                    "compared": compared, "matched": matched,
                    "exact_match": (matched / compared) if compared else None,
                    "first_difference": first_diff})
    return out


def _wave0(plan, cells, matches, run_dir):
    reasons = []
    if any(cell["status"] in ("failed", "oom") for cell in cells):
        reasons.append("failed or OOM cells")
    if sum(1 for cell in cells if cell["status"] == "complete") != len(cells):
        reasons.append("incomplete cells")
    modes = {cell["visual_kv"] for cell in cells
             if cell["method"] != "ar" and cell["experiment"] == "exp2_visual_kv"}
    for mode in ("full", "drop_all"):
        if mode not in modes:
            reasons.append("no %s visual-KV cells" % mode)
            continue
        per_mode = [row for row in matches
                    if row.get("visual_kv") == mode]
        if not per_mode:
            reasons.append("no token-match rows for %s" % mode)
        elif any(row.get("exact_match") != 1.0 for row in per_mode):
            reasons.append("token mismatch in %s" % mode)
    manifest = cfg_read(run_dir / "runtime_manifest.json") or {}
    stages = manifest.get("stages") or []
    if not stages:
        reasons.append("runtime_manifest.json has no stages")
    elif not any(stage.get("cudagraph_mode") for stage in stages):
        reasons.append("cudagraph mode was not recorded")
    return {"pass": not reasons, "reasons": reasons}


def cfg_read(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def write_summary(run_dir, plan, lanes, results, config, wave_name):
    run_dir = Path(run_dir)
    all_rows, cells = [], []
    for cell in plan["cells"]:
        rows = _rows(run_dir, cell)
        all_rows.extend(rows)
        state = lane_mod.read_state(run_dir, cell)
        complete = {(row["repeat"], row["sample_id"]) for row in rows
                    if row.get("status") == "complete"}
        expected = cell["sample_count"] * cell["repeats"]
        status = state.get("status")
        if status not in ("failed", "oom") and expected and len(complete) >= expected:
            status = "complete"
        elif not status:
            status = "partial" if rows else "pending"
        cells.append({
            "cell_id": cell["id"], "experiment": cell["experiment"],
            "target": cell["target"], "method": cell["method"], "arm": cell["arm"],
            "task": cell["task"], "track": cell["track"],
            "dataset": cell["dataset"], "budget": cell["budget"],
            "frames": cell["frames"], "template": cell["template"],
            "block_width": cell["block_width"], "batch_size": cell["batch_size"],
            "apc": cell["apc"], "visual_kv": cell["visual_kv"],
            "grouping": cell["grouping"], "tp": cell["tp"],
            "repeats": cell["repeats"], "sample_count": cell["sample_count"],
            "expected_rows": expected, "rows": len(rows),
            "complete_rows": len(complete), "status": status,
            "error": state.get("error"),
        })
    speedups = speedup_rows(all_rows, config)
    matches = token_match_rows(all_rows)
    summary = {
        "wave": wave_name, "run_dir": str(run_dir),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": cfg_hash(config),
        "skipped_drafters": plan.get("skipped") or [],
        "cell_count": len(cells),
        "complete_cells": sum(1 for c in cells if c["status"] == "complete"),
        "failed_cells": sum(1 for c in cells if c["status"] in ("failed", "oom")),
        "row_count": len(all_rows),
        "lanes": results, "cells": cells, "speedups": speedups,
        "token_match": matches,
        "visual_kl": visual_kl.summarize_visual_kl(all_rows, 64),
    }
    if wave_name == "wave0_smoke":
        summary["wave0"] = _wave0(plan, cells, matches, run_dir)
    lane_mod.write_json_atomic(run_dir / "summary.json", summary)
    _write_csv(run_dir / "summary.csv", cells, speedups)
    return summary


def cfg_hash(config):
    from . import config as cfg
    return cfg.sha256_json({k: v for k, v in config.items() if not k.startswith("_")})


def _write_csv(path, cells, speedups):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for cell in cells:
            writer.writerow(["cell", cell["cell_id"], json.dumps(cell, sort_keys=True)])
        for row in speedups:
            writer.writerow(["speedup", "%s|%s|%s|%s" % (
                row["experiment"], row["method"], row["task"], row["visual_kv"]),
                json.dumps(row, sort_keys=True)])
