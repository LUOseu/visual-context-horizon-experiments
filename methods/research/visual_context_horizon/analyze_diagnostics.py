"""Summarize profiling-lane CUDA event spans and native DFlash verifier counters."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .analyze import collect, load


def summarize(records):
    result = []
    for budget in sorted({key[0] for key in records}):
        for arm in ("ar", "sd"):
            rows = [row for key, row in records.items() if key[0] == budget and key[3] == arm]
            stages, counters = [], []
            for row in rows:
                diag = row["diagnostic"]
                worker, = diag["worker_cuda_peak"]
                gpu = worker["stage_cuda_ms"]
                if gpu is None:
                    raise ValueError("Timed diagnostic request lacks worker CUDA events")
                forward = gpu.get("target_forward", [])
                prefill = forward[0] if forward else None
                stages.append({
                    "visual_budget": budget, "arm": arm, "sample_id": row["id"],
                    "repeat": row["repeat"], "task": row["task"],
                    "renderer_and_mm_prepare_ms": diag["renderer_and_mm_preprocess_ns"] / 1e6,
                    "vision_encoder_ms": sum(gpu.get("vision_encoder", [])),
                    "initial_target_forward_ms": prefill,
                    "initial_target_forward_tokens_per_second": row["context_tokens"] * 1000 / prefill if prefill else None,
                    "later_target_forwards_ms": sum(forward[1:]),
                    "target_forward_calls": len(forward),
                    "sampler_ms": sum(gpu.get("sampler", [])),
                    "target_lm_head_ms": sum(gpu.get("target_lm_head", [])),
                    "draft_proposer_inclusive_ms": sum(gpu.get("draft_proposer", [])),
                    "draft_context_kv_ms": sum(gpu["draft_context_kv"]) if "draft_context_kv" in gpu else None,
                    "draft_forward_ms": sum(gpu["draft_forward"]) if "draft_forward" in gpu else None,
                    "runner_preprocess_inclusive_ms": sum(gpu.get("runner_preprocess_inclusive", [])),
                    "target_kv_reserved_bytes": worker["target_kv_reserved_bytes"],
                    "target_kv_pool_blocks": worker["target_kv_pool_blocks"],
                    "configured_kv_block_tokens": worker["configured_kv_block_tokens"],
                    "max_scheduler_kv_usage_fraction": diag["max_scheduler_kv_usage_fraction"],
                    "logical_prefill_context_tokens": row["context_tokens"],
                    "worker_peak_allocated_bytes": worker["allocated_bytes"],
                    "worker_peak_reserved_bytes": worker["reserved_bytes"],
                })
                rounds = diag["verification_rounds"]
                if rounds:
                    offered = [sum(step["proposed"] > index for step in rounds) for index in range(7)]
                    accepted = [sum(step["accepted_by_slot"][index] for step in rounds) for index in range(7)]
                    counters.append({
                        "id": row["id"], "repeats_index": row["repeat"],
                        "rounds": len(rounds), "offered_by_slot": offered, "accepted_by_slot": accepted,
                        "slot_acceptance_given_offered": [hit / count if count else None
                                                         for hit, count in zip(accepted, offered, strict=True)],
                        "mean_accepted_draft_tokens_per_round": mean(r["accepted_draft_tokens"] for r in rounds),
                        "committed_bonus_tokens": sum(r["committed_bonus_tokens"] for r in rounds),
                        "discarded_proposal_tokens": sum(r["discarded_proposal_tokens"] for r in rounds),
                        "events": rounds,
                    })
            result.append(dict(budget=budget, arm=arm, timings=stages, verifier=counters))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic_run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.diagnostic_run
    sweep = load(run / "sweep.json")
    if sweep["trace"] != "diagnostic" or sweep["dry_run"]:
        parser.error("Specify a completed instrumented native sweep")
    data = summarize(collect(run, sweep))
    args.out.mkdir(parents=True)
    with (args.out / "diagnostics.json").open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    rows = [row for item in data for row in item["timings"]]
    with (args.out / "diagnostic_stages.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
