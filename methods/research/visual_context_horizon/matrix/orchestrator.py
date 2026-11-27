"""Freeze a wave, split it into GPU lanes, launch workers, aggregate results."""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

from . import config as cfg
from . import schedule
from . import workloads


def _stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _lane_id(gpus) -> str:
    return "_".join(str(g) for g in gpus)


def _write_json(path, value):
    cfg.save_json(Path(path), value)


def _load_exports(config, path):
    if path:
        return cfg.read_json(path) or {}
    return cfg.read_json(cfg.paths(config)["exports"] / "exports_manifest.json") or {}


def _placeholder_exports(config) -> dict:
    return {"exports": {
        name: {method: {"path": "<pending prepare_8gpu.sh>", "sha256": None}
               for method in target["drafters"]}
        for name, target in config["targets"].items()}}


def _cell_lanes(plan) -> dict:
    out = {}
    for group in plan["groups"]:
        for cell in group["cells"]:
            out[cell["id"]] = tuple(group["gpus"])
    return out


def _lane_stages(plan, config) -> "OrderedDict":
    lanes = OrderedDict()
    cell_lane = _cell_lanes(plan)
    for stage in schedule.stages(plan["cells"], config):
        buckets = OrderedDict()
        for cell in stage["cells"]:
            buckets.setdefault(cell_lane[cell["id"]], []).append(cell)
        for gpus, cells in buckets.items():
            lane = lanes.setdefault(_lane_id(gpus), {"gpus": list(gpus), "stages": []})
            lane["stages"].append({"repeat": stage["repeat"], "arm": stage["arm"],
                                   "engine_key": stage["engine_key"], "cells": cells})
    return lanes


def prepare_run(config, manifest, exports, wave_name, run_dir):
    plan = schedule.build_plan(config, manifest, wave_name)
    run_dir = Path(run_dir)
    (run_dir / "lanes").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    lanes = _lane_stages(plan, config)
    frozen = cfg.freeze(config, extra={
        "wave": wave_name,
        "plan_sha256": cfg.sha256_json(plan["cells"]),
        "workload_recipe_sha256": manifest.get("recipe_sha256"),
        "workload_stub": bool(manifest.get("stub")),
    })
    _write_json(run_dir / "config_resolved.json", frozen)
    _write_json(run_dir / "plan.json", plan)
    _write_json(run_dir / "workload_manifest.json", manifest)
    _write_json(run_dir / "exports_manifest.json", exports)
    for name, lane in lanes.items():
        _write_json(run_dir / "lanes" / ("lane_%s.json" % name), lane)
    return plan, lanes


def dry_run(plan, lanes, config, wave_name, run_dir, manifest, commands=True):
    print("suite: %s  wave: %s" % (config["suite"], wave_name))
    print("run dir: %s%s" % (run_dir, "  (synthetic manifest)" if manifest.get("stub") else ""))
    print("cells: %d   request estimate: %d   gpus: %s" % (
        plan["cell_count"], plan["request_estimate"], plan["gpus"]))
    by_exp = OrderedDict()
    for cell in plan["cells"]:
        by_exp.setdefault(cell["experiment"], set()).add(cell["method"])
    for name, methods in by_exp.items():
        count = sum(1 for cell in plan["cells"] if cell["experiment"] == name)
        print("  %-20s methods=%-42s cells=%d"
              % (name, ",".join(sorted(methods)), count))
    print("lanes:")
    groups = {tuple(group["gpus"]): group for group in plan["groups"]}
    for lane_id, lane in sorted(lanes.items(), key=lambda item: tuple(item[1]["gpus"])):
        group = groups.get(tuple(lane["gpus"])) or {}
        exps = sorted({c["experiment"] for stage in lane["stages"]
                       for c in stage["cells"]})
        print("  lane %-4s gpus=%-5s stages=%3d cells=%4d requests=%6d  %s" % (
            lane_id, ",".join(str(g) for g in lane["gpus"]),
            len(lane["stages"]), group.get("cell_count", 0),
            group.get("request_estimate", 0), ",".join(exps)))
    if plan.get("pruned"):
        print("pruned cells (native pixels cannot reach the budget): %d"
              % len(plan["pruned"]))
    if plan.get("skipped"):
        for item in plan["skipped"]:
            print("blocked drafter: %s - %s" % (item["method"], item["reason"]))
    if not manifest.get("stub"):
        _check_images(plan, config, manifest)
    for name in config["targets"]:
        path = Path(config["targets"][name]["path"])
        print("  target %-10s %s" % (name, "present" if path.exists() else "MISSING: " + str(path)))
    if commands:
        print("lane command (per lane):")
        print("  PYTHONPATH=%s CUDA_VISIBLE_DEVICES=<gpus> %s -m matrix.stage_worker "
              "--run-dir %s --lane-id <lane>" % (
                  cfg.method_root(), sys.executable, run_dir))
        print("stage command (per stage): adds --stage-index N; a fresh process "
              "runs the adapter bootstrap before torch/vLLM import.")
    print("GPU checks (compile, graph capture, 0|0 match) need the real host; "
          "wave0_smoke verifies them.")


def _check_images(plan, config, manifest):
    seen, missing, total = set(), [], 0
    for cell in plan["cells"]:
        samples = workloads.select_samples(
            manifest, config, cell["task"], cell["budget"],
            frames=cell["frames"] if cell["track"] == "multi_frame" else None,
            limit=cell["samples"].get("limit"))
        for sample in samples:
            if sample["id"] in seen:
                continue
            seen.add(sample["id"])
            total += 1
            for path in workloads.missing_images(sample):
                missing.append(path)
    print("workload images: %d unique samples, %d missing files" % (total, len(missing)))
    for path in missing[:5]:
        print("  missing: %s" % path)


def _spawn(run_dir, lane_id, lane):
    env = os.environ.copy()
    root = str(cfg.method_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["VCH_REPO_ROOT"] = str(cfg.repo_root())
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in lane["gpus"])
    log_path = Path(run_dir) / "logs" / ("lane_%s.log" % lane_id)
    handle = open(log_path, "ab")
    command = [sys.executable, "-m", "matrix.stage_worker", "--run-dir",
               str(run_dir), "--lane-id", lane_id]
    process = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                               stderr=subprocess.STDOUT)
    return {"lane": lane_id, "process": process, "log": str(log_path), "handle": handle}


def launch(run_dir, lanes):
    results = []
    running = [_spawn(run_dir, name, lanes[name]) for name in lanes]
    for job in running:
        code = job["process"].wait()
        job["handle"].close()
        results.append({"lane": job["lane"], "returncode": code, "log": job["log"]})
    return results


def _resolve_run_dir(config, wave_name, requested, resume):
    root = Path(cfg.paths(config)["runs"])
    if requested:
        path = Path(requested)
        if path.exists() and not resume and any(path.iterdir()):
            path = path.with_name(path.name + "_" + _stamp())
        return path
    if resume:
        latest = sorted(glob.glob(str(root / (wave_name + "*"))))
        if latest:
            return Path(latest[-1])
    return root / (wave_name + "_" + _stamp())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--exports")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-commands", action="store_true")
    parser.add_argument("--skip-missing-drafters", action="store_true",
                        help="development only: allow missing exports for drafters")
    args = parser.parse_args(argv)

    config = cfg.load_config(args.config)
    if args.wave not in config["waves"]:
        raise SystemExit("unknown wave %r; known: %s" % (
            args.wave, ", ".join(config["waves"])))
    run_dir = _resolve_run_dir(config, args.wave, args.run_dir, args.resume)

    manifest_path = args.manifest or (
        Path(cfg.paths(config)["manifests"]) /
        config["workloads"].get("manifest_name", "workloads_manifest.json"))
    try:
        manifest = workloads.load_manifest(config, manifest_path)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        manifest = workloads.stub_manifest(config)
        print("warning: %s missing; dry-run uses a synthetic manifest" % manifest_path)
    exports = _load_exports(config, args.exports)
    if not exports:
        exports = _placeholder_exports(config)
        if not args.dry_run and not args.skip_missing_drafters:
            raise SystemExit("exports manifest missing; run scripts/prepare_8gpu.sh")
        if args.skip_missing_drafters:
            print("warning: --skip-missing-drafters; unexported drafters will fail")

    if run_dir.exists() and not args.resume:
        if any(run_dir.iterdir()):
            raise SystemExit("%s already exists; pass --resume or a fresh --run-dir" % run_dir)
    plan, lanes = prepare_run(config, manifest, exports, args.wave, run_dir)
    if args.dry_run:
        dry_run(plan, lanes, config, args.wave, run_dir, manifest,
                commands=not args.no_commands)
        return 0

    (run_dir / "frozen").mkdir(exist_ok=True)
    versions = (cfg.read_json(cfg.repo_root() / "provenance" /
                              "runtime_versions_8gpu.json")
                or {"status": "missing"})
    _write_json(run_dir / "frozen" / "versions.json", versions)
    print("run dir: %s" % run_dir)
    results = launch(run_dir, lanes)
    summary = aggregate(run_dir, plan, lanes, results, config, args.wave)
    print("summary: cells=%d complete=%d failed=%d rows=%d" % (
        summary["cell_count"], summary["complete_cells"],
        summary["failed_cells"], summary["row_count"]))
    failed = [r for r in results if r["returncode"] != 0]
    if failed:
        for item in failed:
            print("lane %s exited %d; log: %s" % (
                item["lane"], item["returncode"], item["log"]))
        return 2
    if args.wave == "wave0_smoke":
        checks = summary.get("wave0") or {}
        if checks.get("pass"):
            print("[WAVE0 PASS] lossless 0|0 for full and drop_all; graphs recorded")
            return 0
        print("[WAVE0 FAIL] %s" % "; ".join(checks.get("reasons") or ["unknown"]))
        return 3
    return 0


def aggregate(run_dir, plan, lanes, results, config, wave_name):
    from . import analysis
    return analysis.write_summary(run_dir, plan, lanes, results, config, wave_name)


if __name__ == "__main__":
    sys.exit(main())
