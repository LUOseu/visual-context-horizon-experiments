"""Expand the vision-overhead experiments into cells, lanes and stages.

A cell is one arm (AR or one drafter) for one workload slice.  AR and SD cells
that share ``pair_key`` use the same samples and repeat indices; the key omits
``method``, ``arm`` and ``visual_kv`` so switchable drafter visual-KV modes are
paired against the same full-vision AR run.  Timed stages alternate AR/SD order
across repeats, matching the frozen protocol.
"""
from __future__ import annotations

from collections import OrderedDict

from . import config as cfg
from . import workloads

GPUS = tuple(range(8))


def _cell(config: dict, manifest: dict, **fields) -> dict:
    engine = config["engine"]
    task = fields["task"]
    spec = cfg.task_spec(config, task) if task != "text_control" else {
        "track": "text_control", "dataset": "text_control"}
    track = fields.get("track") or spec.get("track") or "single_img"
    frames = int(fields.get("frames") or (1 if track != "multi_frame" else 1))
    block_width = int(fields.get("block_width") or 8)
    limit = (fields.get("samples") or {}).get("limit")
    cell = {
        "experiment": fields["experiment"],
        "target": fields["target"],
        "method": fields["method"],
        "arm": "ar" if fields["method"] == "ar" else "sd",
        "task": task,
        "track": track,
        "dataset": fields.get("dataset") or spec.get("dataset") or task,
        "budget": int(fields["budget"]),
        "frames": frames,
        "template": fields.get("template"),
        "block_width": block_width,
        "speculative_tokens": cfg.spec_tokens(config, block_width),
        "batch_size": int(fields.get("batch_size") or 1),
        "temperatures": list(fields.get("temperatures") or [0.0, 0.0]),
        "apc": bool(fields.get("apc") or False),
        "visual_kv": fields.get("visual_kv") or "full",
        "grouping": fields.get("grouping"),
        "tp": int(fields.get("tp") or 1),
        "repeats": int(fields.get("repeats") or config["protocol"]["repeats"]),
        "samples": {"limit": limit},
        "max_new_tokens": int(fields.get("max_new_tokens")
                               or engine["max_new_tokens"]),
        "profile": bool(fields.get("profile") or False),
        "prompt_logprobs": bool(fields.get("prompt_logprobs") or False),
        "phase": fields.get("phase") or "main",
    }
    cell["sample_count"] = len(workloads.select_samples(
        manifest, config, task, cell["budget"],
        frames=cell["frames"] if track == "multi_frame" else None,
        limit=limit))
    identity = dict(cell)
    identity.pop("sample_count", None)
    identity.pop("max_new_tokens", None)
    pair = dict(identity)
    for key in ("method", "arm", "visual_kv"):
        pair.pop(key, None)
    cell["pair_key"] = cfg.sha256_json(pair)[:16]
    cell["id"] = cfg.sha256_json(identity)[:16]
    return cell


def _wave_lanes(wave: dict) -> list:
    return list(wave.get("lanes") or [])


def _wave_methods(config: dict, wave: dict, experiment: str) -> list:
    """Union of methods the wave's lanes schedule for one experiment.

    Waves without explicit lanes (wave2_bonus) cover the full method matrix.
    """
    lanes = _wave_lanes(wave)
    if not lanes:
        return list(config["matrix"]["methods"])
    lanes = [lane for lane in lanes
             if experiment in (lane.get("experiments") or [])]
    if not lanes:
        return []
    methods = []
    for lane in lanes:
        for method in lane.get("methods") or [cfg.VISION_METHODS[0]]:
            if method not in methods:
                methods.append(method)
    return methods


def _usable(config: dict, methods: list) -> list:
    out = []
    for method in methods:
        if method == "ar":
            out.append(method)
            continue
        entry = config["targets"][cfg.VISION_TARGET]["drafters"].get(method) or {}
        if entry.get("runnable", True):
            out.append(method)
    return out


def _tasks(config: dict, exp: dict, wave: dict) -> list:
    tasks = wave.get("tasks") or exp.get("tasks") or []
    return [task for task in tasks if task in config["workloads"]["tasks"]]


def _budgets(config: dict, exp: dict, wave: dict, task: str) -> list:
    spec = config["experiments"][exp["name"]]
    base = wave.get("budgets") or spec.get("budgets") or config["workloads"]["budgets"]
    if task == "text_control":
        targets = [int(value) for value
                   in config["workloads"]["text_control"]["token_targets"]]
        return [int(value) for value in base if int(value) in targets]
    if wave.get("budgets"):
        # An explicit wave budget list is authoritative: the smoke wave needs
        # both a low and a high visual-context slice on the same task.
        return [int(value) for value in base]
    return cfg.task_budgets(config, task, base)


def _frames(config: dict, task: str) -> list:
    if task == "text_control":
        return [1]
    return cfg.task_frames(config, task)


def _slices(config: dict, exp: dict, wave: dict):
    """Yield (task, budget, frames) for an experiment."""
    tasks = _tasks(config, exp, wave)
    if exp.get("include_text_control") and not wave.get("tasks"):
        tasks = tasks + ["text_control"]
    for task in tasks:
        for budget in _budgets(config, exp, wave, task):
            for frames in _frames(config, task):
                yield task, budget, frames


def _exp1(config, manifest, exp, wave, arms, name):
    cells = []
    limit = _limit(wave, exp)
    for task, budget, frames in _slices(config, exp, wave):
        for method in arms:
            cells.append(_cell(
                config, manifest, experiment=name, target=cfg.VISION_TARGET,
                method=method, task=task, budget=budget, frames=frames,
                repeats=_repeats(config, exp, wave), samples={"limit": limit}))
    return cells


def _exp2(config, manifest, exp, wave, arms, name):
    modes = wave.get("modes") or exp.get("modes") or cfg.visual_kv_modes(config)
    repeats = _repeats(config, exp, wave)
    limit = _limit(wave, exp)
    cells = []
    for task, budget, frames in _slices(config, exp, wave):
        if "ar" in arms:
            cells.append(_cell(
                config, manifest, experiment=name, target=cfg.VISION_TARGET,
                method="ar", task=task, budget=budget, frames=frames,
                visual_kv="full", repeats=repeats, samples={"limit": limit}))
        for method in [m for m in arms if m != "ar"]:
            for mode in modes:
                cells.append(_cell(
                    config, manifest, experiment=name, target=cfg.VISION_TARGET,
                    method=method, task=task, budget=budget, frames=frames,
                    visual_kv=mode, repeats=repeats, samples={"limit": limit}))
    return cells


def _exp3(config, manifest, exp, wave, arms, name):
    batches = wave.get("batch_sizes") or exp.get("batch_sizes") or cfg.batch_sizes(config)
    repeats = _repeats(config, exp, wave)
    limit = _limit(wave, exp)
    cells = []
    for task, budget, frames in _slices(config, exp, wave):
        for batch in batches:
            for method in arms:
                cells.append(_cell(
                    config, manifest, experiment=name, target=cfg.VISION_TARGET,
                    method=method, task=task, budget=budget, frames=frames,
                    batch_size=batch, repeats=repeats, samples={"limit": limit}))
    rescue = exp.get("rescue") or {}
    rescue_methods = [m for m in rescue.get("methods") or []
                      if m in arms or m == "ar"]
    for task, budget, frames in _slices(config, exp, wave):
        for batch in rescue.get("batch_sizes") or []:
            for mode in rescue.get("modes") or []:
                for method in rescue_methods:
                    cells.append(_cell(
                        config, manifest, experiment=name + "_rescue",
                        target=cfg.VISION_TARGET, method=method, task=task,
                        budget=budget, frames=frames, batch_size=batch,
                        visual_kv=mode, repeats=repeats,
                        samples={"limit": limit}))
    return cells


def _exp4(config, manifest, exp, wave, arms, name):
    batches = wave.get("batch_sizes") or exp.get("batch_sizes") or [1]
    limit = _limit(wave, exp)
    cells = []
    for task, budget, frames in _slices(config, exp, wave):
        for batch in batches:
            for method in arms:
                cells.append(_cell(
                    config, manifest, experiment=name, target=cfg.VISION_TARGET,
                    method=method, task=task, budget=budget, frames=frames,
                    batch_size=batch, profile=True, repeats=_repeats(config, exp, wave),
                    samples={"limit": limit}))
    return cells


def _exp5(config, manifest, exp, wave, arms, name):
    repeats = _repeats(config, exp, wave)
    limit = _limit(wave, exp)
    cells = []
    tasks = _tasks(config, exp, wave)
    for task, budget, frames in _slices(config, exp, wave):
        for method in arms:
            cells.append(_cell(
                config, manifest, experiment=name, target=cfg.VISION_TARGET,
                method=method, task=task, budget=budget, frames=frames,
                repeats=repeats, samples={"limit": limit}))
    apc = exp.get("apc") or {}
    apc_tasks = [task for task in apc.get("tasks") or []
                 if task in config["workloads"]["tasks"]]
    for task in apc_tasks:
        for budget in apc.get("budgets") or []:
            for frames in _frames(config, task):
                for method in arms:
                    for apc_on in (False, True):
                        cells.append(_cell(
                            config, manifest, experiment=name + "_apc",
                            target=cfg.VISION_TARGET, method=method, task=task,
                            budget=budget, frames=frames, apc=apc_on,
                            repeats=repeats, samples={"limit": limit}))
    widths = (exp.get("block_widths") or {})
    for task in tasks:
        for width in widths.get("widths") or []:
            for budget in widths.get("budgets") or []:
                for method in arms:
                    cells.append(_cell(
                        config, manifest, experiment=name + "_width",
                        target=cfg.VISION_TARGET, method=method, task=task,
                        budget=budget, block_width=width, repeats=repeats,
                        samples={"limit": limit}))
    return cells


EXPERIMENT_ORDER = ("exp1_visual", "exp2_visual_kv", "exp3_concurrency",
                    "exp4_profile", "exp5_bonus")
BUILDERS = {
    "exp1_visual": _exp1, "exp2_visual_kv": _exp2, "exp3_concurrency": _exp3,
    "exp4_profile": _exp4, "exp5_bonus": _exp5,
}


def _repeats(config, exp, wave) -> int:
    return int(wave.get("repeats") or exp.get("repeats")
               or config["protocol"]["repeats"])


def _limit(wave, exp):
    limit = wave.get("samples_per_cell") or exp.get("samples")
    return int(limit) if limit else None


def _experiment_cells(config, manifest, name, wave):
    exp = dict(config["experiments"][name])
    exp["name"] = name
    if not exp.get("enabled", True):
        return []
    if not cfg.is_vision_suite(config):
        raise ValueError("the scheduler only expands vision-overhead-v1; "
                         "replay archived suites with their pinned checkout")
    arms = _usable(config, _wave_methods(config, wave, name))
    if not arms:
        return []
    return BUILDERS[name](config, manifest, exp, wave, arms, name)


def build_plan(config: dict, manifest: dict, wave_name: str) -> dict:
    wave = config["waves"][wave_name]
    experiments = list(wave.get("experiments") or [])
    if not experiments:
        for lane in wave.get("lanes") or []:
            for name in lane.get("experiments") or []:
                if name not in experiments:
                    experiments.append(name)
    cells = []
    for name in EXPERIMENT_ORDER:
        if name in experiments:
            cells.extend(_experiment_cells(config, manifest, name, wave))
    seen = set()
    for cell in cells:
        if cell["id"] in seen:
            raise ValueError("duplicate cell id %s (%s/%s)"
                             % (cell["id"], cell["experiment"], cell["method"]))
        seen.add(cell["id"])
    pruned = [cell for cell in cells if cell["sample_count"] == 0]
    cells = [cell for cell in cells if cell["sample_count"] > 0]
    groups = assign(cells, wave)
    skipped = cfg.blocked_methods(config)
    return {
        "wave": wave_name, "description": wave.get("description"),
        "cells": cells, "groups": groups, "skipped": skipped,
        "pruned": [{"id": cell["id"], "task": cell["task"], "dataset": cell["dataset"],
                    "budget": cell["budget"], "frames": cell["frames"]}
                   for cell in pruned],
        "cell_count": len(cells),
        "request_estimate": sum(cell["sample_count"] * cell["repeats"] for cell in cells),
        "gpus": list(wave.get("gpus") or GPUS),
    }


def _lane_for(cell: dict, wave: dict):
    for lane in wave.get("lanes") or []:
        experiments = lane.get("experiments") or []
        # rescue/apc/width slices belong to their parent experiment's lane
        if not any(cell["experiment"] == name
                   or cell["experiment"].startswith(name + "_")
                   for name in experiments):
            continue
        methods = lane.get("methods")
        if methods and cell["method"] not in methods:
            continue
        return tuple(lane["gpus"])
    raise ValueError("wave %s has no lane for %s/%s"
                     % (wave["description"], cell["experiment"], cell["method"]))


def assign(cells: list, wave: dict) -> list:
    groups = OrderedDict()
    if wave.get("lanes"):
        for cell in cells:
            lane = _lane_for(cell, wave)
            groups.setdefault(lane, []).append(cell)
    else:
        gpus = [int(gpu) for gpu in wave.get("gpus") or GPUS]
        load = {gpu: 0 for gpu in gpus}
        order = sorted(cells, key=lambda cell: (
            -cell["sample_count"] * cell["repeats"], cell["id"]))
        for cell in order:
            gpu = min(gpus, key=lambda item: (load[item], item))
            groups.setdefault((gpu,), []).append(cell)
            load[gpu] += cell["sample_count"] * cell["repeats"]
    out = []
    for lane, members in groups.items():
        out.append({
            "phase": members[0].get("phase") or "main", "gpus": list(lane),
            "cells": members,
            "cell_count": len(members),
            "request_estimate": sum(c["sample_count"] * c["repeats"] for c in members),
        })
    out.sort(key=lambda group: group["gpus"])
    return out


def stages(cells: list, config: dict) -> list:
    """Alternating fresh-engine stages: AR0, SD0, SD1, AR1, AR2, SD2."""
    if not cells:
        return []
    repeats = max(cell["repeats"] for cell in cells)
    by_arm = {"ar": OrderedDict(), "sd": OrderedDict()}
    for cell in cells:
        by_arm[cell["arm"]].setdefault(_engine_key(cell), []).append(cell)
    order = []
    for repeat in range(repeats):
        arms = ("ar", "sd") if repeat % 2 == 0 else ("sd", "ar")
        for arm in arms:
            for key, members in by_arm[arm].items():
                members = [cell for cell in members if repeat < cell["repeats"]]
                if members:
                    order.append({"repeat": repeat, "arm": arm,
                                  "engine_key": key, "cells": members})
    return order


def _engine_key(cell: dict) -> str:
    return "%s|%s|%s|%s|%s|%s|%s" % (
        cell["target"], cell["method"], cell["block_width"], cell["batch_size"],
        cell["apc"], cell["visual_kv"], cell["tp"])


def state_path(run_dir, cell: dict):
    from pathlib import Path
    return Path(run_dir) / "cells" / cell["id"]


def load_rows(run_dir, cell: dict) -> list:
    import json
    path = state_path(run_dir, cell) / "rows.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def completed_keys(run_dir, cell: dict) -> set:
    return {(row["repeat"], row["sample_id"]) for row in load_rows(run_dir, cell)
            if row.get("status") == "complete"}


def _samples(manifest, config, cell):
    return workloads.select_samples(
        manifest, config, cell["task"], cell["budget"],
        frames=cell["frames"] if cell["track"] == "multi_frame" else None,
        limit=cell["samples"].get("limit"))


def is_complete(run_dir, cell: dict, manifest: dict, config: dict) -> bool:
    expected = {(repeat, sample["id"]) for repeat in range(cell["repeats"])
                for sample in _samples(manifest, config, cell)}
    return expected.issubset(completed_keys(run_dir, cell))


def missing_work(run_dir, cell: dict, manifest: dict, config: dict) -> dict:
    samples = _samples(manifest, config, cell)
    done = completed_keys(run_dir, cell)
    out = {}
    for repeat in range(cell["repeats"]):
        missing = [sample for sample in samples if (repeat, sample["id"]) not in done]
        if missing:
            out[repeat] = missing
    return out
