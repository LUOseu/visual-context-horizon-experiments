"""Run one lane (parent) or one stage (child) of a wave.

The orchestration spawns one ``stage_worker`` per GPU lane.  That lane process
owns the GPU lock and telemetry, and spawns a **fresh subprocess per stage**
with ``--stage-index N``.  A stage is one arm, one repeat and one engine
configuration, so adapter bootstraps that must run before ``torch``/``vLLM``
imports (Domino) are isolated, and AR/SD never share allocator or module
state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfg
from . import engine as engine_mod
from . import lane as lane_mod
from . import profiling
from . import runtime
from . import schedule
from . import workloads


def load_frozen_config(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["config"] if "config" in payload else payload


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _error_row(cell, sample, repeat, gpu, status, message):
    return {
        "status": status, "cell_id": cell["id"], "pair_key": cell["pair_key"],
        "experiment": cell["experiment"], "target": cell["target"],
        "method": cell["method"], "arm": cell["arm"], "task": cell["task"],
        "track": cell["track"], "dataset": cell["dataset"],
        "budget": cell["budget"], "frames": cell["frames"],
        "block_width": cell["block_width"], "batch_size": cell["batch_size"],
        "apc": cell["apc"], "visual_kv": cell["visual_kv"],
        "drafter_visual_kv_mode": (cell["visual_kv"]
                                   if cell["arm"] == "sd" else "full"),
        "grouping": cell["grouping"], "tp": cell["tp"],
        "repeat": repeat, "phase": cell["phase"],
        "sample_id": sample["id"], "gpu": gpu,
        "oom_tp1": bool(status == "oom" and cell["tp"] == 1),
        "error": str(message)[:500], "timestamp": _stamp(),
    }


def _chunks(items, size):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _skip_state(state, retry_failed):
    status = state.get("status")
    if status == "complete":
        return True
    if status in ("failed", "oom") and not retry_failed:
        return True
    return False


def _pending(config, manifest, stage, run_dir, retry_failed):
    repeat = int(stage["repeat"])
    pending = []
    for cell in stage["cells"]:
        if _skip_state(lane_mod.read_state(run_dir, cell), retry_failed):
            continue
        missing = schedule.missing_work(run_dir, cell, manifest, config).get(repeat)
        if missing:
            pending.append((cell, missing))
    return pending


def _run_cell(engine, processor, SamplingParams, config, manifest, cell, samples,
              repeat, run_dir, gpu_label, profiler):
    lane_mod.write_state(run_dir, cell, status="running", pid=os.getpid(),
                         gpu=gpu_label, repeat=repeat, started=_stamp())
    done = 0
    try:
        for batch in _chunks(samples, cell["batch_size"]):
            rows = runtime.request_rows(
                engine.llm, processor, SamplingParams, config, cell, batch,
                repeat, gpu_label, profiler=profiler if cell["profile"] else None)
            lane_mod.append_rows(run_dir, cell, rows)
            done += len(rows)
    except Exception as error:
        status = "oom" if engine_mod.is_oom(error) else "failed"
        rows = [_error_row(cell, sample, repeat, gpu_label, status, error)
                for sample in samples[done:]]
        lane_mod.append_rows(run_dir, cell, rows)
        lane_mod.write_state(run_dir, cell, status=status, error=str(error)[:500],
                             oom_tp1=bool(status == "oom" and cell["tp"] == 1),
                             finished=_stamp())
        if status == "oom":
            raise engine_mod.OutOfMemory(str(error))
        return
    if schedule.is_complete(run_dir, cell, manifest, config):
        lane_mod.write_state(run_dir, cell, status="complete", finished=_stamp())
    else:
        lane_mod.write_state(run_dir, cell, status="partial", finished=_stamp())


def _record_stage_metadata(run_dir, meta):
    path = Path(run_dir) / "runtime_manifest.json"
    payload = {"stages": []}
    if path.is_file():
        try:
            payload = _load_json(path)
        except ValueError:
            payload = {"stages": []}
    payload.setdefault("stages", []).append(meta)
    lane_mod.write_json_atomic(path, payload)


def _run_stage_inprocess(config, manifest, exports, stage, run_dir, gpu_label,
                         retry_failed, processors, SamplingParams, prepared):
    cells = stage["cells"]
    repeat = int(stage["repeat"])
    pending = _pending(config, manifest, stage, run_dir, retry_failed)
    if not pending:
        return {"stage": "skipped", "repeat": repeat, "arm": stage["arm"],
                "engine_key": stage["engine_key"], "cells": len(cells)}
    engine = engine_mod.Engine(config, pending[0][0], exports, gpu_label,
                               prepared=prepared)
    started = time.time()
    profiler = profiling.StageProfiler(
        trace_dir=os.path.join(run_dir, "profile"),
        keep_traces=int(config["engine"].get("profile_keep_traces", 1)),
        enabled=any(cell["profile"] for cell, _ in pending))
    try:
        engine.start()
        meta = engine.metadata()
        meta.update({"lane": gpu_label, "repeat": repeat, "arm": stage["arm"],
                     "engine_key": stage["engine_key"], "wall_started": _stamp()})
        _record_stage_metadata(run_dir, meta)
        first_cell, first_missing = pending[0]
        processor = processors[first_cell["target"]]
        runtime.warmup(engine.llm, processor, SamplingParams, config, first_cell,
                       first_missing)
        engine.verify_visual_kv()
        for index, (cell, missing) in enumerate(pending):
            try:
                _run_cell(engine, processor, SamplingParams, config, manifest, cell,
                          missing, repeat, run_dir, gpu_label, profiler)
            except engine_mod.EngineError as error:
                for later_cell, later_missing in pending[index + 1:]:
                    status = "oom" if engine_mod.is_oom(error) else "failed"
                    for sample in later_missing:
                        lane_mod.append_rows(run_dir, later_cell, [
                            _error_row(later_cell, sample, repeat, gpu_label,
                                       status, "stage aborted: %s" % error)])
                    lane_mod.write_state(run_dir, later_cell, status=status,
                                         error=str(error)[:500])
                break
    finally:
        engine.close()
    return {"stage": "done", "repeat": repeat, "arm": stage["arm"],
            "engine_key": stage["engine_key"], "cells": len(pending),
            "seconds": round(time.time() - started, 1)}


def _stage_result_path(run_dir, lane_id, index):
    return Path(run_dir) / "lanes" / ("lane_%s.stage_%03d.json" % (lane_id, index))


def _run_stage_child(config, manifest, exports, lane, run_dir, lane_id, index,
                     retry_failed):
    stage = lane["stages"][index]
    gpus = lane["gpus"]
    gpu_label = ",".join(str(gpu) for gpu in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    pending = _pending(config, manifest, stage, run_dir, retry_failed)
    if not pending:
        result = {"stage": "skipped", "repeat": stage["repeat"],
                  "arm": stage["arm"], "engine_key": stage["engine_key"],
                  "cells": len(stage["cells"])}
        lane_mod.write_json_atomic(_stage_result_path(run_dir, lane_id, index), result)
        return 0
    # Adapter bootstrap must happen before torch/vLLM are imported.
    prepared = engine_mod.prepare_process(config, pending[0][0])
    import torch  # noqa: F401
    from transformers import AutoProcessor
    from vllm import SamplingParams
    processors = {}
    for cell, _missing in pending:
        name = cell["target"]
        if name not in processors:
            processors[name] = AutoProcessor.from_pretrained(
                config["targets"][name]["path"], trust_remote_code=True,
                local_files_only=True)
    try:
        result = _run_stage_inprocess(config, manifest, exports, stage, run_dir,
                                      gpu_label, retry_failed, processors,
                                      SamplingParams, prepared)
    except Exception as error:
        status = "oom" if engine_mod.is_oom(error) else "failed"
        for cell, missing in pending:
            for sample in missing:
                lane_mod.append_rows(run_dir, cell, [
                    _error_row(cell, sample, stage["repeat"], gpu_label, status, error)])
            lane_mod.write_state(run_dir, cell, status=status,
                                 error=str(error)[:500])
        result = {"stage": status, "repeat": stage["repeat"], "arm": stage["arm"],
                  "engine_key": stage["engine_key"], "cells": len(pending),
                  "error": str(error)[:500]}
        lane_mod.write_json_atomic(_stage_result_path(run_dir, lane_id, index), result)
        return 1
    lane_mod.write_json_atomic(_stage_result_path(run_dir, lane_id, index), result)
    return 0


def _run_lane(config, manifest, exports, lane, run_dir, lane_id, retry_failed,
              max_stages):
    gpus = lane["gpus"]
    telemetry = lane_mod.Telemetry(
        gpus, os.path.join(run_dir, "telemetry", "telemetry_gpu_%s.csv" % lane_id),
        interval=float(config["engine"]["telemetry_interval_s"]))
    telemetry.start()
    locks = lane_mod.acquire_locks(run_dir, gpus)
    results = []
    failed = 0
    try:
        stages = lane["stages"]
        if max_stages:
            stages = stages[:max_stages]
        for index, stage in enumerate(stages):
            command = [sys.executable, "-m", "matrix.stage_worker",
                       "--run-dir", str(run_dir), "--lane-id", lane_id,
                       "--stage-index", str(index)]
            if retry_failed:
                command.append("--retry-failed")
            process = subprocess.run(command, cwd=str(cfg.method_root()))
            result_path = _stage_result_path(run_dir, lane_id, index)
            result = (_load_json(result_path) if result_path.is_file()
                      else {"stage": "crashed", "returncode": process.returncode,
                            "repeat": stage["repeat"], "arm": stage["arm"],
                            "engine_key": stage["engine_key"],
                            "cells": len(stage["cells"])})
            result["returncode"] = process.returncode
            results.append(result)
            if process.returncode != 0:
                failed += 1
            lane_mod.write_json_atomic(
                Path(run_dir) / "lanes" / ("lane_%s.result.json" % lane_id),
                {"lane": ",".join(str(gpu) for gpu in gpus), "results": results,
                 "failed_stages": failed, "updated": _stamp()})
    finally:
        telemetry.stop()
        lane_mod.release_locks(locks)
    return 0 if failed == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--stage-index", type=int, default=None)
    parser.add_argument("--config", help="frozen config json (default: run-dir)")
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true",
                        default=bool(int(os.environ.get("VCH_RETRY_FAILED", "0"))))
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    config_path = args.config or os.path.join(run_dir, "config_resolved.json")
    config = load_frozen_config(config_path)
    manifest = _load_json(os.path.join(run_dir, "workload_manifest.json"))
    exports = _load_json(os.path.join(run_dir, "exports_manifest.json"))
    lane = _load_json(os.path.join(run_dir, "lanes", "lane_%s.json" % args.lane_id))
    if args.stage_index is not None:
        return _run_stage_child(config, manifest, exports, lane, run_dir,
                                args.lane_id, args.stage_index, args.retry_failed)
    return _run_lane(config, manifest, exports, lane, run_dir, args.lane_id,
                     args.retry_failed, args.max_stages)


if __name__ == "__main__":
    sys.exit(main())
