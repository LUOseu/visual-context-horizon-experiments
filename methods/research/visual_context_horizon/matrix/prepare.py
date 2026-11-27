"""Prepare assets: verify pins, export drafters, materialize workloads, record runtime.

No weights are downloaded here; ``scripts/download_assets_8gpu.sh`` fills the
checkpoint cache first.  Exports keep the trained bytes unchanged: the vendor
exporter writes one ``model.safetensors`` plus ``config.json``.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfg
from . import workloads


def _sha(path) -> str:
    return cfg.sha256(Path(path))


def _run(command, cwd=None, env=None):
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], cwd=str(cwd) if cwd else None,
                   check=True, env=env)


def _vendor_env(vendor: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(vendor) + (os.pathsep + env["PYTHONPATH"]
                                       if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def verify_targets(config: dict) -> dict:
    report = {}
    for name, target in config["targets"].items():
        root = Path(target["path"])
        try:
            pins = cfg.target_file_pins(config, name)
        except FileNotFoundError:
            pins = {}
        missing, mismatch, checked = [], [], 0
        for filename, digest in pins.items():
            path = root / filename
            if not path.is_file():
                missing.append(filename)
                continue
            checked += 1
            if digest and _sha(path) != digest:
                mismatch.append(filename)
        report[name] = {"path": str(root), "pinned": len(pins), "checked": checked,
                        "missing": missing[:20], "mismatch": mismatch[:20],
                        "ok": bool(pins) and not missing and not mismatch}
    return report


def verify_data(config: dict) -> dict:
    paths = cfg.paths(config)
    missing = []
    legacy = cfg.repo_root() / "workloads" / "selected-500.json"
    if not legacy.is_file():
        missing.append(str(legacy))
    try:
        pins = cfg.data_pins()
    except FileNotFoundError:
        return {"ok": False, "missing": ["configs/matrix_data_files.json"]}
    for split in ("4k", "8k"):
        path = paths["cache"] / "hrbench" / ("hr_bench_%s.parquet" % split)
        if not path.is_file():
            missing.append(str(path))
    for entry in pins["mp_docvqa"]["files"]:
        path = paths["cache"] / "mp_docvqa" / Path(entry["path"]).name
        if not path.is_file():
            missing.append(str(path))
    video = pins["video_mme"]
    annotation = paths["cache"] / "video_mme" / Path(video["annotation_file"]).name
    if not annotation.is_file():
        missing.append(str(annotation))
    for entry in video["videos"]:
        path = paths["cache"] / "video_mme" / Path(entry["path"]).name
        if not path.is_file():
            missing.append(str(path))
        elif entry.get("size") and path.stat().st_size != int(entry["size"]):
            missing.append("%s (size %d != %d)" % (path, path.stat().st_size, entry["size"]))
    optional = {}
    for task in workloads.OPTIONAL_TASKS:
        manifest = paths["data"] / task / "raw_manifest.json"
        optional[task] = manifest.is_file()
    return {"ok": not missing, "missing": missing[:40], "optional_sources": optional}


def _checkpoint(config: dict, checkpoints: Path, target: str, method: str) -> Path:
    return checkpoints / target / method / "ckpt_final.pt"


def _export_drafter(config: dict, method: str, checkpoint: Path, out: Path) -> dict:
    adapter = cfg.adapter(config, cfg.VISION_TARGET, method)
    vendor = cfg.vendor_root()
    script = vendor / str(adapter.get("export") or "")
    if not script.is_file():
        raise FileNotFoundError(
            "export script missing for %s: %s (VCH_VELADRAFT_DIR=%s)"
            % (method, script, vendor))
    if (out / "model.safetensors").is_file() and (out / "config.json").is_file():
        print("export exists: %s" % out)
        return {"status": "cached"}
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("export directory is not empty: %s" % out)
    _run([sys.executable, script, checkpoint, out], cwd=vendor, env=_vendor_env(vendor))
    return {"status": "exported"}


def export_all(config: dict, checkpoints: Path, only=None, skip_missing=False) -> dict:
    checkpoints = Path(checkpoints).resolve()
    root = Path(cfg.paths(config)["exports"]).resolve()
    exports = {cfg.VISION_TARGET: {}}
    for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
        if only and method not in only:
            continue
        if not entry.get("runnable", True):
            exports[cfg.VISION_TARGET][method] = {
                "status": "blocked", "reason": entry.get("blocked"),
                "repository": entry.get("repo"), "revision": entry.get("revision")}
            print("drafter %s blocked: %s" % (method, entry.get("blocked")))
            continue
        checkpoint = _checkpoint(config, checkpoints, cfg.VISION_TARGET, method)
        if not checkpoint.is_file():
            message = "checkpoint missing: %s" % checkpoint
            if skip_missing:
                exports[cfg.VISION_TARGET][method] = {"status": "missing",
                                                      "reason": message}
                print("WARNING: " + message)
                continue
            raise FileNotFoundError(message + " (run scripts/download_assets_8gpu.sh)")
        expected = entry.get("sha256")
        if expected and _sha(checkpoint) != expected:
            raise ValueError("checkpoint SHA256 differs from its pin: %s" % checkpoint)
        out = root / cfg.VISION_TARGET / method
        state = _export_drafter(config, method, checkpoint, out)
        record = lock_entry(entry, checkpoint, out)
        record.update(state)
        exports[cfg.VISION_TARGET][method] = record
    manifest = {"suite": config["suite"], "generated": _now(),
                "exports": exports}
    cfg.save_json(root / "exports_manifest.json", manifest)
    return manifest


def lock_entry(entry: dict, checkpoint: Path, out: Path) -> dict:
    return {
        "path": str(out),
        "repository": entry.get("repo"), "revision": entry.get("revision"),
        "source_checkpoint": str(checkpoint),
        "source_sha256": entry.get("sha256"),
        "config_sha256": _sha(out / "config.json") if (out / "config.json").is_file() else None,
        "model_safetensors_sha256": (_sha(out / "model.safetensors")
                                     if (out / "model.safetensors").is_file() else None),
        "model_safetensors_size": ((out / "model.safetensors").stat().st_size
                                   if (out / "model.safetensors").is_file() else None),
    }


def lock_exports(config: dict) -> dict:
    root = Path(cfg.paths(config)["exports"]).resolve()
    exports = {cfg.VISION_TARGET: {}}
    for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
        if not entry.get("runnable", True):
            exports[cfg.VISION_TARGET][method] = {
                "status": "blocked", "reason": entry.get("blocked")}
            continue
        out = root / cfg.VISION_TARGET / method
        if not (out / "config.json").is_file():
            exports[cfg.VISION_TARGET][method] = {
                "path": str(out), "status": "missing",
                "repository": entry.get("repo"), "revision": entry.get("revision")}
            continue
        record = lock_entry(entry, Path(entry.get("checkpoint", "")), out)
        record["status"] = "locked"
        exports[cfg.VISION_TARGET][method] = record
    manifest = {"suite": config["suite"], "generated": _now(),
                "mode": "lock-from-local", "exports": exports}
    cfg.save_json(root / "exports_manifest.json", manifest)
    return manifest


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def materialize_workloads(config: dict, force: bool = False) -> dict:
    paths = cfg.paths(config)
    name = config["workloads"].get("manifest_name", "workloads_manifest.json")
    out = paths["manifests"] / name
    if out.is_file() and not force:
        return workloads.load_manifest(config, out)
    manifest = workloads.materialize(config)
    cfg.save_json(out, manifest)
    return workloads.load_manifest(config, out)


def runtime_manifest(config: dict) -> dict:
    vendor = cfg.vendor_root()
    packages = {}
    for name in ("torch", "transformers", "vllm", "safetensors", "pyarrow",
                 "PIL", "huggingface_hub"):
        try:
            packages[name] = importlib.metadata.version(
                {"PIL": "pillow"}.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = None
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:
        pass
    veladraft = {"dir": str(vendor), "head": None, "adapters": {}}
    if (vendor / ".git").is_dir():
        veladraft["head"] = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None
    for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
        adapter = entry.get("adapter") or {}
        files = {}
        for key in ("bootstrap", "export", "manifest"):
            relative = adapter.get(key)
            if not relative:
                continue
            path = vendor / relative
            files[key] = {"path": relative,
                          "sha256": _sha(path) if path.is_file() else None}
        veladraft["adapters"][method] = files
    hook = cfg.repo_root() / config["environment"]["visual_kv_hook"]
    pins = cfg.read_json(cfg.repo_root() / "provenance" /
                         "runtime_versions_8gpu.json") or {}
    hook_sha = _sha(hook) if hook.is_file() else None
    hook_pin = pins.get("visual_kv_hook_sha256")
    if hook_pin and hook_sha != hook_pin:
        raise SystemExit(
            "visual KV hook sha256 %s != pinned %s; update provenance only "
            "after reviewing the hook change" % (hook_sha, hook_pin))
    manifest = {
        "generated": _now(), "suite": config["suite"],
        "configured": config["environment"], "packages": packages, "cuda": cuda,
        "conda": os.environ.get("CONDA_DEFAULT_ENV"),
        "veladraft": veladraft,
        "rejection_sampler": {
            "sha256_expected": config["environment"].get(
                "vllm_rejection_sampler_sha256"),
            "status": "verified by scripts/setup_8gpu.sh"},
        "visual_kv_hook": {"path": config["environment"]["visual_kv_hook"],
                           "sha256": hook_sha, "sha256_expected": hook_pin,
                           "status": "verified" if hook_pin else "unpinned"},
        "cuda_graph_capture": "deferred: recorded per stage in the run runtime_manifest.json",
        "nvidia_smi": _nvidia_smi(),
    }
    cfg.save_json(Path(cfg.paths(config)["artifacts"]) / "runtime_manifest.json", manifest)
    cfg.save_json(Path(cfg.paths(config)["manifests"]) / "runtime_manifest.json", manifest)
    return manifest


def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines() if out.returncode == 0 else []
    except Exception:
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("verify", "workloads", "exports", "lock",
                                         "runtime", "all"))
    parser.add_argument("--config")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--methods")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-missing-drafters", action="store_true",
                        help="development only: record missing exports instead of failing")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    config = cfg.load_config(args.config)
    checkpoints = Path(args.checkpoints) if args.checkpoints else (
        Path(cfg.paths(config)["cache"]) / "checkpoints")
    only = [m for m in (args.methods or "").split(",") if m] or None
    report = {"step": args.step, "generated": _now()}
    if args.step in ("verify", "all"):
        report["targets"] = verify_targets(config)
        report["data"] = verify_data(config)
    if args.step in ("workloads", "all"):
        manifest = materialize_workloads(config, force=args.force)
        report["workloads"] = {"counts": manifest["counts"],
                               "recipe_sha256": manifest["recipe_sha256"],
                               "notes": manifest.get("notes")}
    if args.step in ("exports", "all"):
        report["exports"] = export_all(config, checkpoints, only=only,
                                       skip_missing=args.skip_missing_drafters)
    if args.step in ("lock", "all"):
        report["locked"] = lock_exports(config)
    if args.step in ("runtime", "all"):
        report["runtime"] = runtime_manifest(config)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.report:
        cfg.save_json(Path(args.report), report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
