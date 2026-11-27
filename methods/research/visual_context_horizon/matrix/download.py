"""Parallel, SHA-verified asset download for the 8-GPU suite.

Targets and drafter checkpoints are LFS blobs pinned by revision + SHA256;
dataset parquets carry SHA256 pins and the 50 Video-MME clips are pinned by
byte size.  ``--verify-only`` performs no network I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config as cfg

JOBS = 4


def _sha(path) -> str:
    return cfg.sha256(Path(path))


def _download(repo, filename, revision, local_dir, endpoint=None):
    from huggingface_hub import hf_hub_download
    kwargs = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    return Path(hf_hub_download(repo, filename, revision=revision,
                                local_dir=str(local_dir), **kwargs))


def _check(path: Path, sha: str = None, size: int = None) -> dict:
    if not path.is_file():
        return {"ok": False, "reason": "missing", "path": str(path)}
    if size is not None and path.stat().st_size != int(size):
        return {"ok": False, "reason": "size", "path": str(path)}
    if sha is not None and _sha(path) != sha:
        return {"ok": False, "reason": "sha256", "path": str(path)}
    return {"ok": True, "path": str(path)}


def plan_targets(config):
    jobs = []
    for name, target in config["targets"].items():
        root = Path(target["path"])
        for filename, digest in cfg.target_file_pins(config, name).items():
            jobs.append({"kind": "target", "target": name, "repo": target["repo"],
                         "revision": target["revision"], "filename": filename,
                         "sha256": digest, "dest": root / filename})
    return jobs


def plan_drafters(config):
    cache = Path(cfg.paths(config)["cache"]) / "checkpoints"
    jobs = []
    for name, target in config["targets"].items():
        for method, entry in target["drafters"].items():
            jobs.append({"kind": "drafter", "target": name, "method": method,
                         "repo": entry["repo"], "revision": entry["revision"],
                         "filename": entry["checkpoint"],
                         "sha256": entry.get("sha256"), "size": entry.get("size"),
                         "dest": cache / name / method / "ckpt_final.pt"})
    return jobs


def plan_datasets(config):
    paths = cfg.paths(config)
    cache = Path(paths["cache"])
    pins = cfg.data_pins()
    jobs = []
    for entry in pins["hrbench"]["files"]:
        jobs.append({"kind": "dataset", "repo": pins["hrbench"]["repo"],
                     "revision": pins["hrbench"]["revision"],
                     "filename": entry["path"], "sha256": entry.get("sha256"),
                     "size": entry.get("size"),
                     "dest": cache / "hrbench" / Path(entry["path"]).name})
    for entry in pins["mp_docvqa"]["files"]:
        jobs.append({"kind": "dataset", "repo": pins["mp_docvqa"]["repo"],
                     "revision": pins["mp_docvqa"]["revision"],
                     "filename": entry["path"], "sha256": entry.get("sha256"),
                     "size": entry.get("size"),
                     "dest": cache / "mp_docvqa" / Path(entry["path"]).name})
    video = pins["video_mme"]
    jobs.append({"kind": "dataset", "repo": video["annotation_repo"],
                 "revision": video["annotation_revision"],
                 "filename": video["annotation_file"],
                 "sha256": video.get("annotation_sha256"), "size": None,
                 "dest": cache / "video_mme" / Path(video["annotation_file"]).name})
    for entry in video["videos"]:
        jobs.append({"kind": "dataset", "repo": video["repo"],
                     "revision": video["revision"], "filename": entry["path"],
                     "sha256": entry.get("sha256"), "size": entry.get("size"),
                     "dest": cache / "video_mme" / Path(entry["path"]).name})
    return jobs


def _run_job(job, verify_only, staging):
    dest = Path(job["dest"])
    check = _check(dest, job.get("sha256"), job.get("size"))
    if check["ok"]:
        return {**check, "job": job, "state": "verified"}
    if verify_only:
        return {**check, "job": job, "state": "failed"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        fetched = _download(job["repo"], job["filename"], job["revision"],
                            staging / job["kind"], endpoint=os.environ.get("HF_ENDPOINT"))
        # hf_hub_download mirrors the repo-relative path; flatten to dest.
        target = dest
        if fetched.resolve() != target.resolve():
            shutil.move(str(fetched), str(target))
        check = _check(target, job.get("sha256"), job.get("size"))
        return {**check, "job": job, "state": "downloaded" if check["ok"] else "failed"}
    except Exception as error:
        return {"ok": False, "reason": str(error)[:300], "path": str(dest),
                "job": job, "state": "failed"}


def run(config, jobs, verify_only=False, workers=JOBS, staging=None):
    staging = Path(staging or (Path(cfg.paths(config)["cache"]) / "_staging"))
    results = []
    if not jobs:
        return results
    if verify_only or workers <= 1:
        for job in jobs:
            results.append(_run_job(job, verify_only, staging))
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job, verify_only, staging): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def summarize(results) -> dict:
    failed = [r for r in results if not r["ok"]]
    by_kind = {}
    for result in results:
        kind = result["job"]["kind"]
        by_kind.setdefault(kind, {"ok": 0, "failed": 0})
        by_kind[kind]["ok" if result["ok"] else "failed"] += 1
    return {"checked": len(results), "ok": len(results) - len(failed),
            "failed": len(failed), "by_kind": by_kind,
            "failures": [{"path": r["path"], "reason": r.get("reason"),
                          "repo": r["job"]["repo"],
                          "filename": r["job"]["filename"]} for r in failed[:30]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", JOBS)))
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    config = cfg.load_config(args.config)
    models = args.models or not (args.models or args.data)
    data = args.data or not (args.models or args.data)
    jobs = []
    if models:
        jobs.extend(plan_targets(config) + plan_drafters(config))
    if data:
        jobs.extend(plan_datasets(config))
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs),
                          "targets": len(plan_targets(config)),
                          "drafters": len(plan_drafters(config)),
                          "datasets": len(plan_datasets(config))}, indent=2))
        return 0
    started = time.time()
    results = run(config, jobs, verify_only=args.verify_only, workers=args.jobs)
    summary = summarize(results)
    summary["seconds"] = round(time.time() - started, 1)
    summary["verify_only"] = args.verify_only
    print(json.dumps(summary, indent=2))
    if args.report:
        cfg.save_json(Path(args.report), summary)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
