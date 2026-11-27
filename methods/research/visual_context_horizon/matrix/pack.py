"""Stage one wave's results for upload: aggregate, compress telemetry, strip payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

HEAVY_SUFFIXES = (".pt", ".safetensors", ".bin", ".ckpt", ".jpg", ".jpeg", ".png",
                  ".gif", ".mp4", ".parquet", ".arrow", ".trace.json")
SECRET_HINTS = ("token.txt", ".env", "id_rsa", ".pem", "credentials")


def _sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage(run_dir: Path, wave: str, out_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    copied = []
    for name in ("summary.json", "summary.csv", "plan.json", "runtime_manifest.json",
                 "config_resolved.json", "workload_manifest.json"):
        if (run_dir / name).is_file():
            _copy(run_dir / name, out_dir / name)
            copied.append(name)
    # Aggregated matrix summary: the per-wave summary already contains the
    # paired speedup table; expose it under the stable aggregate name.
    if (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text())
        suite = "vision-overhead-v1"
        resolved = run_dir / "config_resolved.json"
        if resolved.is_file():
            try:
                suite = json.loads(resolved.read_text()).get("suite") or suite
            except ValueError:
                pass
        aggregate = {"suite": suite, "waves": [{
            "wave": wave, "generated": summary.get("generated"),
            "cell_count": summary.get("cell_count"),
            "complete_cells": summary.get("complete_cells"),
            "row_count": summary.get("row_count"),
            "speedups": summary.get("speedups", []),
            "token_match": summary.get("token_match", []),
            "visual_kl": summary.get("visual_kl", {}),
        }]}
        (out_dir / "summary_matrix.json").write_text(json.dumps(aggregate, indent=2))
        copied.append("summary_matrix.json")
        _matrix_csv(summary, out_dir / "summary_matrix.csv")
        copied.append("summary_matrix.csv")
    telemetry_dir = out_dir / "telemetry"
    if (run_dir / "telemetry").is_dir():
        shutil.copytree(run_dir / "telemetry", telemetry_dir)
        copied.append("telemetry/")
    lanes = out_dir / "lanes"
    if (run_dir / "lanes").is_dir():
        shutil.copytree(run_dir / "lanes", lanes,
                        ignore=shutil.ignore_patterns("*.json"))
        for path in sorted((run_dir / "lanes").glob("*.result.json")):
            _copy(path, lanes / path.name)
        copied.append("lanes/")
    logs = out_dir / "logs"
    if (run_dir / "logs").is_dir():
        shutil.copytree(run_dir / "logs", logs)
        copied.append("logs/")
    telemetry_tar = out_dir / "per_request_telemetry.tar.gz"
    _telemetry_tar(run_dir, telemetry_tar)
    copied.append(telemetry_tar.name)
    removed = _strip(out_dir)
    manifest = {"wave": wave, "source_run": str(run_dir),
                "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": copied, "stripped": removed}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _sha256sums(out_dir)
    return manifest


def _matrix_csv(summary, path):
    import csv
    rows = summary.get("speedups", [])
    keys = sorted({key for row in rows for key in row}) or ["experiment"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _telemetry_tar(run_dir: Path, path: Path):
    with tarfile.open(path, "w:gz") as archive:
        for rows in sorted((run_dir / "cells").glob("*/rows.jsonl")):
            archive.add(rows, arcname="cells/%s/rows.jsonl" % rows.parent.name)
        for state in sorted((run_dir / "cells").glob("*/state.json")):
            archive.add(state, arcname="cells/%s/state.json" % state.parent.name)


def _strip(root: Path) -> list:
    removed = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (path.suffix.lower() in HEAVY_SUFFIXES
                or any(hint in name for hint in SECRET_HINTS)
                or "profile" in str(path.relative_to(root)).lower()):
            removed.append(str(path.relative_to(root)))
            path.unlink()
    return removed


def _sha256sums(root: Path):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append("%s  ./%s\n" % (_sha(path), path.relative_to(root)))
    (root / "SHA256SUMS").write_text("".join(lines))


def upload(folder: Path, repo: str, wave: str, private: bool, token=None) -> dict:
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
    api.create_repo(repo_id=repo, repo_type="dataset", private=private,
                    exist_ok=True)
    result = api.upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(folder),
        path_in_repo=wave, commit_message="Add %s results" % wave)
    return {"repo": repo, "path_in_repo": wave,
            "url": getattr(result, "commit_url", None), "status": "uploaded"}


def local_bundle(folder: Path, wave: str, gpu: str, dest_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d", time.gmtime())
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in (gpu or "gpu"))
    dest = Path(dest_dir) / ("vch_%s_%s_%s.tar.gz" % (wave, slug, stamp))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as archive:
        archive.add(folder, arcname=wave)
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--gpu", default="unknown-gpu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo", default=os.environ.get("HF_RESULT_REPO",
                                                        "khazic/visual-context-horizon-results"))
    parser.add_argument("--private", default=os.environ.get("HF_REPO_PRIVATE", "0"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--bundle-dir", default=None)
    args = parser.parse_args(argv)
    folder = Path(args.out_dir)
    manifest = stage(Path(args.run_dir), args.wave, folder)
    report = {"folder": str(folder), "manifest": manifest, "status": "staged"}
    if not args.no_upload:
        try:
            report.update(upload(folder, args.repo, args.wave,
                                 private=str(args.private).lower() in ("1", "true", "yes")))
        except Exception as error:
            print("HF upload failed: %s" % error, file=sys.stderr)
            report["upload_error"] = str(error)[:500]
    if report.get("status") != "uploaded":
        bundle = local_bundle(folder, args.wave, args.gpu,
                              args.bundle_dir or Path.cwd())
        report.update({"status": "local_bundle", "bundle": str(bundle)})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
