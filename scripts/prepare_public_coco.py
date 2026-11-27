#!/usr/bin/env python3
"""Materialize the canonical VQAv2 and COCO long-form samples from official JPEGs.

The sample identity comes from workloads/selected-500.json; every downloaded
image is checked against its recorded SHA256. No COCO/VQA archive is needed.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHORT_INSTRUCTION = "\nAnswer the question using a single word or phrase."
BASES = {
    "general_vqa": "http://images.cocodataset.org/val2014",
    "long_form": "http://images.cocodataset.org/val2017",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest, expected, retries=5):
    dest = Path(dest)
    if dest.is_file() and sha256(dest) == expected:
        return "cached"
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            start = part.stat().st_size if part.is_file() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
            try:
                response = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as error:
                if error.code == 416 and start:
                    part.unlink(missing_ok=True)
                    continue
                raise
            with response, part.open("ab" if start else "wb") as stream:
                while block := response.read(1 << 20):
                    stream.write(block)
            if sha256(part) != expected:
                if attempt + 1 == retries:
                    raise ValueError(f"downloaded bytes differ from the recorded SHA256: {dest.name}")
                part.unlink(missing_ok=True)
                continue
            part.replace(dest)
            return "downloaded"
        except (OSError, ValueError) as error:
            if attempt + 1 == retries:
                raise
            print(f"retry {attempt + 1} {dest.name}: {error}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text())["tasks"]
    jobs = []
    for task in ("general_vqa", "long_form"):
        target_dir = args.out / task / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = args.cache / task
        cache_dir.mkdir(parents=True, exist_ok=True)
        for row in selected[task]:
            name = Path(row["image_relative"]).name
            dest = cache_dir / name
            jobs.append((task, name, row["image_sha256"], dest, target_dir / name))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, f"{BASES[task]}/{name}", dest, expected)
                   for task, name, expected, dest, _ in jobs]
        for future in futures:
            future.result()

    for task, name, expected, dest, final in jobs:
        got = sha256(dest)
        if got != expected:
            raise SystemExit(f"{task} image hash mismatch after download: {dest}")
        if not final.is_file():
            final.write_bytes(dest.read_bytes())

    prompts, targets = [], []
    for row in selected["general_vqa"]:
        sid = row["id"].split(":", 1)[1]
        name = Path(row["image_relative"]).name
        question = row["question"]
        if not question.endswith(SHORT_INSTRUCTION):
            raise SystemExit(f"question does not carry the frozen short instruction: {sid}")
        bare = question[: -len(SHORT_INSTRUCTION)]
        prompts.append({"id": sid, "images": [name],
                        "conversations": [{"from": "human", "value": question}]})
        targets.append({
            "id": sid,
            "image": name,
            "image_sha256": row["image_sha256"],
            "question_sha256": hashlib.sha256(
                json.dumps(bare, ensure_ascii=False).encode()).hexdigest(),
            "answers": row["answers"],
        })

    vqa = args.out / "general_vqa"
    (vqa / "vqav2_v2_prompts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts))
    (vqa / "vqav2_v2_targets.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in targets))
    print(f"materialized {len(jobs)} images; wrote general_vqa prompts/targets")


if __name__ == "__main__":
    main()
