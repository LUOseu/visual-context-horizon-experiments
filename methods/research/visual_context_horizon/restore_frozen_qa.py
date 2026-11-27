"""Restore original image bytes for the frozen real-gold VelaDraft QA subsets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

from vlm_dflash.eval.native import digest

from .prepare_workloads import jsonl

SOURCES = {
    "textvqa": ("textvqa", "data", "9c0699cd19768ac5ab97568f6b3cbac4c0062884"),
    "docvqa": ("DocVQA", "DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e"),
    "chartqa": ("ChartQA", "data", "9e63b7df1592a1c2158e735cc1725454aef0d6d9"),
}


def restore(task, metadata, out, cache, endpoint):
    import pyarrow.parquet as parquet
    name, folder, rev = SOURCES[task]
    source = metadata / task
    destination = out / task
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)
    metadata_names = [f"{task}_v2_{suffix}.jsonl" for suffix in ("prompts", "targets")]
    targets = jsonl(source / metadata_names[1])
    pending = {}
    for row in targets:
        path = images / row["image"]
        if path.is_file():
            if digest(path) != row["image_sha256"]:
                raise ValueError(f"Image in existing cache changed: {path}")
        else:
            pending[row["source_index"]] = row
    offset = 0
    manifest = json.loads((source / "manifest.json").read_text())
    for basename, expected in sorted(manifest["shards"].items()):
        local = cache / task / basename if cache else None
        with tempfile.TemporaryDirectory(prefix=f"vch-{task}-") as temporary:
            if local is None or not local.is_file():
                local = Path(temporary) / basename
                url = f"{endpoint}/datasets/lmms-lab-encoder/{name}/resolve/{rev}/{folder}/{basename}"
                print(f"{task} {basename}: retrieving pinned bytes", flush=True)
                for attempt in range(3):
                    request = f"{url}?download=true&vch_attempt={time.time_ns()}"
                    try:
                        with urllib.request.urlopen(request, timeout=60) as response, local.open("wb") as file:
                            shutil.copyfileobj(response, file, length=2**20)
                        if digest(local) == expected:
                            break
                        raise ValueError(f"Downloaded shard hash mismatch: {basename}")
                    except (OSError, ValueError) as error:
                        if attempt == 2:
                            raise
                        print(f"{task} {basename}: retry after {error}", flush=True)
            elif digest(local) != expected:
                raise ValueError(f"Cached shard hash mismatch: {basename}")
            pf = parquet.ParquetFile(local)
            ceiling = offset + pf.metadata.num_rows
            wanted = {index: target for index, target in pending.items() if offset <= index < ceiling}
            if wanted:
                cursor = offset
                for batch in pf.iter_batches(columns=["image"], batch_size=16, use_threads=False):
                    found = [index for index in wanted if cursor <= index < cursor + batch.num_rows]
                    for index in found:
                        row = wanted.pop(index)
                        data = batch.column(0)[index - cursor].as_py()["bytes"]
                        if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
                            raise ValueError(f"Source image bytes differ: {row['id']}")
                        path = images / row["image"]
                        temporary_image = path.with_suffix(path.suffix + ".tmp")
                        temporary_image.write_bytes(data)
                        temporary_image.replace(path)
                        del pending[index]
                    if not wanted:
                        break
                    cursor += batch.num_rows
            print(f"{task} {basename}: verified, {len(pending)} images pending", flush=True)
            offset = ceiling
    if pending or offset != manifest["num_rows_in_split"]:
        raise ValueError(f"Unresolved frozen rows or unexpected total: {task}")
    for basename in metadata_names:
        shutil.copyfile(source / basename, destination / basename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--task", action="append", choices=SOURCES, required=True)
    args = parser.parse_args()
    for task in args.task:
        restore(task, args.metadata, args.out, args.cache, args.endpoint.rstrip("/"))


if __name__ == "__main__":
    main()
