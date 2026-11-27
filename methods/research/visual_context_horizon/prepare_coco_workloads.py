"""Prepare seeded, image-disjoint VQAv2 and COCO long-form samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zlib
import urllib.request
from zipfile import ZipFile

from PIL import Image

from .prepare_workloads import SHORT_INSTRUCTION
from .run_visual_sweep import CONFIG


def from_zip(path):
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"Expected a single official JSON: {path}")
        return json.loads(archive.read(names[0]))


def check_image(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError("COCO source did not return JPEG bytes")
        image.verify()
    return hashlib.sha256(data).hexdigest()


def download_vqa_image(sample, image_dir, base):
    image_id, question = sample
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    path = image_dir / filename
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"{base}/{filename}", timeout=45) as response:
                    data = response.read()
                check_image(data)
                temp = path.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(path)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    sha = check_image(path.read_bytes())
    return question, filename, sha


def fetch_coco(entry, directory, base):
    dest = directory / Path(entry.filename).name
    for attempt in range(3):
        try:
            if dest.exists():
                data = dest.read_bytes()
            else:
                with urllib.request.urlopen(f"{base}/{dest.name}", timeout=45) as response:
                    data = response.read()
            if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
                raise ValueError(f"Individual COCO JPEG differs from release ZIP: {dest.name}")
            check_image(data)
            if not dest.exists():
                temp = dest.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(dest)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--coco-val2017", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=CONFIG["full_per_task"])
    parser.add_argument("--coco-image-base", help="Use official individual JPEGs with validation against ZIP metadata")
    parser.add_argument("--vqa-image-base", default="http://images.cocodataset.org/val2014")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Expected a positive count")
    vqa = args.output / "general_vqa"
    image_dir = vqa / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    by_image = {}
    for row in from_zip(args.questions)["questions"]:
        previous = by_image.get(row["image_id"])
        if previous is None or row["question_id"] < previous["question_id"]:
            by_image[row["image_id"]] = row
    chosen = random.Random(CONFIG["sample_seed"]).sample(sorted(by_image), args.count)
    qids = {by_image[key]["question_id"] for key in chosen}
    gold = {row["question_id"]: row for row in from_zip(args.annotations)["annotations"] if row["question_id"] in qids}
    if set(gold) != qids:
        raise ValueError("VQAv2 question/annotation join incomplete")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda sample: download_vqa_image(sample, image_dir, args.vqa_image_base),
                             ((key, by_image[key]) for key in chosen)))
    prompts, targets = [], []
    for row, name, sha in rows:
        qid = row["question_id"]
        annotation = gold[qid]
        if row["image_id"] != annotation["image_id"]:
            raise ValueError("VQAv2 image ID disagrees across original sources")
        sid = f"vqa-{qid}"
        question = row["question"]
        prompts.append(dict(id=sid, images=[name], conversations=[
            dict(**{"from": "human"}, value=question + SHORT_INSTRUCTION)]))
        targets.append(dict(id=sid, image=name, image_sha256=sha,
                            question_sha256=hashlib.sha256(json.dumps(question, ensure_ascii=False).encode()).hexdigest(),
                            answers=[item["answer"] for item in annotation["answers"]]))
    for suffix, values in [("prompts", prompts), ("targets", targets)]:
        path = vqa / f"vqav2_v2_{suffix}.jsonl"
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
        temp.replace(path)
    image_dir = args.output / "long_form" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    used = set(chosen)
    with ZipFile(args.coco_val2017) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith("val2017/") and Path(name).suffix == ".jpg"
                       and int(Path(name).stem) not in used)
        if len(names) < args.count:
            raise ValueError("COCO validation archive is incomplete")
        selected = [archive.getinfo(name) for name in random.Random(CONFIG["sample_seed"]).sample(names, args.count)]
        if args.coco_image_base:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda entry: fetch_coco(entry, image_dir, args.coco_image_base), selected))
        else:
            for entry in selected:
                data = archive.read(entry)
                check_image(data)
                (image_dir / Path(entry.filename).name).write_bytes(data)
    print(f"Verified {len(rows)} VQAv2 images and {args.count} COCO validation images")


if __name__ == "__main__":
    main()
