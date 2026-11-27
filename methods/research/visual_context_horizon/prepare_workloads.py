"""Build a byte-verified and deterministically ordered workload manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from vlm_dflash.eval.native import digest, save

from .run_visual_sweep import CONFIG

TASKS = {
    "general_vqa": ("HuggingFaceM4/VQAv2", "validation"),
    "textvqa": ("lmms-lab/textvqa", "validation"),
    "docvqa": ("lmms-lab/DocVQA", "DocVQA/validation"),
    "chartqa": ("lmms-lab/ChartQA", "test"),
    "long_form": ("COCO 2017", "val2017"),
}
SHORT_INSTRUCTION = "\nAnswer the question using a single word or phrase."


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_task(task, root, *, allow_missing_images=False):
    dataset, split = TASKS[task]
    root = root.resolve()
    images = root / "images"
    if task == "long_form":
        files = sorted(path for path in images.glob("*.jpg") if len(path.stem) == 12 and path.stem.isdecimal())
        prompts = [{
            "id": f"coco-{path.stem}", "images": [path.name],
            "conversations": [{"from": "human", "value": CONFIG["long_form_prompt"]}],
        } for path in files]
        targets = None
    else:
        name = "vqav2" if task == "general_vqa" else task
        prompts = jsonl(root / f"{name}_v2_prompts.jsonl")
        values = jsonl(root / f"{name}_v2_targets.jsonl")
        targets = {str(row["id"]): row for row in values}
        if len(targets) != len(values) or set(targets) != {str(row["id"]) for row in prompts}:
            raise ValueError(f"{task}: prompt/gold IDs disagree")
    seen = set()
    rows = []
    for prompt in prompts:
        sid = str(prompt["id"])
        if sid in seen or len(prompt["images"]) != 1:
            raise ValueError(f"{task}: duplicate ID or invalid image scope: {sid}")
        seen.add(sid)
        turns = prompt["conversations"]
        if not 1 <= len(turns) <= 2 or turns[0]["from"] != "human" or (
            len(turns) == 2 and (turns[1]["from"] != "gpt" or turns[1]["value"])
        ):
            raise ValueError(f"{task}: expected a single question without an answer: {sid}")
        question = turns[0]["value"]
        path = (images / prompt["images"][0]).resolve()
        if images.resolve() not in path.parents or not question.strip():
            raise ValueError(f"{task}: invalid question or image scope: {sid}")
        if not path.is_file() and allow_missing_images:
            continue
        sha = digest(path)
        answers = None
        if targets is not None:
            gold = targets[sid]
            encoded = json.dumps(question.removesuffix(SHORT_INSTRUCTION), ensure_ascii=False, sort_keys=True)
            if (gold["image"] != prompt["images"][0] or gold["image_sha256"] != sha
                    or gold["question_sha256"] != hashlib.sha256(encoded.encode()).hexdigest()):
                raise ValueError(f"{task}: gold, question, or image mismatch: {sid}")
            answers = gold["answers"]
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
                raise ValueError(f"{task}: invalid answers: {sid}")
            if task in {"general_vqa", "textvqa"} and len(answers) != 10:
                raise ValueError(f"{task}: expected ten gold answers: {sid}")
        rows.append(dict(id=f"{task}:{sid}", dataset_id=dataset, dataset_split=split, task=task,
                         image=str(path), image_sha256=sha, question=question, answers=answers))
    rows.sort(key=lambda row: row["id"])
    random.Random(CONFIG["sample_seed"]).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true", help="Permit explicitly incomplete image caches")
    parser.add_argument("--count", type=int, help="Keep only N verified samples per task for smoke")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists")
    if args.count is not None and (not args.smoke_only or args.count < 1):
        parser.error("--count requires --smoke-only and a positive count")
    grouped = {}
    for spec in args.task:
        task, path = spec.split("=", 1)
        if task in grouped or task not in TASKS:
            parser.error(f"Unexpected or duplicate task {task}")
        grouped[task] = read_task(task, Path(path), allow_missing_images=args.smoke_only)[:args.count]
        if not grouped[task]:
            parser.error(f"No usable samples for {task}")
        print(f"{task}: {len(grouped[task])} verified")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, {"seed": CONFIG["sample_seed"], "tasks": grouped, "smoke_only": args.smoke_only})


if __name__ == "__main__":
    main()
