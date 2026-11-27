"""Deterministic, task-aware workloads for the vision-overhead suite.

Five tasks come from the frozen ``workloads/selected-500.json`` pool
(textvqa, docvqa, chartqa, vqav2, coco_long_form).  HR-Bench 4K/8K,
MP-DocVQA and Video-MME are materialised from their pinned parquet/video
caches.  MathVista, OCRBench and InfoVQA are optional: they are ingested from
``$DATA_ROOT/<task>/raw_manifest.json`` when that file is present.

No image is ever enlarged and no pixel is synthesised.  A sample is eligible
for a (budget, frames) pair only when its native pixels can supply that many
visual tokens after downsampling to the budget.  Actual token counts are
measured again at request time from the tokenizer output.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

from . import config as cfg

LEGACY_TASKS = {
    "textvqa": "textvqa",
    "docvqa": "docvqa",
    "chartqa": "chartqa",
    "general_vqa": "vqav2",
    "long_form": "coco_long_form",
}
OPTIONAL_TASKS = ("mathvista", "ocrbench", "infovqa")
HORIZON_TEMPLATES = {
    "direct": ("Answer with the final answer only. Do not explain.", 16),
    "brief": ("Look at the image carefully. Answer the question in one or two concise "
              "sentences, then state the final answer on its own line. "
              "Do not add extra detail.", 96),
    "cot": ("Look at the image carefully and reason step by step before answering.\n"
            "Use this structure:\n"
            "1. Identify every object, region, label and number that the question "
            "could refer to.\n"
            "2. Describe their spatial relationships and any text they contain.\n"
            "3. Check each option or candidate answer against what is visible.\n"
            "4. Resolve contradictions by returning to the image rather than guessing.\n"
            "5. State the final answer on its own line, prefixed with 'Final answer:'.\n"
            "Keep the reasoning concrete and grounded in visible evidence.", 384),
}


def visual_tokens_for_pixels(pixels: int, config: dict) -> int:
    engine = config["engine"]
    unit = engine["patch_size"] * engine["spatial_merge_size"]
    return int(pixels) // (unit * unit)


def pixel_bounds(config: dict, budget: int, frames: int = 1) -> int:
    """Pixels per image so that ``frames`` images supply ``budget`` visual tokens."""
    engine = config["engine"]
    unit = engine["patch_size"] * engine["spatial_merge_size"]
    per_frame = max(1, int(math.ceil(float(budget) / max(1, frames))))
    pixels = int(round(per_frame * engine["pixel_floor_ratio"])) * (unit * unit)
    return max(engine["min_pixels_floor"], min(engine["max_pixels_cap"], pixels))


def _open_image(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data))


def _image_bytes(value):
    if isinstance(value, dict):
        if value.get("bytes"):
            return bytes(value["bytes"])
        value = value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        if value.startswith("/") or value.startswith("file:"):
            return Path(value.replace("file:", "", 1)).read_bytes()
        return base64.b64decode(value)
    raise TypeError("unsupported image value %r" % type(value))


def _image_meta(path: Path) -> dict:
    with _open_image(path.read_bytes()) as image:
        width, height = image.size
    return {"path": str(path.resolve()), "sha256": cfg.sha256(path),
            "width": width, "height": height, "pixels": width * height}


def _write_jpeg(data: bytes, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data[:2] == b"\xff\xd8":
        path.write_bytes(data)
    else:
        with _open_image(data) as image:
            image.convert("RGB").save(path, format="JPEG", quality=95)
    return _image_meta(path)


def _sample(task: str, dataset: str, track: str, sample_id: str, images: list,
            question: str, answers, source: dict, config: dict,
            frames: int = None, template: str = None) -> dict:
    metas = [_image_meta(Path(path)) for path in images]
    return {
        "id": sample_id, "task": task, "dataset": dataset, "track": track,
        "images": [meta["path"] for meta in metas],
        "image_sha256": [meta["sha256"] for meta in metas],
        "image_sizes": [[meta["width"], meta["height"]] for meta in metas],
        "native_visual_tokens": [
            visual_tokens_for_pixels(meta["pixels"], config) for meta in metas],
        "pixels": [meta["pixels"] for meta in metas],
        "question": question, "answers": answers, "source": source,
        "frames": int(frames if frames is not None else len(metas)),
        "template": template,
    }


def _legacy_rows(config: dict) -> dict:
    path = cfg.repo_root() / "workloads" / "selected-500.json"
    payload = cfg.read_json(path)
    if not payload:
        raise FileNotFoundError("%s is missing; run scripts/prepare_8gpu.sh" % path)
    return payload


def materialize_selected(config: dict, per_task: int) -> dict:
    """textvqa/docvqa/chartqa/vqav2/coco_long_form from the frozen pool."""
    payload = _legacy_rows(config)
    data_root = cfg.paths(config)["data"]
    rng = random.Random(int(config["workloads"]["seed"]))
    out = {}
    for legacy, task in LEGACY_TASKS.items():
        rows = list(payload["tasks"].get(legacy) or [])
        rows.sort(key=lambda row: row["id"])
        if len(rows) < per_task:
            raise ValueError("task %s has %d frozen samples, need %d"
                             % (task, len(rows), per_task))
        picked = rng.sample(rows, per_task)
        picked.sort(key=lambda row: row["id"])
        samples = []
        for row in picked:
            relative = row.get("image_relative")
            if not relative:
                raise ValueError("sample %s has no image_relative" % row["id"])
            path = data_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = cfg.sha256(path)
            if row.get("image_sha256") and row["image_sha256"] != digest:
                raise ValueError("image changed: %s" % path)
            samples.append(_sample(
                task, task, "single_img", "%s:%s" % (task, row["id"]),
                [str(path)], row["question"], row.get("answers"),
                {"dataset": row.get("dataset_id"),
                 "split": row.get("dataset_split")}, config))
        out[task] = samples
    return out


def _hrbench_pool(config: dict):
    import pyarrow.parquet as pq
    paths = cfg.paths(config)
    pins = cfg.data_pins()
    out_dir = paths["artifacts"] / "workloads" / "images" / "hrbench"
    counts = config["workloads"]["high_res"]
    rows = []
    for split, count in (("4k", counts["4k"]), ("8k", counts["8k"])):
        parquet = paths["cache"] / "hrbench" / ("hr_bench_%s.parquet" % split)
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        table = sorted(pq.read_table(parquet).to_pylist(),
                       key=lambda row: int(row["index"]))
        picked = random.Random(int(config["workloads"]["seed"])).sample(
            table, min(int(count), len(table)))
        picked.sort(key=lambda row: int(row["index"]))
        for row in picked:
            option_text = "\n".join("%s. %s" % (letter, row[letter]) for letter in "ABCD")
            question = ("%s\n%s\nAnswer with the option letter only."
                        % (row["question"], option_text))
            path = out_dir / split / ("hrbench-%s-%05d.jpg" % (split, int(row["index"])))
            meta = _write_jpeg(_image_bytes(row["image"]), path)
            rows.append(_sample(
                "hrbench", "hrbench", "single_img",
                "hrbench:%s:%05d" % (split, int(row["index"])),
                [meta["path"]], question, [row["answer"]],
                {"dataset": "DreamMr/HR-Bench", "split": split,
                 "revision": pins["hrbench"]["revision"],
                 "index": int(row["index"])}, config))
    rows.sort(key=lambda row: row["id"])
    return rows


def _mp_docvqa_pool(config: dict):
    import pyarrow.parquet as pq
    paths = cfg.paths(config)
    pin = cfg.data_pins()["mp_docvqa"]
    questions = int(config["workloads"]["docs"]["questions"])
    files = sorted(paths["cache"] / "mp_docvqa" / Path(entry["path"]).name
                   for entry in pin["files"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("MP-DocVQA parquet missing: %s" % missing[0])
    eligible = []
    for path in files:
        for row in pq.read_table(path, columns=["questionId", "question", "answers",
                                                "page_ids"]).to_pylist():
            pages = [p for p in str(row["page_ids"]).split(",") if p]
            if len(pages) >= 16:
                eligible.append(dict(row, file=str(path)))
    eligible.sort(key=lambda row: row["questionId"])
    rng = random.Random(int(config["workloads"]["seed"]))
    picked = rng.sample(eligible, min(questions, len(eligible)))
    picked.sort(key=lambda row: row["questionId"])
    wanted = {row["questionId"] for row in picked}
    images = {}
    for path in files:
        names = [name for name in pq.read_schema(path).names
                 if name.startswith("image_")]
        table = pq.read_table(path, columns=["questionId"] + sorted(names)).to_pylist()
        for row in table:
            if row["questionId"] in wanted:
                images[row["questionId"]] = {
                    name: _image_bytes(row[name]) for name in sorted(names)
                    if row.get(name) is not None}
    out = []
    for row in picked:
        payloads = images.get(row["questionId"]) or {}
        if len(payloads) < 16:
            continue
        folder = paths["artifacts"] / "workloads" / "images" / "mp_docvqa" / row["questionId"]
        paths_out = []
        for index, name in enumerate(sorted(payloads)):
            paths_out.append(_write_jpeg(payloads[name],
                                         folder / ("page-%02d.jpg" % index))["path"])
        answers = ([row["answers"]] if isinstance(row["answers"], str)
                   else row["answers"])
        out.append(_sample(
            "mp_docvqa", "mp_docvqa", "multi_frame",
            "mp_docvqa:%s" % row["questionId"], paths_out,
            row["question"] + "\nAnswer with a single word or phrase.", answers,
            {"dataset": pin["repo"], "revision": pin["revision"],
             "question_id": row["questionId"]}, config))
    if len(out) < questions:
        raise ValueError("only %d MP-DocVQA samples survived extraction" % len(out))
    out.sort(key=lambda row: row["id"])
    return out


def extract_frames(video: Path, folder: Path, count: int) -> list:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required to extract Video-MME frames")
    duration = float(subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True).stdout.strip())
    if duration <= 0:
        raise ValueError("non-positive duration for %s" % video)
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(video),
         "-vf", "fps=%d/%f" % (count, duration), "-frames:v", str(count),
         "-q:v", "2", str(folder / "frame-%03d.jpg")],
        check=True)
    return [str(path) for path in sorted(folder.glob("frame-*.jpg"))]


def _video_mme_pool(config: dict):
    import pyarrow.parquet as pq
    paths = cfg.paths(config)
    pin = cfg.data_pins()["video_mme"]
    clips = int(config["workloads"]["video"]["clips"])
    frames_per_clip = int(config["workloads"]["video"]["frames"])
    annotation = paths["cache"] / "video_mme" / Path(pin["annotation_file"]).name
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    by_video = {}
    for row in pq.read_table(annotation).to_pylist():
        by_video.setdefault(row["videoID"], []).append(row)
    videos = {entry["video_id"]: entry for entry in pin["videos"]}
    available = sorted(set(videos) & set(by_video))
    rng = random.Random(int(config["workloads"]["seed"]))
    picked = sorted(rng.sample(available, min(clips, len(available))))
    out = []
    for video_id in picked:
        video = paths["cache"] / "video_mme" / Path(videos[video_id]["path"]).name
        if not video.is_file():
            raise FileNotFoundError(video)
        options = sorted(by_video[video_id], key=lambda row: row["question_id"])
        question = options[rng.randrange(len(options))]
        folder = paths["artifacts"] / "workloads" / "images" / "video_mme" / video_id
        paths_out = extract_frames(video, folder, frames_per_clip)
        if len(paths_out) < frames_per_clip:
            shutil.rmtree(folder, ignore_errors=True)
            continue
        option_text = "\n".join(question["options"])
        text = ("%s\n%s\nAnswer with the option letter only."
                % (question["question"], option_text))
        out.append(_sample(
            "video_mme", "video_mme", "multi_frame",
            "video_mme:%s" % video_id, paths_out, text, [question["answer"]],
            {"dataset": pin["repo"], "revision": pin["revision"],
             "video_id": video_id, "question_id": question["question_id"],
             "annotation_revision": pin["annotation_revision"]}, config))
    if len(out) < clips:
        raise ValueError("only %d Video-MME clips produced %d frames"
                         % (len(out), frames_per_clip))
    out.sort(key=lambda row: row["id"])
    return out


def _raw_pool(config: dict, task: str):
    """Optional task ingested from ``$DATA_ROOT/<task>/raw_manifest.json``.

    Each row: {"id", "image" or "images", "question", "answers"} with image
    paths relative to the task directory.  Absent sources yield no samples.
    """
    root = cfg.paths(config)["data"] / task
    manifest = root / "raw_manifest.json"
    if not manifest.is_file():
        return []
    rows = cfg.read_json(manifest) or []
    out = []
    for row in rows:
        paths = row.get("images") or [row.get("image")]
        paths = [root / path for path in paths if path]
        if not paths or any(not path.is_file() for path in paths):
            continue
        out.append(_sample(
            task, task, "multi_frame" if row.get("images") else "single_img",
            "%s:%s" % (task, row.get("id")), [str(path) for path in paths],
            row["question"], row.get("answers"),
            {"dataset": row.get("dataset", task)}, config))
    out.sort(key=lambda row: row["id"])
    return out


def _text_control(config: dict) -> list:
    corpus_path = cfg.repo_root() / config["workloads"]["text_corpus"]
    corpus = corpus_path.read_text(encoding="utf-8").strip()
    if not corpus:
        raise ValueError("empty text corpus")
    spec = config["workloads"]["text_control"]
    targets = [int(value) for value in spec["token_targets"]]
    prompts = int(spec["prompts"])
    repeats = int(math.ceil(max(targets) * 3.6 / len(corpus))) + 2
    body = "\n\n".join(corpus for _ in range(repeats))
    rows = []
    for index in range(prompts):
        for target in targets:
            rows.append({
                "id": "text_control:%03d:%d" % (index, target),
                "task": "text_control", "dataset": "text_control",
                "track": "text_control", "images": [], "image_sha256": [],
                "image_sizes": [], "native_visual_tokens": [],
                "question": "Prompt %03d\n\n%s" % (index, body),
                "answers": None, "source": {"corpus": str(corpus_path)},
                "frames": 0, "template": None,
                "text_target_tokens": int(target)})
    return rows


def materialize(config: dict) -> dict:
    """Materialise every task into one manifest (images live under artifacts)."""
    per_task = int(config["workloads"]["per_task"])
    samples = {}
    samples.update(materialize_selected(config, per_task))
    samples["hrbench"] = _hrbench_pool(config)
    samples["mp_docvqa"] = _mp_docvqa_pool(config)
    samples["video_mme"] = _video_mme_pool(config)
    notes = []
    for task in OPTIONAL_TASKS:
        rows = _raw_pool(config, task)
        if rows:
            samples[task] = rows
        elif cfg.task_spec(config, task).get("optional_data"):
            notes.append("%s: no raw_manifest.json under DATA_ROOT; task omitted" % task)
    samples["text_control"] = _text_control(config)
    pins = None
    try:
        pins = cfg.data_pins()
    except FileNotFoundError:
        notes.append("configs/matrix_data_files.json missing; data pins not recorded")
    recipe = {
        "suite": config["suite"], "seed": config["workloads"]["seed"],
        "workloads": config["workloads"],
        "drafters": {method: entry["sha256"] for method, entry in
                     config["targets"][cfg.VISION_TARGET]["drafters"].items()},
        "data_pins_sha256": cfg.sha256_json(pins) if pins else None,
    }
    return {
        "suite": config["suite"], "version": 1, "recipe": recipe,
        "recipe_sha256": cfg.sha256_json(recipe), "materialized": True,
        "counts": {name: len(rows) for name, rows in samples.items()},
        "samples": samples,
        "notes": notes + [
            "Images are byte-identical when the source is JPEG; otherwise re-encoded once.",
            "native_visual_tokens uses patch_size*merge_size (32 pixels per token).",
            "Eligibility never upscales: budgets run only where native pixels reach them.",
            "text_control is image-free and trimmed to the exact token count at request time.",
        ],
    }


def stub_manifest(config: dict) -> dict:
    """Shape-only manifest for ``--dry-run`` without downloaded data."""
    samples = {}
    for task in cfg.task_order(config, bonus=True):
        spec = cfg.task_spec(config, task)
        if spec.get("track") == "multi_frame":
            per_frame = int(config["workloads"].get("stub_frame_tokens", 1024))
            if task == "mp_docvqa":
                frames, count = 16, 60
            else:
                frames, count = 64, 50
            samples[task] = [{
                "id": "stub-%s-%03d" % (task, index), "task": task,
                "dataset": spec.get("dataset", task), "track": "multi_frame",
                "images": ["stub/%s/%03d-%03d.jpg" % (task, index, frame)
                           for frame in range(frames)],
                "question": "stub", "frames": frames,
                "native_visual_tokens": [per_frame] * frames}
                for index in range(count)]
        else:
            samples[task] = [{
                "id": "stub-%s-%03d" % (task, index), "task": task,
                "dataset": spec.get("dataset", task), "track": "single_img",
                "images": [], "question": "stub", "frames": 1,
                "native_visual_tokens": [16384]}
                for index in range(200)]
    samples["text_control"] = [{
        "id": "stub-text-%d-%02d" % (int(budget), index), "task": "text_control",
        "dataset": "text_control", "track": "text_control", "images": [],
        "question": "stub", "frames": 0, "native_visual_tokens": [],
        "text_target_tokens": int(budget)}
        for budget in config["workloads"]["text_control"]["token_targets"]
        for index in range(4)]
    return {
        "suite": config["suite"], "version": 1, "recipe": {"stub": True},
        "recipe_sha256": cfg.sha256_json({"stub": True}), "materialized": True,
        "stub": True, "counts": {name: len(rows) for name, rows in samples.items()},
        "samples": samples,
        "notes": ["synthetic dry-run manifest; no real images are referenced"],
    }


def load_manifest(config: dict, path=None) -> dict:
    if path is None:
        name = config["workloads"].get("manifest_name", "workloads_manifest.json")
        path = cfg.paths(config)["manifests"] / name
    path = Path(path)
    manifest = cfg.read_json(path)
    if not manifest:
        raise FileNotFoundError("%s is missing; run scripts/prepare_8gpu.sh" % path)
    if manifest.get("recipe_sha256") != cfg.sha256_json(manifest.get("recipe")):
        raise ValueError("workload manifest recipe hash is corrupt: %s" % path)
    if not manifest.get("materialized"):
        raise ValueError("workload manifest is a recipe only; run scripts/prepare_8gpu.sh")
    for sample in manifest["samples"].get("text_control") or []:
        if sample["images"]:
            raise ValueError("text_control sample carries an image")
    return manifest


def select_samples(manifest: dict, config: dict, task: str, budget: int,
                   frames: int = None, limit: int = None) -> list:
    """Return eligible samples for one cell, deterministically ordered."""
    budget = int(budget)
    if task == "text_control":
        pool = [row for row in manifest["samples"].get("text_control") or []
                if int(row.get("text_target_tokens") or 0) == budget]
        pool.sort(key=lambda row: row["id"])
        return pool[:limit] if limit else pool
    pool = list(manifest["samples"].get(task) or [])
    if frames:
        pool = [row for row in pool if len(row.get("images") or []) >= int(frames)]
    pool = [row for row in pool if eligible(row, budget, frames)]
    pool.sort(key=lambda row: row["id"])
    return pool[:limit] if limit else pool


def eligible(sample: dict, budget: int, frames: int = None) -> bool:
    tokens = sample.get("native_visual_tokens") or []
    if not tokens:
        return False
    used = tokens[:int(frames)] if frames else tokens
    return sum(int(value) for value in used) >= int(budget)


def missing_images(sample: dict) -> list:
    return [path for path in sample.get("images") or [] if not Path(path).is_file()]
