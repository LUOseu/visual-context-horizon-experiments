"""Request construction, faithful timing rows, acceptance counters and profiling."""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

from . import config as cfg
from . import workloads

METRIC_NAMES = {
    "vllm:spec_decode_num_drafts": "draft_rounds",
    "vllm:spec_decode_num_draft_tokens": "proposed_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted_draft_tokens",
}


def image_target_pixels(config: dict, cell: dict) -> int:
    frames = max(1, int(cell["frames"]))
    return workloads.pixel_bounds(config, int(cell["budget"]), frames)


def _load_rgb(path: str):
    from PIL import Image
    image = Image.open(path)
    image.load()
    return image.convert("RGB")


def prepare_image(path: str, pixels: int, masked: bool):
    """Downsample to the budget; never upscale; optional deterministic mask."""
    from PIL import Image
    image = _load_rgb(path)
    width, height = image.size
    current = width * height
    if current > pixels:
        scale = math.sqrt(float(pixels) / current)
        size = (max(32, int(width * scale)), max(32, int(height * scale)))
        if size != (width, height):
            image = image.resize(size, Image.LANCZOS)
    if masked:
        # Same visual-token footprint, no visual content.  This is an input
        # mask, not an attention-kernel sparsity claim.
        image = Image.new("RGB", image.size, (127, 127, 127))
    return image


def build_request(sample: dict, cell: dict, config: dict, processor,
                  masked: bool = False):
    images = []
    content = []
    paths = sample.get("images") or []
    frames = int(cell["frames"]) if cell["track"] == "multi_frame" else len(paths)
    frames = min(max(1, frames), len(paths)) if paths else 0
    expected = sample.get("image_sha256") or []
    if paths:
        for index, path in enumerate(paths[:frames]):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            digest = cfg.sha256(path)
            if index < len(expected) and expected[index] != digest:
                raise ValueError("physical input image changed: %s" % path)
            content.append({"type": "image"})
            images.append(prepare_image(path, image_target_pixels(config, cell), masked))
    text = sample["question"]
    if cell["track"] == "text_control":
        target = int(sample.get("text_target_tokens") or cell["budget"])
        text, prompt = fit_text_prompt(processor, text, target, config)
        return prompt, images, text, target
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **config.get("chat_template_kwargs", {"enable_thinking": False}))
    return prompt, images, text, None


def fit_text_prompt(processor, text: str, target: int, config: dict):
    """Trim a long text body so the final templated prompt is within a few tokens."""
    tokenizer = processor.tokenizer
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("empty text control prompt")
    budget = max(1, int(target) - 32)
    body_ids = ids[:budget]
    prompt = None
    for _ in range(10):
        body = tokenizer.decode(body_ids, skip_special_tokens=True)
        messages = [{"role": "user", "content": [{"type": "text", "text": body}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **config.get("chat_template_kwargs", {"enable_thinking": False}))
        length = len(tokenizer(prompt)["input_ids"])
        delta = length - int(target)
        if delta == 0:
            break
        if delta > 0:
            body_ids = body_ids[:max(1, len(body_ids) - delta)]
        else:
            extra = ids[len(body_ids):len(body_ids) - delta]
            if not extra:
                break
            body_ids = body_ids + extra
    return tokenizer.decode(body_ids, skip_special_tokens=True), prompt


def sampling_params(cell: dict, config: dict, SamplingParams):
    engine = config["engine"]
    target = config["targets"][cell["target"]]
    temperature = float(cell["temperatures"][0 if cell["arm"] == "ar" else 1])
    kwargs = {
        "temperature": temperature,
        "max_tokens": int(cell["max_new_tokens"]),
        "seed": int(engine["seed"]),
        "stop_token_ids": list(target["stop_token_ids"]),
        "ignore_eos": False,
    }
    if temperature > 0:
        kwargs.update(top_p=1.0, top_k=-1)
    if cell["arm"] == "ar" and temperature == 0:
        kwargs["logprobs"] = int(engine["topk_logprobs"])
    if cell.get("prompt_logprobs"):
        # Visual-KL cells only need the next-token distribution after the
        # prompt; one decode step keeps the pass cheap.
        kwargs["max_tokens"] = 1
        kwargs["prompt_logprobs"] = int(engine["topk_logprobs"])
    return SamplingParams(**kwargs)


def _metric_value(metric):
    value = getattr(metric, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_metrics(llm) -> dict:
    values = {name: 0.0 for name in METRIC_NAMES.values()}
    found = {name: False for name in METRIC_NAMES.values()}
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None
    for metric in metrics:
        name = getattr(metric, "name", None)
        key = METRIC_NAMES.get(name)
        if not key:
            continue
        value = _metric_value(metric)
        if value is None:
            continue
        values[key] += value
        found[key] = True
    if not all(found.values()):
        return None
    return values


def metric_delta(before, after) -> dict:
    if before is None or after is None:
        return {"status": "unavailable"}
    values = {key: after[key] - before[key] for key in before}
    if min(values.values()) < 0 or values["accepted_draft_tokens"] > values["proposed_draft_tokens"]:
        return {"status": "unavailable"}
    rounds = values["draft_rounds"]
    proposed = values["proposed_draft_tokens"]
    accepted = values["accepted_draft_tokens"]
    return dict(
        status="complete", **values,
        accepted_per_round=(accepted / rounds) if rounds else None,
        accepted_draft_fraction=(accepted / proposed) if proposed else None,
        definition="accepted draft tokens only; bonus token excluded",
    )


def entropy_from_logprobs(logprobs) -> tuple:
    if not logprobs:
        return None, None
    entropies, top1 = [], []
    for step in logprobs:
        if not step:
            continue
        values = list(step.values()) if isinstance(step, dict) else [step]
        probs = []
        for value in values:
            logprob = getattr(value, "logprob", None)
            if logprob is None and isinstance(value, (list, tuple)):
                logprob = value[1] if len(value) > 1 else None
            if logprob is None:
                continue
            probs.append(math.exp(float(logprob)))
        total = sum(probs)
        if total <= 0:
            continue
        probs = [value / total for value in probs]
        entropies.append(-sum(p * math.log(p) for p in probs if p > 0))
        top1.append(max(probs))
    if not entropies:
        return None, None
    return sum(entropies) / len(entropies), sum(top1) / len(top1)


def _ttft_ms(output) -> float:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    first = getattr(metrics, "first_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    if first is None or arrival is None:
        return None
    value = (first - arrival) * 1000.0
    return value if value >= 0 else None


def _topk_entry(entry, topk: int):
    """Normalize one vLLM prompt-logprob position to {token_id: logprob}."""
    if not entry:
        return None
    out = {}
    for token, value in entry.items():
        logprob = getattr(value, "logprob", None)
        if logprob is None:
            logprob = value if isinstance(value, (int, float)) else None
        if logprob is None:
            continue
        try:
            out[str(int(token))] = float(logprob)
        except (TypeError, ValueError):
            continue
    if not out:
        return None
    ranked = sorted(out.items(), key=lambda kv: -kv[1])[:max(1, int(topk))]
    return dict(ranked)


def prompt_topk(output, topk: int) -> tuple:
    """Last prompt position plus a short tail, for offline visual-KL."""
    positions = getattr(output, "prompt_logprobs", None)
    if not positions:
        return None, None
    last = _topk_entry(positions[-1], topk)
    tail = [_topk_entry(entry, topk) for entry in positions[-8:] if entry]
    tail = [entry for entry in tail if entry]
    return last, (tail or None)


def _image_token_count(prompt_ids, target: dict) -> int:
    token = target.get("image_token_id")
    if token is None:
        return 0
    return sum(1 for value in prompt_ids if int(value) == int(token))


def request_rows(llm, processor, SamplingParams, config: dict, cell: dict,
                 samples: list, repeat: int, gpu_label: str, profiler=None) -> list:
    """Run one timed batch (or one sample when batch_size == 1)."""
    import torch
    params = sampling_params(cell, config, SamplingParams)
    target = config["targets"][cell["target"]]
    requests, prompts, images = [], [], []
    for sample in samples:
        prompt, pictures, text, _target = build_request(
            sample, cell, config, processor,
            masked=cell["visual_kv"] == "masked")
        entry = {"prompt": prompt}
        if pictures:
            entry["multi_modal_data"] = {"image": pictures}
        requests.append(entry)
        prompts.append((prompt, text))
        images.extend(pictures)
    before = read_metrics(llm) if cell["arm"] == "sd" else None
    started = time.perf_counter_ns()
    try:
        if profiler is not None:
            outputs = profiler.generate(llm, requests, params, cell, samples)
        else:
            outputs = llm.generate(requests, params, use_tqdm=False)
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass
    ended = time.perf_counter_ns()
    after = read_metrics(llm) if cell["arm"] == "sd" else None
    delta = metric_delta(before, after)
    batch_ms = (ended - started) / 1e6
    rows = []
    if len(outputs) != len(samples):
        raise ValueError("batch1 contract changed: %d outputs for %d inputs"
                         % (len(outputs), len(samples)))
    for index, (sample, output) in enumerate(zip(samples, outputs)):
        if len(output.outputs) != 1:
            raise ValueError("single-output contract changed")
        generated = output.outputs[0]
        prompt_ids = list(output.prompt_token_ids)
        token_ids = list(generated.token_ids)
        entropy = top1 = None
        if cell["arm"] == "ar" and float(cell["temperatures"][0]) == 0:
            entropy, top1 = entropy_from_logprobs(getattr(generated, "logprobs", None))
        visual_tokens = _image_token_count(prompt_ids, target)
        row = {
            "status": "complete",
            "cell_id": cell["id"], "pair_key": cell["pair_key"],
            "experiment": cell["experiment"], "target": cell["target"],
            "method": cell["method"], "arm": cell["arm"],
            "task": cell["task"], "track": cell["track"],
            "dataset": cell["dataset"],
            "budget": cell["budget"], "frames": cell["frames"],
            "template": cell["template"], "block_width": cell["block_width"],
            "batch_size": cell["batch_size"], "apc": cell["apc"],
            "visual_kv": cell["visual_kv"],
            "drafter_visual_kv_mode": (cell["visual_kv"]
                                       if cell["arm"] == "sd" else "full"),
            "grouping": cell["grouping"],
            "tp": cell["tp"], "repeat": repeat, "phase": cell["phase"],
            "sample_id": sample["id"], "batch_index": index,
            "prompt_tokens": len(prompt_ids),
            "visual_tokens": visual_tokens,
            "text_prompt_tokens": len(prompt_ids) - visual_tokens,
            "output_tokens": len(token_ids),
            "ttft_ms": _ttft_ms(output),
            "e2e_ms": batch_ms,
            "per_request_ms": batch_ms / max(1, len(samples)),
            "batch_e2e_ms": batch_ms,
            "accepted_draft_tokens": delta.get("accepted_draft_tokens"),
            "draft_rounds": delta.get("draft_rounds"),
            "proposed_draft_tokens": delta.get("proposed_draft_tokens"),
            "accepted_per_round": delta.get("accepted_per_round"),
            "accepted_draft_fraction": delta.get("accepted_draft_fraction"),
            "metric_status": delta.get("status"),
            "token_entropy": entropy,
            "top1_prob": top1,
            "logprob_topk": config["engine"]["topk_logprobs"] if entropy is not None else None,
            "token_ids": token_ids,
            "token_ids_sha256": hashlib.sha256(
                (",".join(str(v) for v in token_ids)).encode()).hexdigest(),
            "text_sha256": hashlib.sha256(generated.text.encode()).hexdigest(),
            "finish_reason": getattr(generated, "finish_reason", None),
            "stop_reason": getattr(generated, "stop_reason", None),
            "seed": config["engine"]["seed"],
            "temperature": float(cell["temperatures"][0 if cell["arm"] == "ar" else 1]),
            "gpu": gpu_label,
            "oom_tp1": False,
            "sync_used": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if cell.get("prompt_logprobs"):
            last, tail = prompt_topk(output, config["engine"]["topk_logprobs"])
            row["next_token_topk"] = last
            row["prompt_topk_tail"] = tail
            row["topk_definition"] = "renormalized over stored top-k prompt logprobs"
        if cell["profile"] and profiler is not None:
            row["profile"] = profiler.last
        rows.append(row)
    return rows


def warmup(llm, processor, SamplingParams, config, cell, samples, masked=False):
    if not samples:
        return
    import torch
    params = sampling_params(cell, config, SamplingParams)
    for index in range(int(config["engine"]["warmup_requests_per_cell"])):
        sample = samples[index % len(samples)]
        prompt, pictures, _text, _target = build_request(
            sample, cell, config, processor, masked=masked)
        entry = {"prompt": prompt}
        if pictures:
            entry["multi_modal_data"] = {"image": pictures}
        try:
            llm.generate([entry], params, use_tqdm=False)
        finally:
            for image in pictures:
                try:
                    image.close()
                except Exception:
                    pass


def token_match(ar_rows: list, sd_rows: list) -> dict:
    """Exact-match comparison for greedy 0|0 pairs, per sample and repeat."""
    index = {(row["sample_id"], row["repeat"]): row for row in ar_rows}
    matched = compared = 0
    first_diff = None
    for row in sd_rows:
        key = (row["sample_id"], row["repeat"])
        if key not in index:
            continue
        other = index[key]
        compared += 1
        if other.get("token_ids") == row.get("token_ids"):
            matched += 1
        elif first_diff is None:
            first_diff = {"sample_id": row["sample_id"], "repeat": row["repeat"]}
    return {"compared": compared, "matched": matched,
            "exact_match": (matched / compared) if compared else None,
            "first_difference": first_diff}
