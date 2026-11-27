"""Run the frozen VelaDraft native DFlash budget sweep."""
import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

from vlm_dflash.eval import native


CONFIG_PATH = Path(__file__).parent / "configs" / "dflash.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
TARGET_FILES = CONFIG_PATH.with_name("target_files.json")


def bounds(budget):
    unit = (CONFIG["target_patch_size"] * CONFIG["target_spatial_merge_size"]) ** 2
    return round(budget * CONFIG["pixel_floor_ratio"]) * unit, budget * unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    native.add_arguments(parser)
    parser.add_argument("--phase", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--count", type=int, help="Override sample count for smoke only")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--compile-cache", type=Path, help="Shared on-disk compile artifacts for fresh workers")
    parser.add_argument("--full-warmup", action="store_true", help="Allow up to 512 tokens on every warmup request")
    parser.add_argument("--budgets", nargs="+", type=int, choices=CONFIG["budgets"])
    parser.set_defaults(max_new_tokens=CONFIG["max_new_tokens"])
    args = parser.parse_args()
    if args.out and Path(args.out).exists():
        parser.error("Choose a new output directory")
    if (args.method != "dflash" or not args.runtime_manifest or len(native.resolve_gpus(args)) != 1
            or args.speculative_tokens != CONFIG["speculative_tokens"]
            or args.max_new_tokens != CONFIG["max_new_tokens"] or args.max_model_len != CONFIG["max_model_len"] or args.system_prompt
            or args.chat_template_kwargs != "{}" or args.images_dir
            or args.sd_cudagraph_mode != "PIECEWISE"):
        parser.error("Use the admitted TP1 spatial DFlash B8 runtime and frozen generation parameters")
    if args.phase != "smoke" and (args.count is not None or args.repeats < CONFIG["repeats"]):
        parser.error("Pilot/full require the predeclared sample count and at least three repeats")
    selected_budgets = args.budgets or CONFIG["budgets"]
    if len(set(selected_budgets)) != len(selected_budgets):
        parser.error("Duplicate visual budgets")
    if args.phase != "smoke" and selected_budgets != CONFIG["budgets"]:
        parser.error("Pilot/full use all five budgets in canonical order")
    target = Path(args.target)
    target_pins = json.loads(TARGET_FILES.read_text())
    unexpected = [path.name for pattern in ("*.safetensors", "*.bin", "*.index.json")
                  for path in target.glob(pattern)
                  if path.name not in target_pins and not path.name.startswith("._")]
    mismatch = [name for name, sha in target_pins.items()
                if not (target / name).is_file() or native.digest(target / name) != sha]
    if unexpected or mismatch:
        parser.error(f"Target files differ from the pinned Qwen3-VL-8B-Instruct revision: {unexpected + mismatch}")
    processor_config = json.loads((target / "preprocessor_config.json").read_text())
    if (processor_config["patch_size"], processor_config["merge_size"]) != (
        CONFIG["target_patch_size"], CONFIG["target_spatial_merge_size"]
    ):
        parser.error("Target image processor patch or merge size differs from the frozen Qwen3-VL configuration")
    manifest_path = Path(args.runtime_manifest).absolute()
    adapter = json.loads(manifest_path.read_text())
    ckpt = (manifest_path.parent / adapter["checkpoint"]).resolve()
    checkpoint_pins = {str((manifest_path.parent / path).resolve()): sha for path, sha in adapter["files"].items()}
    if checkpoint_pins.get(str(ckpt)) != CONFIG["checkpoint_sha256"]:
        parser.error("Runtime manifest must pin the historical 8B DFlash checkpoint")
    from .prepare_runtime import ADAPTER, REJECTION_SAMPLER, REJECTION_SHA256
    sampler = Path(distribution("vllm").locate_file(REJECTION_SAMPLER)).resolve()
    if ((manifest_path.parent / adapter["bootstrap"]).resolve() != ADAPTER / "bootstrap.py"
            or checkpoint_pins.get(str(sampler)) != REJECTION_SHA256
            or native.digest(sampler) != REJECTION_SHA256):
        parser.error("Use a freshly bound historical spatial adapter and upstream greedy verifier")
    source = Path(args.source).resolve()
    manifest = json.loads(source.read_text())
    if manifest["seed"] != CONFIG["sample_seed"] or not manifest["tasks"]:
        parser.error("Invalid workload manifest seed or task scope")
    count = args.count or {"smoke": 2, "pilot": CONFIG["pilot_per_task"], "full": CONFIG["full_per_task"]}[args.phase]
    if count < 1:
        parser.error("Expected a positive sample count")
    if manifest.get("smoke_only") and args.phase != "smoke":
        parser.error("Incomplete local image caches are usable only for smoke")
    if args.phase == "full" and set(manifest["tasks"]) != {"general_vqa", "textvqa", "docvqa", "chartqa", "long_form"}:
        parser.error("Full sweep requires all five workloads")
    if any(len(rows) < count for rows in manifest["tasks"].values()):
        parser.error("One or more workloads are smaller than the predeclared sample count")
    samples = {}
    for task, rows in sorted(manifest["tasks"].items()):
        for row in rows[:count]:
            if row["id"] in samples or row["task"] != task or native.digest(row["image"]) != row["image_sha256"]:
                raise ValueError(f"Duplicate ID or invalid workload image: {row['id']}")
            row["prompt_hash"] = hashlib.sha256(row["question"].encode()).hexdigest()
            samples[row["id"]] = row
    out = Path(args.out).resolve()
    out.mkdir(parents=True)
    native.save(out / "sweep.json", {
        "phase": args.phase, "trace": "diagnostic" if args.diagnostic else "timing",
        "dry_run": args.dry_run, "budgets": selected_budgets, "repeats": args.repeats, "samples": samples,
        "workload_manifest_sha256": native.digest(source), "config_sha256": native.digest(CONFIG_PATH),
        "target_model_id": CONFIG["target"], "target_revision": CONFIG["target_revision"],
        "target_artifact_manifest_sha256": native.digest(TARGET_FILES), "drafter_model_id": CONFIG["drafter"],
        "drafter_checkpoint_sha256": CONFIG["checkpoint_sha256"],
    })
    request_path = out / "requests.jsonl"
    with request_path.open("x") as stream:
        for row in samples.values():
            request = {"id": row["id"], "images": [row["image"]], "category": row["task"],
                       "conversations": [{"from": "human", "value": row["question"]}]}
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
    args.source = str(request_path)
    args.num_prompts = len(samples)
    args.visual_context_trace = "diagnostic" if args.diagnostic else "timing"
    args.formal_timing = args.phase in {"pilot", "full"} and not args.diagnostic
    if args.formal_timing and not args.full_warmup:
        args.short_reference_and_warmup = True
    for budget in selected_budgets:
        args.min_pixels, args.max_pixels = bounds(budget)
        args.out = str(out / f"v{budget}.json")
        if native.run(args):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
