"""Bind the historical spatial DFlash adapter to the current machine's native runtime."""
import argparse
from importlib.metadata import distribution, version
import json
from pathlib import Path

from vlm_dflash.eval import native

from .run_visual_sweep import CONFIG

ADAPTER = Path(__file__).resolve().parent.parent / "qwen3vl8b_native_vllm" / "spatial_adapter"
REJECTION_SAMPLER = "vllm/v1/sample/rejection_sampler.py"
REJECTION_SHA256 = "0e2be02449692e8f4aea22c4637727e8d1fc296fb413e5407b1aa79419bf7b41"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new manifest path")
    checkpoint = args.checkpoint.resolve()
    if native.digest(checkpoint) != CONFIG["checkpoint_sha256"]:
        parser.error("Checkpoint bytes do not match the original B8 run")
    historical = json.loads((ADAPTER.parent / "dflash_runtime_manifest_v2.json").read_text())
    files = {checkpoint: CONFIG["checkpoint_sha256"]}
    for old_path, sha in historical["files"].items():
        path = ADAPTER / Path(old_path).name
        if path.suffix not in {".py", ".json"}:
            continue
        if native.digest(path) != sha:
            raise ValueError(f"Adapter source differs from historical snapshot: {path}")
        files[path] = sha
    runtime = distribution("vllm")
    if runtime.version != CONFIG["vllm"] or version("transformers") != "5.11.0" or version("torch").split("+")[0] != "2.11.0":
        raise ValueError("Expected historical vLLM 0.22.1, Transformers 5.11.0 and Torch 2.11.0")
    sampler = Path(runtime.locate_file(REJECTION_SAMPLER)).resolve()
    if native.digest(sampler) != REJECTION_SHA256:
        raise ValueError("Native rejection sampler differs from upstream 0.22.1")
    files[sampler] = REJECTION_SHA256
    for name in ("vllm/v1/engine/llm_engine.py", "vllm/v1/engine/output_processor.py",
                 "vllm/entrypoints/offline_utils.py", "vllm/model_executor/models/qwen3_vl.py"):
        path = Path(runtime.locate_file(name)).resolve()
        files[path] = native.digest(path)
    native.save(args.out, dict(
        method="dflash", scope=historical["scope"], bootstrap=str(ADAPTER / "bootstrap.py"),
        checkpoint=str(checkpoint), environment={}, files={str(path): sha for path, sha in files.items()},
        verification="Upstream vLLM greedy target-argmax prefix acceptance, mismatch recovery and target bonus",
    ))


if __name__ == "__main__":
    main()
