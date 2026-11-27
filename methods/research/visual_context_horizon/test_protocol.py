import pytest

from research.visual_context_horizon import analyze, collect_runtime_trace as trace


def test_commit_round_assigns_one_observed_time_and_checks_exact_ids():
    expected = [4, 8, 9]
    assert trace.commit_times([([4], 100), ([8, 9], 140)], expected, 80, 150) == [20, 60, 60]
    with pytest.raises(ValueError, match="commit"):
        trace.commit_times([([4], 100), ([9, 8], 140)], expected, 80, 150)


def test_budget_token_count_uses_grid_and_prompt_ids():
    assert trace.visual_count([1, 32, 32], 2, [99] * 256 + [7], 99) == 256
    with pytest.raises(ValueError, match="grid"):
        trace.visual_count([1, 32, 32], 2, [99] * 255, 99)


def test_paired_horizon_uses_common_samples_for_every_budget_and_repeat():
    lengths = {256: {"short": 5, "long": 6}, 512: {"short": 4, "long": 6}}
    groups = {
        (budget, sample, repeat, arm): {"token_commit_ns": list(range(10, 10 + length))}
        for budget, samples in lengths.items()
        for sample, length in samples.items()
        for repeat in (0, 1)
        for arm in ("ar", "sd")
    }
    assert analyze.common_samples(groups, [256, 512], ["short", "long"], 2, 5) == ["long"]
    assert analyze.common_samples(groups, [256, 512], ["short", "long"], 2, 7) == []


def test_qwen3_budget_uses_16_pixel_patches_and_2x2_spatial_merge():
    from research.visual_context_horizon.run_visual_sweep import bounds
    assert bounds(256) == (218 * 32**2, 256 * 32**2)
    assert bounds(1280) == (1088 * 32**2, 1280 * 32**2)


def test_vqav2_uses_official_comma_rule():
    assert analyze.score("general_vqa", "1,,000", ["1000"] * 9 + ["other"]) == 0
    assert analyze.score("general_vqa", "1,000", ["1000"] * 9 + ["other"]) == 1


def test_native_installed_sampler_cannot_replace_approval_pin(tmp_path, monkeypatch):
    import argparse
    import json
    from types import SimpleNamespace
    from vlm_dflash.eval import native

    target = tmp_path / "target"
    target.mkdir()
    (target / "generation_config.json").write_text('{"eos_token_id": 10}')
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "model.safetensors").write_bytes(b"fake draft export")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake test image")
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "1", "images": [image.name],
                                  "conversations": [{"from": "human", "value": "question"}]}) + "\n")
    names = [
        "vllm/engine/arg_utils.py", "vllm/model_executor/models/qwen3_dflash.py",
        "vllm/v1/spec_decode/dflash.py", "vllm/v1/sample/rejection_sampler.py",
    ]
    files = {name: tmp_path / name for name in names}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"approved fixture")
    sampler = files[names[-1]]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text("# fixture")
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"method": "dflash", "scope": "fixture",
                                    "checkpoint": str(checkpoint), "bootstrap": str(bootstrap),
                                    "files": {str(p): native.digest(p) for p in (sampler, checkpoint, bootstrap)}}))
    monkeypatch.setattr(native.importlib.metadata, "distribution",
                        lambda name: SimpleNamespace(version="0.22.1", locate_file=files.__getitem__))
    parser = argparse.ArgumentParser()
    native.add_arguments(parser)
    args = parser.parse_args(["--target", str(target), "--draft", str(draft), "--source", str(source),
                              "--out", str(tmp_path / "out"), "--runtime-manifest", str(manifest),
                              "--num-prompts", "1", "--images-dir", str(tmp_path), "--dry-run"])
    assert native.make_plan(args)["files"][str(sampler.resolve())] == native.digest(sampler)
    sampler.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="Installed runtime source differs from adapter pin"):
        native.make_plan(args)
