"""CPU checks for the vision-overhead-v1 planner, workloads and packer.

Run on any host::

    PYTHONPATH=methods/research/visual_context_horizon \
      VCH_REPO_ROOT=$PWD python3 -m unittest discover -s tests -v

The visual-KV mask tests need torch and are skipped on machines without it;
they run on the GPU host that executes the waves.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METHODS = REPO / "methods" / "research" / "visual_context_horizon"
os.environ.setdefault("VCH_REPO_ROOT", str(REPO))
if str(METHODS) not in sys.path:
    sys.path.insert(0, str(METHODS))

from matrix import config as cfg  # noqa: E402
from matrix import pack, schedule, visual_kv_hook, workloads  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = cfg.load_config()

    def test_suite_and_target(self):
        self.assertEqual(self.config["suite"], "vision-overhead-v1")
        self.assertEqual(self.config["environment"]["vllm"], "0.22.1")
        self.assertEqual(set(self.config["targets"]), {cfg.VISION_TARGET})
        self.assertEqual(
            tuple(self.config["targets"][cfg.VISION_TARGET]["drafters"]),
            cfg.VISION_DRAFTERS)

    def test_spec_tokens_and_widths(self):
        self.assertEqual(cfg.spec_tokens(self.config, 4), 3)
        self.assertEqual(cfg.spec_tokens(self.config, 8), 7)
        self.assertEqual(cfg.spec_tokens(self.config, 16), 15)
        self.assertEqual(cfg.visual_kv_modes(self.config),
                         ["full", "keep_50", "keep_25", "drop_all"])

    def test_blocked_drafters_are_documented_and_excluded(self):
        blocked = {item["method"] for item in cfg.blocked_methods(self.config)}
        self.assertEqual(blocked, {"dspark", "domino"})
        self.assertEqual(set(cfg.runnable_methods(self.config)),
                         {"dflash", "dflash2", "jetspec", "dflare", "dfly"})

    def test_task_budgets_intersect_task_override(self):
        base = [256, 512, 1024, 2048, 4096, 8192, 16384]
        self.assertEqual(cfg.task_budgets(self.config, "hrbench", base),
                         [256, 512, 1024, 2048, 4096])
        self.assertEqual(cfg.task_budgets(self.config, "mp_docvqa", base),
                         [2048, 4096, 8192, 16384])


class WorkloadTest(unittest.TestCase):
    def setUp(self):
        self.config = cfg.load_config()
        self.manifest = workloads.stub_manifest(self.config)

    def test_every_task_is_materialisable(self):
        for task in cfg.task_order(self.config, bonus=True) + ["text_control"]:
            self.assertIn(task, self.manifest["samples"])

    def test_frame_filter_uses_images_not_frames_field(self):
        eight = workloads.select_samples(
            self.manifest, self.config, "video_mme", 8192, frames=8, limit=5)
        self.assertEqual(len(eight), 5)
        self.assertTrue(all(len(row["images"]) >= 8 for row in eight))

    def test_eligibility_never_upscales(self):
        row = {"native_visual_tokens": [100], "images": ["a.jpg"]}
        self.assertTrue(workloads.eligible(row, 100))
        self.assertFalse(workloads.eligible(row, 101))
        self.assertFalse(workloads.eligible({"images": []}, 1))

    def test_text_control_selects_exact_token_budget(self):
        rows = workloads.select_samples(
            self.manifest, self.config, "text_control", 512)
        self.assertTrue(rows)
        self.assertTrue(all(row["text_target_tokens"] == 512 for row in rows))
        self.assertTrue(all(not row["images"] for row in rows))


class ScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = cfg.load_config()
        cls.manifest = workloads.stub_manifest(cls.config)
        cls.plans = {wave: schedule.build_plan(cls.config, cls.manifest, wave)
                     for wave in cfg.VISION_WAVES}

    def test_all_waves_have_cells(self):
        for wave, plan in self.plans.items():
            self.assertGreater(plan["cell_count"], 0, wave)
            self.assertEqual(plan["cell_count"], len(plan["cells"]))

    def test_wave0_honours_samples_per_cell(self):
        plan = self.plans["wave0_smoke"]
        self.assertTrue(all(cell["sample_count"] == 2 for cell in plan["cells"]))
        self.assertEqual(plan["request_estimate"],
                         sum(2 * cell["repeats"] for cell in plan["cells"]))

    def test_wave1_explicit_lane_mapping(self):
        groups = {tuple(group["gpus"]): group for group in
                  self.plans["wave1_main"]["groups"]}
        self.assertEqual(set(groups), {(gpu,) for gpu in range(8)})
        expected = {
            0: ({"exp1_visual"}, {"ar", "dflash"}),
            1: ({"exp1_visual"}, {"dflash2"}),
            2: ({"exp1_visual"}, {"jetspec"}),
            3: ({"exp1_visual"}, {"dflare", "dfly"}),
            4: ({"exp2_visual_kv"}, {"ar", "dflash", "dflash2"}),
            5: ({"exp2_visual_kv"}, {"dflare"}),
            6: ({"exp3_concurrency", "exp3_concurrency_rescue"}, None),
            7: ({"exp4_profile"}, None),
        }
        for gpu, (experiments, methods) in expected.items():
            cells = groups[(gpu,)]["cells"]
            self.assertEqual({cell["experiment"] for cell in cells}, experiments)
            if methods is not None:
                self.assertEqual({cell["method"] for cell in cells}, methods)

    def test_wave2_covers_full_runnable_matrix(self):
        plan = self.plans["wave2_bonus"]
        methods = {cell["method"] for cell in plan["cells"]}
        self.assertEqual(methods, {"ar"} | set(cfg.runnable_methods(self.config)))
        self.assertEqual({cell["experiment"] for cell in plan["cells"]},
                         {"exp5_bonus", "exp5_bonus_apc", "exp5_bonus_width"})

    def test_apc_on_and_off_exist_for_docvqa(self):
        cells = [cell for cell in self.plans["wave2_bonus"]["cells"]
                 if cell["experiment"] == "exp5_bonus_apc"]
        self.assertTrue(cells)
        self.assertEqual({cell["task"] for cell in cells}, {"docvqa"})
        self.assertEqual({cell["apc"] for cell in cells}, {False, True})

    def test_visual_kv_modes_pair_with_one_ar_run(self):
        cells = [cell for cell in self.plans["wave1_main"]["cells"]
                 if cell["experiment"] == "exp2_visual_kv"
                 and cell["task"] == "docvqa" and cell["budget"] == 512]
        ar = [cell for cell in cells if cell["method"] == "ar"]
        self.assertEqual(len(ar), 1)
        modes = {(cell["method"], cell["visual_kv"]) for cell in cells
                 if cell["method"] != "ar"}
        self.assertEqual(modes, {(method, mode)
                                 for method in ("dflash", "dflash2", "dflare")
                                 for mode in cfg.visual_kv_modes(self.config)})
        for cell in cells:
            self.assertEqual(cell["pair_key"], ar[0]["pair_key"])
            if cell["method"] != "ar":
                self.assertNotEqual(cell["id"], ar[0]["id"])

    def test_ids_are_unique_and_pruned_cells_are_explained(self):
        for wave, plan in self.plans.items():
            ids = [cell["id"] for cell in plan["cells"]]
            self.assertEqual(len(ids), len(set(ids)), wave)
            for cell in plan["pruned"]:
                self.assertTrue(0 <= cell["budget"])

    def test_resume_skips_completed_samples(self):
        cell = next(cell for cell in self.plans["wave0_smoke"]["cells"]
                    if cell["method"] == "dflash")
        samples = workloads.select_samples(
            self.manifest, self.config, cell["task"], cell["budget"],
            frames=cell["frames"] if cell["track"] == "multi_frame" else None,
            limit=cell["samples"].get("limit"))
        self.assertEqual(len(samples), cell["sample_count"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cells" / cell["id"] / "rows.jsonl"
            path.parent.mkdir(parents=True)
            first = samples[0]
            path.write_text(json.dumps({
                "repeat": 0, "sample_id": first["id"], "status": "complete"}) + "\n")
            self.assertEqual(schedule.completed_keys(tmp, cell),
                             {(0, first["id"])})
            missing = schedule.missing_work(tmp, cell, self.manifest, self.config)
            self.assertNotIn(first["id"],
                             [row["id"] for row in missing.get(0, [])])
            self.assertFalse(schedule.is_complete(tmp, cell, self.manifest,
                                                  self.config))
            with path.open("a") as handle:
                for sample in samples[1:]:
                    handle.write(json.dumps({
                        "repeat": 0, "sample_id": sample["id"],
                        "status": "complete"}) + "\n")
            self.assertTrue(schedule.is_complete(tmp, cell, self.manifest,
                                                 self.config))
            self.assertEqual(schedule.missing_work(
                tmp, cell, self.manifest, self.config), {})


@unittest.skipUnless(HAVE_TORCH, "torch is required for mask semantics")
class VisualKVTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.mask = torch.tensor(
            [True, True, True, True, False, True, True, True, True, True])

    def test_full_is_identity(self):
        self.assertIs(visual_kv_hook.restrict_mask(self.mask, "full"), self.mask)
        self.assertIsNone(visual_kv_hook.restrict_mask(None, "keep_50"))

    def test_keep_50_keeps_every_second_visual_position(self):
        kept = visual_kv_hook.restrict_mask(self.mask, "keep_50")
        self.assertEqual(kept.tolist(),
                         [True, False, True, False, False, True, False,
                          True, False, True])

    def test_keep_25_keeps_every_fourth_visual_position(self):
        kept = visual_kv_hook.restrict_mask(self.mask, "keep_25")
        self.assertEqual(kept.tolist(),
                         [True, False, False, False, False, True, False,
                          False, False, True])

    def test_drop_all_is_empty_and_same_shape(self):
        kept = visual_kv_hook.restrict_mask(self.mask, "drop_all")
        self.assertEqual(tuple(kept.shape), tuple(self.mask.shape))
        self.assertFalse(bool(kept.any()))

    def test_unknown_mode_is_rejected(self):
        old = os.environ.get(visual_kv_hook.ENV)
        os.environ[visual_kv_hook.ENV] = "keep_10"
        try:
            with self.assertRaises(ValueError):
                visual_kv_hook.current_mode()
        finally:
            if old is None:
                os.environ.pop(visual_kv_hook.ENV, None)
            else:
                os.environ[visual_kv_hook.ENV] = old


class PackTest(unittest.TestCase):
    def test_stage_strips_weights_secrets_and_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "cells" / "abc").mkdir(parents=True)
            (run / "logs" / "profile").mkdir(parents=True)
            (run / "cells" / "abc" / "rows.jsonl").write_text(
                json.dumps({"status": "complete"}) + "\n")
            (run / "cells" / "abc" / "state.json").write_text("{}")
            (run / "logs" / "weights.safetensors").write_bytes(b"x")
            (run / "logs" / "profile" / "trace.json").write_text("{}")
            (run / "logs" / "token.txt").write_text("hf_secret")
            (run / "config_resolved.json").write_text(
                json.dumps({"suite": "vision-overhead-v1"}))
            (run / "summary.json").write_text(json.dumps({
                "generated": "now", "cell_count": 1, "complete_cells": 1,
                "row_count": 1, "speedups": [], "token_match": []}))
            out = Path(tmp) / "out"
            manifest = pack.stage(run, "wave0_smoke", out)
            self.assertFalse((out / "logs" / "weights.safetensors").exists())
            self.assertFalse((out / "logs" / "token.txt").exists())
            self.assertFalse((out / "logs" / "profile" / "trace.json").exists())
            self.assertTrue((out / "SHA256SUMS").is_file())
            self.assertTrue((out / "MANIFEST.json").is_file())
            self.assertTrue(manifest["stripped"])
            self.assertEqual(json.loads(
                (out / "summary_matrix.json").read_text())["suite"],
                "vision-overhead-v1")
            with tarfile.open(out / "per_request_telemetry.tar.gz") as archive:
                names = archive.getnames()
            self.assertIn("cells/abc/rows.jsonl", names)


if __name__ == "__main__":
    unittest.main()
