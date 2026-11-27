"""Exp5 isolated profiling: six buckets from one torch.profiler trace.

Only Exp5 cells use this module.  Exp1-4 must never synchronize a device; the
timing there is host wall time around ``LLM.generate``.  Here we accept the
profiler's sync cost and attribute device time by kernel-name pattern, which is
the only stage boundary a stock vLLM build exposes without patching kernels.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

BUCKETS = ("vit", "target_prefill", "visual_conditioning", "draft", "verify",
           "resync_host")

# First match wins.
_RULES = (
    ("draft", re.compile(r"draft|dflash|dspark|dflare|markov|sweep|eagle", re.I)),
    ("verify", re.compile(r"reject|verify|spec_decode|bonus|argmax_reduction", re.I)),
    ("vit", re.compile(r"vision|(^|[^a-z])vit([^a-z]|$)|patch_embed|pos_embed|"
                       r"image_encoder|qwen[23]_vl|merger", re.I)),
    ("visual_conditioning", re.compile(r"context|resync|visual_kv|condition|cross_attn", re.I)),
    ("target_prefill", re.compile(r"attn|attention|moe|gemm|linear|matmul|rmsnorm|"
                                  r"layernorm|rotary|silu|activation|fused", re.I)),
)
_KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def classify(name: str) -> str:
    for bucket, rule in _RULES:
        if rule.search(name):
            return bucket
    return "resync_host"


def _ttft_ms(output):
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    first = getattr(metrics, "first_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    if first is None or arrival is None:
        return None
    value = (first - arrival) * 1000.0
    return value if value >= 0 else None


class StageProfiler:
    def __init__(self, trace_dir=None, keep_traces: int = 1, enabled: bool = True):
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.keep_traces = max(0, int(keep_traces))
        self.enabled = bool(enabled)
        self.last = {"status": "not_run"}
        self._kept = 0

    def generate(self, llm, requests, params, cell, samples):
        if not self.enabled:
            return llm.generate(requests, params, use_tqdm=False)
        import torch
        try:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False) as prof:
                started = time.perf_counter_ns()
                outputs = llm.generate(requests, params, use_tqdm=False)
                ended = time.perf_counter_ns()
        except Exception as error:  # profiling must not kill the cell
            self.last = {"status": "unavailable", "error": str(error)[:300]}
            return llm.generate(requests, params, use_tqdm=False)
        self.last = self._summarize(prof, cell, outputs, (ended - started) / 1e6)
        return outputs

    def _summarize(self, prof, cell, outputs, wall_ms):
        tmp = None
        events = None
        try:
            directory = self.trace_dir or Path("/tmp")
            directory.mkdir(parents=True, exist_ok=True)
            tmp = directory / ("_trace_%s_%d.json" % (cell["id"], time.time_ns()))
            prof.export_chrome_trace(str(tmp))
            events = json.loads(tmp.read_text(encoding="utf-8")).get("traceEvents", [])
        except Exception as error:
            self.last = {"status": "unavailable", "error": str(error)[:300],
                         "wall_ms": wall_ms}
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            return self.last
        totals = {name: 0.0 for name in BUCKETS}
        kernels = 0
        for event in events:
            if event.get("ph") != "X" or str(event.get("cat", "")).lower() not in _KERNEL_CATS:
                continue
            duration = event.get("dur")
            if not duration:
                continue
            kernels += 1
            totals[classify(str(event.get("name", "")))] += float(duration) / 1000.0
        device_ms = sum(totals.values())
        trace_path = None
        if tmp is not None and self._kept < self.keep_traces and self.trace_dir is not None:
            target = self.trace_dir / ("trace_%s.json" % cell["id"])
            try:
                shutil.move(str(tmp), str(target))
                trace_path = str(target)
                self._kept += 1
            except Exception:
                pass
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)
        ttfts = [value for value in (_ttft_ms(o) for o in outputs) if value is not None]
        prefill = (sum(ttfts) / len(ttfts)) if ttfts else None
        return {
            "status": "complete" if kernels else "no_kernels",
            "buckets_ms": {name: round(value, 3) for name, value in totals.items()},
            "device_kernel_ms": round(device_ms, 3),
            "kernel_count": kernels,
            "wall_ms": round(wall_ms, 3),
            "prefill_wall_ms": None if prefill is None else round(prefill, 3),
            "decode_wall_ms": None if prefill is None else round(max(0.0, wall_ms - prefill), 3),
            "trace_path": trace_path,
            "attribution": "first matching kernel-name pattern; remainder is resync_host",
        }
