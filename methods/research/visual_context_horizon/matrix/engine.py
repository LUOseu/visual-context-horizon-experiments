"""Build one vLLM 0.22.1 engine for a stage.  GPU imports stay lazy.

Each stage runs in a fresh subprocess (see ``stage_worker``).  The adapter
bootstrap for a drafter must execute before torch/vLLM are imported, so
``prepare_process`` is called by the child process before any heavy import.
"""
from __future__ import annotations

import gc
import os
import runpy
import sys
import time
from pathlib import Path

from . import config as cfg
from . import visual_kv_hook

MIB = 1 << 20


class EngineError(RuntimeError):
    pass


class OutOfMemory(EngineError):
    pass


def adapter_entry(config: dict, cell: dict) -> dict:
    if cell["arm"] == "ar":
        return {}
    return cfg.adapter(config, cell["target"], cell["method"])


def stage_environment(config: dict, cell: dict) -> dict:
    """Environment that must be set before torch/vLLM are imported."""
    env = {"VCH_VELADRAFT_DIR": str(cfg.vendor_root()),
           "VCH_REPO_ROOT": str(cfg.repo_root())}
    if cell["arm"] == "sd":
        adapter = adapter_entry(config, cell)
        env.update({str(key): str(value)
                    for key, value in (adapter.get("env") or {}).items()})
        env[config["engine"]["visual_kv_env"]] = cell["visual_kv"]
    else:
        env[config["engine"]["visual_kv_env"]] = "full"
    return env


def apply_environment(env: dict) -> None:
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def bootstrap_adapter(config: dict, cell: dict) -> str:
    """Execute the drafter bootstrap before torch/vLLM are imported."""
    adapter = adapter_entry(config, cell)
    if not adapter:
        return None
    vendor = cfg.vendor_root()
    path = vendor / str(adapter["bootstrap"])
    if not path.is_file():
        raise EngineError(
            "adapter bootstrap missing for %s/%s: %s\n"
            "run scripts/setup_8gpu.sh (VCH_VELADRAFT_DIR=%s)"
            % (cell["target"], cell["method"], path, vendor))
    root = vendor / str(adapter.get("root") or ".")
    for entry in (str(root), str(vendor)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
        existing = os.environ.get("PYTHONPATH") or ""
        if entry not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = entry + (
                os.pathsep + existing if existing else "")
    runpy.run_path(str(path))
    call = adapter.get("bootstrap_call")
    if call:
        import importlib
        module, _, function = call.partition(":")
        getattr(importlib.import_module(module), function)()
    return str(path)


def prepare_process(config: dict, cell: dict) -> dict:
    """Apply adapter env + bootstrap.  Call before importing torch."""
    env = stage_environment(config, cell)
    apply_environment(env)
    bootstrap = bootstrap_adapter(config, cell)
    visual_kv_hook.install()
    return {"environment": env, "bootstrap": bootstrap}


def draft_path(config: dict, cell: dict, exports_manifest: dict) -> str:
    entry = exports_manifest["exports"][cell["target"]][cell["method"]]
    path = entry.get("path")
    if not path or "<pending" in str(path):
        raise EngineError("no exported drafter for %s/%s; run scripts/prepare_8gpu.sh"
                          % (cell["target"], cell["method"]))
    return path


def engine_kwargs(config: dict, cell: dict, exports_manifest: dict) -> dict:
    engine = config["engine"]
    target = config["targets"][cell["target"]]
    batch = max(1, int(cell["batch_size"]))
    per_batch_limit = config["guardrails"]["max_num_batched_tokens_by_batch"]
    max_batched = int(per_batch_limit.get(str(batch), per_batch_limit.get(batch, 32768)))
    kwargs = {
        "model": target["path"],
        "dtype": engine["dtype"],
        "tensor_parallel_size": int(cell["tp"]),
        "max_model_len": engine["max_model_len"],
        "max_num_seqs": batch,
        "max_num_batched_tokens": max_batched,
        "gpu_memory_utilization": engine["gpu_memory_utilization"],
        "enable_prefix_caching": bool(cell["apc"]),
        "enable_chunked_prefill": bool(engine["enable_chunked_prefill"]),
        "mm_processor_cache_gb": engine["mm_processor_cache_gb"],
        "limit_mm_per_prompt": {"image": 64},
        "mm_processor_kwargs": {"min_pixels": 1,
                                "max_pixels": int(engine["max_pixels_cap"])},
        "enforce_eager": False,
        "generation_config": "vllm",
        "disable_log_stats": cell["arm"] == "ar",
        "compilation_config": {"cudagraph_mode": engine["cudagraph_mode"]},
        "attention_backend": engine["attention_backend"],
    }
    if cell["arm"] == "sd":
        kwargs["speculative_config"] = {
            "model": draft_path(config, cell, exports_manifest),
            "method": cfg.spec_method(config),
            "num_speculative_tokens": int(cell["speculative_tokens"]),
            "draft_tensor_parallel_size": int(cell["tp"]),
            "attention_backend": engine["attention_backend"],
            "use_local_argmax_reduction": False,
        }
    return kwargs


class Engine:
    def __init__(self, config: dict, cell: dict, exports_manifest: dict,
                 gpu_label: str, prepared: dict = None):
        self.config = config
        self.cell = cell
        self.gpu_label = gpu_label
        self._exports_manifest = exports_manifest
        self.llm = None
        self.kwargs = None
        self.started = None
        self.prepared = prepared or {}
        self.visual_kv = {"mode": "full"}
        self._prepared = bool(prepared)

    def start(self):
        if not self._prepared:
            self.prepared = prepare_process(self.config, self.cell)
        import torch
        from vllm import LLM
        self.kwargs = engine_kwargs(self.config, self.cell, self._exports_manifest)
        self.started = time.time()
        try:
            self.llm = _build_llm(LLM, dict(self.kwargs))
        except torch.cuda.OutOfMemoryError as error:
            raise OutOfMemory(str(error)) from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                raise OutOfMemory(str(error)) from error
            raise
        # The draft model classes exist only after the engine has imported them.
        self.visual_kv = visual_kv_hook.ensure_installed(strict=False)
        return self

    def verify_visual_kv(self):
        """Re-check after warmup, when every draft module has been imported."""
        self.visual_kv = visual_kv_hook.ensure_installed(
            strict=bool(self.config["engine"].get("visual_kv_strict", True)))
        return self.visual_kv

    def metadata(self) -> dict:
        llm = self.llm
        vllm_config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
        compilation = getattr(vllm_config, "compilation_config", None)
        spec = getattr(vllm_config, "speculative_config", None)
        parallel = getattr(vllm_config, "parallel_config", None)
        meta = {
            "gpu": self.gpu_label,
            "stage_environment": self.prepared.get("environment")
            or stage_environment(self.config, self.cell),
            "adapter_bootstrap": self.prepared.get("bootstrap"),
            "visual_kv": self.visual_kv,
            "drafter_visual_kv_mode": self.cell["visual_kv"],
            "batch_size": self.cell["batch_size"],
            "engine_kwargs": _jsonable(self.kwargs),
            "cuda_graph_requested": not bool(self.kwargs.get("enforce_eager")),
            "enforce_eager": bool(self.kwargs.get("enforce_eager")),
            "cudagraph_mode": _attr(compilation, "cudagraph_mode"),
            "cudagraph_mode_requested": (
                self.kwargs.get("compilation_config") or {}).get("cudagraph_mode"),
            "cudagraph_capture_sizes": _attr(compilation, "cudagraph_capture_sizes"),
            "capture_evidence": "vllm_config; startup log is teed by scripts/run_matrix_8gpu.sh",
            "tensor_parallel": _attr(parallel, "tensor_parallel_size"),
            "speculative_tokens": _attr(spec, "num_speculative_tokens"),
            "draft_model": _attr(spec, "model"),
            "method": _attr(spec, "method"),
            "startup_seconds": (time.time() - self.started) if self.started else None,
        }
        rpc = {}
        for name in ("native_eval_metadata", "backend_metadata", "method_metadata"):
            function = getattr(llm, "collective_rpc", None)
            if not callable(function):
                break
            try:
                rpc[name] = _jsonable(function(name))
            except Exception as error:
                rpc[name] = {"error": str(error)[:300]}
        if rpc:
            meta["worker_rpc"] = rpc
        return meta

    def reset_cache(self):
        if self.cell["apc"]:
            return False
        llm = self.llm
        for name in ("reset_prefix_cache",):
            function = getattr(llm, name, None)
            if callable(function):
                try:
                    function()
                    return True
                except Exception:
                    pass
        return False

    def close(self):
        llm, self.llm = self.llm, None
        if llm is None:
            return
        try:
            core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
            if core is not None:
                core.shutdown()
        except Exception:
            pass
        try:
            del llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _build_llm(LLM, kwargs):
    """vLLM 0.22 accepts attention_backend; tolerate builds that do not."""
    try:
        return LLM(**kwargs)
    except TypeError as error:
        if "attention_backend" not in str(error):
            raise
        kwargs.pop("attention_backend", None)
        spec = kwargs.get("speculative_config")
        if isinstance(spec, dict):
            spec.pop("attention_backend", None)
        return LLM(**kwargs)


def _attr(obj, name):
    value = getattr(obj, name, None)
    if hasattr(value, "value"):
        value = value.value
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, OutOfMemory) or "out of memory" in text
