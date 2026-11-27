"""Drafter-side visual KV ablation hook (Exp2 core).

The drafter's cross-attention resampler receives ``image_mask`` - a boolean
tensor marking which positions of the target's hidden states are visual.
Restricting that mask removes visual conditioning from the *drafter* without
touching the prompt, the images or the target model, so the target always
verifies with 100% of its vision.

Modes (env ``VCH_DRAFTER_VISUAL_KV_MODE``):

* ``full``     - unchanged.
* ``keep_50``  - every second visual position (deterministic, uniform).
* ``keep_25``  - every fourth visual position.
* ``drop_all`` - empty mask, i.e. a text-only (HiViS-style) drafter.

Positional striding uses a cumulative count of visual tokens, so it is uniform
even when the visual span is not contiguous.  The mask keeps its shape, which
matters for CUDA graph capture: only values change, never tensor extents.

The patch is applied to ``DFlashVisualLayer`` and
``DFlashPooledMLPVisualLayer`` wherever those classes live (vLLM or the
VelaDraft adapter).  It is installed after the adapter bootstrap has run and
before the first warmup request.
"""
from __future__ import annotations

import os
import sys

ENV = "VCH_DRAFTER_VISUAL_KV_MODE"
MODES = ("full", "keep_50", "keep_25", "drop_all")
_STRIDE = {"keep_50": 2, "keep_25": 4}
_TARGETS = ("DFlashVisualLayer", "DFlashPooledMLPVisualLayer")
_PATCHED = []
_MISSING = []


def current_mode() -> str:
    value = os.environ.get(ENV, "full").strip().lower()
    if value not in MODES:
        raise ValueError("%s=%r is not one of %s" % (ENV, value, MODES))
    return value


def restrict_mask(mask, mode: str):
    """Return a same-shaped mask with only the retained visual positions set."""
    if mode == "full" or mask is None:
        return mask
    import torch
    boolean = mask.to(torch.bool) if hasattr(mask, "to") else mask
    if mode == "drop_all":
        return torch.zeros_like(boolean, dtype=torch.bool)
    stride = _STRIDE[mode]
    # index of each visual position, -1 elsewhere; keep every stride-th one
    rank = torch.cumsum(boolean.long(), dim=-1) - 1
    keep = boolean & (rank >= 0) & (torch.remainder(rank, stride) == 0)
    return keep


def _wrap(cls):
    original = cls.build_context
    if getattr(original, "_vch_visual_kv", False):
        return False
    import functools
    import inspect
    try:
        names = list(inspect.signature(original).parameters)[1:]
    except (TypeError, ValueError):
        names = []

    @functools.wraps(original)
    def build_context(self, *args, **kwargs):
        mode = current_mode()
        if mode != "full":
            if "image_mask" in kwargs:
                kwargs["image_mask"] = restrict_mask(kwargs["image_mask"], mode)
            elif "image_mask" in names:
                index = names.index("image_mask")
                if index < len(args):
                    args = list(args)
                    args[index] = restrict_mask(args[index], mode)
                    args = tuple(args)
            else:  # fall back to the last boolean tensor argument
                for index in range(len(args) - 1, -1, -1):
                    value = args[index]
                    if hasattr(value, "dtype") and str(value.dtype) == "torch.bool":
                        args = list(args)
                        args[index] = restrict_mask(value, mode)
                        args = tuple(args)
                        break
        return original(self, *args, **kwargs)

    build_context._vch_visual_kv = True
    cls.build_context = build_context
    return True


def _find(name: str):
    for module in list(sys.modules.values()):
        cls = getattr(module, name, None)
        if isinstance(cls, type) and hasattr(cls, "build_context"):
            return cls
    return None


def install() -> list:
    """Patch every visual layer class that is already imported (idempotent)."""
    patched = []
    for name in _TARGETS:
        cls = _find(name)
        if cls is None:
            if name not in _MISSING:
                _MISSING.append(name)
            continue
        if _wrap(cls):
            record = "%s.%s" % (cls.__module__, cls.__qualname__)
            _PATCHED.append(record)
            patched.append(record)
    return patched


def ensure_installed(strict: bool = True) -> dict:
    """Install after engine construction; fail loudly if the seam was missed."""
    install()
    mode = current_mode()
    status = {"mode": mode, "patched": list(_PATCHED),
              "missing": [name for name in _TARGETS if name not in
                          {record.split(".")[-1] for record in _PATCHED}]}
    if mode != "full" and not _PATCHED and strict:
        raise RuntimeError(
            "visual KV mode %r is active but no DFlash visual layer was "
            "imported; the ablation would silently run as full vision" % mode)
    return status


def describe() -> dict:
    return {"mode": current_mode(), "patched": list(_PATCHED)}
