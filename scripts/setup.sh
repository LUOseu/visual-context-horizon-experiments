#!/usr/bin/env bash
# Assemble the pinned VelaDraft checkout and verify the native runtime.
#
#   bash scripts/setup.sh                 clone/checkout, copy method, apply patch, strict verify
#   bash scripts/setup.sh --assemble-only assemble only, skip version checks
#   bash scripts/setup.sh --verify-only   verify an existing assembly
#   bash scripts/setup.sh --dry-run       print the plan without writing anything
#
# Environment overrides: VELADRAFT_DIR, PYTHON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODE="full"
case "${1:-}" in
  ""|--full) MODE="full" ;;
  --assemble-only) MODE="assemble" ;;
  --verify-only) MODE="verify" ;;
  --dry-run) MODE="dry" ;;
  -h|--help)
    awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
    exit 0 ;;
  *) die "unknown argument: $1 (use --assemble-only, --verify-only or --dry-run)" ;;
esac

log "repository:     $REPO_ROOT"
log "VelaDraft:      $VELADRAFT_DIR"
log "pinned commit:  $VELADRAFT_COMMIT"
log "python:         $PYTHON"

if [ "$MODE" = "dry" ]; then
  need_cmd git
  need_cmd curl
  if [ -d "$VELADRAFT_DIR/.git" ]; then
    log "existing checkout HEAD: $(git -C "$VELADRAFT_DIR" rev-parse HEAD)"
  else
    log "would shallow-fetch $VELADRAFT_COMMIT from $VELADRAFT_URL"
  fi
  log "would copy $METHOD_SRC -> $METHOD_DST"
  log "would apply $PATCH_FILE (skipped when already applied)"
  log "dry run: no files were changed"
  exit 0
fi

assemble() {
  need_cmd git
  if [ -d "$VELADRAFT_DIR/.git" ]; then
    local actual
    actual="$(git -C "$VELADRAFT_DIR" rev-parse HEAD)"
    [ "$actual" = "$VELADRAFT_COMMIT" ] || die \
      "existing VelaDraft checkout is at $actual, expected $VELADRAFT_COMMIT. Set VELADRAFT_DIR to a fresh path or move the existing checkout away."
  else
    [ -e "$VELADRAFT_DIR" ] && die "$VELADRAFT_DIR exists but is not a VelaDraft git checkout"
    log "shallow-fetching pinned commit"
    mkdir -p "$(dirname "$VELADRAFT_DIR")"
    git init -q "$VELADRAFT_DIR"
    git -C "$VELADRAFT_DIR" remote add origin "$VELADRAFT_URL"
    if ! git -C "$VELADRAFT_DIR" fetch --depth 1 origin "$VELADRAFT_COMMIT"; then
      die "cannot fetch $VELADRAFT_COMMIT from $VELADRAFT_URL; check network access"
    fi
    git -C "$VELADRAFT_DIR" checkout -q FETCH_HEAD
  fi
  [ "$(git -C "$VELADRAFT_DIR" rev-parse HEAD)" = "$VELADRAFT_COMMIT" ] || die "checkout is not at the pinned commit"

  require_dir "$METHOD_SRC"
  if [ -d "$METHOD_DST" ] && diff -r -x '__pycache__' -x '*.pyc' "$METHOD_SRC" "$METHOD_DST" >/dev/null 2>&1; then
    log "method directory already up to date"
  else
    rm -rf "$METHOD_DST"
    mkdir -p "$METHOD_DST"
    cp -R "$METHOD_SRC/." "$METHOD_DST/"
    find "$METHOD_DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    log "copied research/visual_context_horizon"
  fi

  require_file "$PATCH_FILE"
  if git -C "$VELADRAFT_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    log "native-eval.patch already applied"
  elif git -C "$VELADRAFT_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$VELADRAFT_DIR" apply "$PATCH_FILE"
    log "applied native-eval.patch"
  else
    die "native-eval.patch does not apply cleanly to $VELADRAFT_DIR; inspect with: git -C $VELADRAFT_DIR apply --check $PATCH_FILE"
  fi
}

verify_environment() {
  local strict="${1:-strict}"
  PYTHONPATH="$(python_path_env)" VCH_REPO_ROOT="$REPO_ROOT" VCH_VELADRAFT_DIR="$VELADRAFT_DIR" \
    VCH_STRICT="$strict" "$PYTHON" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["VCH_REPO_ROOT"])
vendor = Path(os.environ["VCH_VELADRAFT_DIR"])
failures, warnings = [], []

def check(label, ok, detail=""):
    stream = sys.stdout if ok else sys.stderr
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail and not ok else ''}", file=stream)
    if not ok:
        failures.append(label)

def warn(label, detail=""):
    print(f"  [WARN] {label}{': ' + detail if detail else ''}", file=sys.stderr)
    warnings.append(label)

print("runtime verification")

# Python 3.12.3
py = ".".join(map(str, sys.version_info[:3]))
check("Python 3.12.3", py == "3.12.3", f"found {py}")

# torch 2.11.0 + CUDA 13.0
try:
    import torch
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    ok = torch_version.split("+")[0] == "2.11.0" and cuda_version == "13.0"
    check("PyTorch 2.11.0+cu130", ok, f"found {torch_version} (CUDA {cuda_version})")
    if not torch.cuda.is_available():
        warn("torch.cuda.is_available() is false in this shell; rerun setup in the actual GPU environment")
except Exception as exc:  # pragma: no cover - environment dependent
    check("PyTorch 2.11.0+cu130", False, repr(exc))

# transformers 5.11.0
try:
    tf = metadata.version("transformers")
    check("Transformers 5.11.0", tf == "5.11.0", f"found {tf}")
except metadata.PackageNotFoundError:
    check("Transformers 5.11.0", False, "transformers is not installed")

# vLLM 0.22.1 and its runtime sources
VLLM_FILES = {
    "vllm/v1/sample/rejection_sampler.py": None,
    "vllm/v1/engine/llm_engine.py": None,
    "vllm/v1/engine/output_processor.py": None,
    "vllm/entrypoints/offline_utils.py": None,
    "vllm/model_executor/models/qwen3_vl.py": None,
}
try:
    dist = metadata.distribution("vllm")
    check("vLLM 0.22.1", dist.version == "0.22.1", f"found {dist.version}")
    expected = {}
    manifest = json.loads((repo / "provenance/runtime_manifest.json").read_text())
    for path, sha in manifest["files"].items():
        name = Path(path).name
        for rel in VLLM_FILES:
            if Path(rel).name == name:
                expected[rel] = sha
    for rel in VLLM_FILES:
        path = Path(dist.locate_file(rel)).resolve()
        if not path.is_file():
            check(rel, False, f"missing at {path}")
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        want = expected.get(rel)
        ok = want is None or sha == want
        if rel.endswith("rejection_sampler.py"):
            check("rejection_sampler.py SHA256", ok, f"{sha} != {want}")
        elif not ok:
            warn(f"{rel} differs from vLLM 0.22.1", f"{sha} != {want}")
        else:
            print(f"  [PASS] {rel} matches vLLM 0.22.1")
except metadata.PackageNotFoundError:
    check("vLLM 0.22.1", False, "vllm is not installed (use a separate CUDA runtime environment)")

# Method import and copied files
try:
    import research.visual_context_horizon.prepare_runtime as _pr  # noqa: F401
    print("  [PASS] research.visual_context_horizon imports")
except Exception as exc:
    check("research.visual_context_horizon imports", False, repr(exc))

if failures:
    print("", file=sys.stderr)
    print("setup verification failed. Fix the failures above, then rerun scripts/setup.sh --verify-only.", file=sys.stderr)
    if any(f.startswith("vLLM") or f.endswith(".py") or f.endswith("SHA256") for f in failures):
        print("vLLM check hint: pip install --no-deps --force-reinstall vllm==0.22.1", file=sys.stderr)
    sys.exit(1)
print("setup verification passed" + (f" ({len(warnings)} warning(s))" if warnings else ""))
PY
}

case "$MODE" in
  full)
    assemble
    verify_environment strict
    ;;
  assemble)
    assemble
    log "assembly complete; environment verification skipped (--assemble-only)"
    ;;
  verify)
    require_dir "$VELADRAFT_DIR" "run scripts/setup.sh first"
    [ "$(git -C "$VELADRAFT_DIR" rev-parse HEAD)" = "$VELADRAFT_COMMIT" ] || die "VelaDraft is not at $VELADRAFT_COMMIT"
    [ -d "$METHOD_DST" ] || die "method directory is missing; run scripts/setup.sh"
    git -C "$VELADRAFT_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1 || die "native-eval.patch is not applied"
    verify_environment strict
    ;;
esac
