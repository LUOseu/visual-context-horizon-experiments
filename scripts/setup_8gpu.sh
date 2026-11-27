#!/usr/bin/env bash
# Assemble the pinned VelaDraft checkout and verify the 8xA100 runtime.
#
#   bash scripts/setup_8gpu.sh                 assemble + verify + runtime manifest
#   bash scripts/setup_8gpu.sh --assemble-only assemble only
#   bash scripts/setup_8gpu.sh --verify-only   verify an existing assembly
#   bash scripts/setup_8gpu.sh --dry-run       print the plan
#
# Pins: vLLM 0.22.1, transformers 5.11.0, torch 2.11.0+cu130,
# VelaDraft e89526b30b028d396005a3ff045238d829d6c24a, and the rejection
# sampler SHA256 recorded in provenance/runtime_versions_8gpu.json.
# Environment: VELADRAFT_DIR, PYTHON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

VELADRAFT_URL="https://github.com/LUOseu/VelaDraft.git"
VELADRAFT_COMMIT="e89526b30b028d396005a3ff045238d829d6c24a"
VELADRAFT_DIR="${VELADRAFT_DIR:-$REPO_ROOT/vendor/veladraft}"
METHOD_SRC="$REPO_ROOT/methods/research/visual_context_horizon"
METHOD_DST="$VELADRAFT_DIR/research/visual_context_horizon"
export VELADRAFT_DIR

MODE=full
case "${1:-}" in
  ""|--full) MODE=full ;;
  --assemble-only) MODE=assemble ;;
  --verify-only) MODE=verify ;;
  --dry-run) MODE=dry ;;
  -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
  *) die "unknown argument: $1" ;;
esac

need_cmd git
need_cmd "$PYTHON"
log "repository:     $REPO_ROOT"
log "VelaDraft:      $VELADRAFT_DIR @ $VELADRAFT_COMMIT"
log "python:         $PYTHON"

if [ "$MODE" = dry ]; then
  log "would fetch $VELADRAFT_URL at $VELADRAFT_COMMIT"
  log "would copy $METHOD_SRC -> $METHOD_DST"
  log "would verify the seven adapter bootstraps/exporters and the rejection-sampler SHA"
  log "would verify vLLM 0.22.1 / transformers 5.11.0 / torch 2.11.0+cu130"
  log "would write artifacts/manifests/runtime_manifest.json"
  exit 0
fi

assemble() {
  if [ -d "$VELADRAFT_DIR/.git" ]; then
    [ "$(git -C "$VELADRAFT_DIR" rev-parse HEAD)" = "$VELADRAFT_COMMIT" ] || die \
      "existing checkout is at $(git -C "$VELADRAFT_DIR" rev-parse HEAD), expected $VELADRAFT_COMMIT"
  else
    [ -e "$VELADRAFT_DIR" ] && die "$VELADRAFT_DIR exists but is not a git checkout"
    mkdir -p "$(dirname "$VELADRAFT_DIR")"
    git init -q "$VELADRAFT_DIR"
    git -C "$VELADRAFT_DIR" remote add origin "$VELADRAFT_URL"
    git -C "$VELADRAFT_DIR" fetch --depth 1 origin "$VELADRAFT_COMMIT" || die \
      "cannot fetch $VELADRAFT_COMMIT (network?)"
    git -C "$VELADRAFT_DIR" checkout -q FETCH_HEAD
  fi
  require_dir "$METHOD_SRC"
  rm -rf "$METHOD_DST"
  mkdir -p "$METHOD_DST"
  cp -R "$METHOD_SRC/." "$METHOD_DST/"
  find "$METHOD_DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  log "copied research/visual_context_horizon into the checkout"
}

verify_files() {
  VCH_REPO_ROOT="$REPO_ROOT" VCH_VELADRAFT_DIR="$VELADRAFT_DIR" \
  PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon" \
    "$PYTHON" - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
from matrix import config as cfg
config = cfg.load_config()
vendor = cfg.vendor_root()
missing = []
for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
    adapter = entry.get("adapter") or {}
    if not entry.get("runnable", True):
        print("adapter %-8s blocked (%s)" % (method, entry.get("blocked", "")[:60]))
        continue
    for key in ("bootstrap", "export", "manifest"):
        relative = adapter.get(key)
        if not relative:
            missing.append("%s: adapter.%s is not configured" % (method, key))
        elif not (vendor / relative).is_file():
            missing.append("%s: %s" % (method, relative))
if missing:
    sys.exit("required VelaDraft adapter files are missing:\n  " + "\n  ".join(missing))
print("seven adapters checked; five runnable, two documented as blocked")
PY
}

verify_runtime() {
  "$PYTHON" - "$REPO_ROOT" <<'PY'
import importlib.metadata as metadata
import hashlib
import json
import sys
from pathlib import Path
repo = Path(sys.argv[1])
pins = json.loads((repo / "provenance" / "runtime_versions_8gpu.json").read_text())
required = {"vllm": pins["vllm"], "transformers": pins["transformers"]}
problems = []
for name, want in required.items():
    try:
        got = metadata.version(name)
    except metadata.PackageNotFoundError:
        problems.append(f"{name} is not installed (need {want})")
        continue
    if got != want:
        problems.append(f"{name}=={got}, expected {want}")
try:
    import huggingface_hub
    print("huggingface_hub %s" % huggingface_hub.__version__)
except Exception as error:
    problems.append(f"huggingface_hub import failed: {error}")
try:
    import torch
    if not torch.version.cuda:
        problems.append("torch has no CUDA (CPU-only build)")
    if not torch.__version__.startswith(pins["torch"].split("+")[0]):
        problems.append(f"torch=={torch.__version__}, expected {pins['torch']}")
except Exception as error:
    problems.append(f"torch import failed: {error}")
import os
if os.environ.get("VCH_SKIP_REJECTION_CHECK") != "1":
    try:
        import vllm
        path = Path(vllm.__file__).parent / "v1" / "sample" / "rejection_sampler.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pins["vllm_rejection_sampler_sha256"]:
            problems.append(
                "rejection_sampler.py sha256 %s != pinned %s"
                % (digest, pins["vllm_rejection_sampler_sha256"]))
    except Exception as error:
        problems.append(f"cannot verify rejection_sampler.py: {error}")
if problems:
    sys.exit("environment check failed:\n  " + "\n  ".join(problems))
print("runtime OK: vllm %s, transformers %s, torch %s"
      % (required["vllm"], required["transformers"], torch.__version__))
PY
  PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon" VCH_REPO_ROOT="$REPO_ROOT" \
    VCH_VELADRAFT_DIR="$VELADRAFT_DIR" "$PYTHON" -m matrix.prepare runtime >/dev/null
  log "wrote artifacts/manifests/runtime_manifest.json"
}

case "$MODE" in
  assemble) assemble; verify_files ;;
  verify) verify_files; verify_runtime ;;
  full) assemble; verify_files; verify_runtime ;;
esac
log "setup complete"
