#!/usr/bin/env bash
# Export the frozen drafter, bind the runtime manifest and freeze the workload manifest.
#
# Required environment:
#   TARGET      path to Qwen/Qwen3-VL-8B-Instruct at revision 0c351dd...
#   CHECKPOINT  path to dflash_final.pt (sha256 86b40b55...)
#   DATA_ROOT   directory holding textvqa/, docvqa/, chartqa/, general_vqa/, long_form/
#
#   bash scripts/prepare.sh [--dry-run] [--no-canonical-check]
#
# Existing artifacts are validated and reused; nothing is silently overwritten.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

DRY_RUN=0
CANONICAL_CHECK=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-canonical-check) CANONICAL_CHECK=0 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

log "TARGET=$TARGET"
log "CHECKPOINT=$CHECKPOINT"
log "DATA_ROOT=$DATA_ROOT"
log "DRAFT_DIR=$DRAFT_DIR"
log "RUNTIME_MANIFEST=$RUNTIME_MANIFEST"
log "WORKLOAD_MANIFEST=$WORKLOAD_MANIFEST"

require_dir "$VELADRAFT_DIR" "run scripts/setup.sh first"
[ -d "$METHOD_DST" ] || die "method directory missing; run scripts/setup.sh"
require_dir "$TARGET" "run scripts/download_assets.sh or export TARGET"
require_file "$CHECKPOINT" "run scripts/download_assets.sh or export CHECKPOINT"
require_dir "$DATA_ROOT" "run scripts/download_assets.sh or export DATA_ROOT"

# ---- input data layout -------------------------------------------------------
for task in textvqa docvqa chartqa; do
  require_dir "$DATA_ROOT/$task/images" "missing $task images; run scripts/download_assets.sh"
  require_file "$DATA_ROOT/$task/${task}_v2_prompts.jsonl" "missing $task prompts; run scripts/download_assets.sh"
  require_file "$DATA_ROOT/$task/${task}_v2_targets.jsonl" "missing $task targets; run scripts/download_assets.sh"
done
require_dir "$DATA_ROOT/general_vqa/images" "missing general_vqa images; run scripts/download_assets.sh"
require_file "$DATA_ROOT/general_vqa/vqav2_v2_prompts.jsonl" "missing VQAv2 prompts; run scripts/download_assets.sh"
require_file "$DATA_ROOT/general_vqa/vqav2_v2_targets.jsonl" "missing VQAv2 targets; run scripts/download_assets.sh"
require_dir "$DATA_ROOT/long_form/images" "missing long_form images; run scripts/download_assets.sh"

run_py() {
  PYTHONPATH="$(python_path_env)" "$PYTHON" "$@"
}

export_target_files="$METHOD_SRC/configs/target_files.json"
"$PYTHON" - "$TARGET" "$export_target_files" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
pins = json.loads(Path(sys.argv[2]).read_text())
missing = [name for name in pins if not (target / name).is_file()]
if missing:
    raise SystemExit(f"TARGET is incomplete, missing {len(missing)} pinned file(s): {missing[:5]}")
print(f"target directory contains all {len(pins)} pinned files")
PY

# ---- 1. drafter export -------------------------------------------------------
if [ -f "$DRAFT_DIR/model.safetensors" ] && [ -f "$DRAFT_DIR/config.json" ]; then
  log "reusing drafter export at $DRAFT_DIR"
else
  if [ -e "$DRAFT_DIR" ] && [ -n "$(ls -A "$DRAFT_DIR" 2>/dev/null || true)" ]; then
    die "DRAFT_DIR exists but is not a complete export: $DRAFT_DIR (remove it or set DRAFT_DIR)"
  fi
  if [ "$DRY_RUN" = 1 ]; then
    log "would run export_checkpoint.py $CHECKPOINT $DRAFT_DIR"
  else
    mkdir -p "$(dirname "$DRAFT_DIR")"
    log "exporting drafter checkpoint"
    run_py "$VELADRAFT_DIR/research/qwen3vl8b_native_vllm/spatial_adapter/export_checkpoint.py" "$CHECKPOINT" "$DRAFT_DIR"
  fi
fi

# ---- 2. runtime manifest -----------------------------------------------------
if [ -f "$RUNTIME_MANIFEST" ]; then
  log "validating existing runtime manifest"
  "$PYTHON" - "$RUNTIME_MANIFEST" "$CHECKPOINT" "$DRAFT_DIR" "$REPO_ROOT" "$VELADRAFT_DIR" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import distribution
from pathlib import Path

manifest_path, checkpoint, draft, repo, vendor = (
    Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(),
    Path(sys.argv[4]).resolve(), Path(sys.argv[5]).resolve())
manifest = json.loads(manifest_path.read_text())
config = json.loads((repo / "methods/research/visual_context_horizon/configs/dflash.json").read_text())
sampler_sha = "0e2be02449692e8f4aea22c4637727e8d1fc296fb413e5407b1aa79419bf7b41"

recorded = Path(manifest.get("checkpoint", "")).resolve()
if recorded != checkpoint:
    raise SystemExit(f"runtime manifest pins {recorded}, expected {checkpoint}; remove {manifest_path} or set RUNTIME_MANIFEST")
pins = {str(Path(path).resolve()): sha for path, sha in manifest.get("files", {}).items()}
if pins.get(str(checkpoint)) != config["checkpoint_sha256"]:
    raise SystemExit(f"runtime manifest checkpoint hash is not the frozen B8 checkpoint; remove {manifest_path}")

expected_bootstrap = (vendor / "research/qwen3vl8b_native_vllm/spatial_adapter/bootstrap.py").resolve()
if Path(manifest.get("bootstrap", "")).resolve() != expected_bootstrap:
    raise SystemExit(f"runtime manifest bootstrap is not {expected_bootstrap}; remove {manifest_path}")

sampler = Path(distribution("vllm").locate_file("vllm/v1/sample/rejection_sampler.py")).resolve()
if pins.get(str(sampler)) != sampler_sha or hashlib.sha256(sampler.read_bytes()).hexdigest() != sampler_sha:
    raise SystemExit("runtime manifest does not pin the upstream greedy verifier; rerun scripts/setup.sh")
if not (draft / "model.safetensors").is_file():
    raise SystemExit(f"drafter export is incomplete: {draft}")
print("runtime manifest verified (checkpoint, adapter bootstrap and greedy verifier)")
PY
else
  if [ "$DRY_RUN" = 1 ]; then
    log "would run prepare_runtime.py $CHECKPOINT $RUNTIME_MANIFEST"
  else
    mkdir -p "$(dirname "$RUNTIME_MANIFEST")"
    log "writing runtime manifest"
    run_py -m research.visual_context_horizon.prepare_runtime "$CHECKPOINT" "$RUNTIME_MANIFEST"
  fi
fi

# ---- 3. workload manifest ----------------------------------------------------
if [ -f "$WORKLOAD_MANIFEST" ]; then
  log "validating existing workload manifest"
  "$PYTHON" - "$WORKLOAD_MANIFEST" "$REPO_ROOT" "$DATA_ROOT" "$CANONICAL_CHECK" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, repo, data_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
canonical = sys.argv[4] == "1"
manifest = json.loads(manifest_path.read_text())
tasks = {"general_vqa", "textvqa", "docvqa", "chartqa", "long_form"}
if manifest.get("seed") != 42 or set(manifest.get("tasks", {})) != tasks:
    raise SystemExit(f"workload manifest must use seed 42 and all five tasks; remove {manifest_path}")
if manifest.get("smoke_only"):
    raise SystemExit("workload manifest is marked smoke_only and cannot be used for pilot/full")
for task, rows in manifest["tasks"].items():
    if len(rows) < 100:
        raise SystemExit(f"workload {task} has only {len(rows)} samples; full requires 100")
    for row in rows:
        path = Path(row["image"])
        if not path.is_file():
            raise SystemExit(f"workload image is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["image_sha256"]:
            raise SystemExit(f"workload image hash changed: {path}")
if canonical:
    selected = json.loads((repo / "workloads/selected-500.json").read_text())["tasks"]
    for task, rows in selected.items():
        expected = [row["id"] for row in rows]
        actual = [row["id"] for row in manifest["tasks"][task][:100]]
        if actual != expected:
            raise SystemExit(
                f"workload order for {task} differs from workloads/selected-500.json; "
                "use the pinned data or pass --no-canonical-check for a new sample set")
print("workload manifest verified (hashes and canonical sample order)")
PY
else
  if [ "$DRY_RUN" = 1 ]; then
    log "would run prepare_workloads.py for all five tasks -> $WORKLOAD_MANIFEST"
  else
    mkdir -p "$(dirname "$WORKLOAD_MANIFEST")"
    log "freezing workload manifest"
    args=(--out "$WORKLOAD_MANIFEST")
    for task in textvqa docvqa chartqa general_vqa long_form; do
      args+=(--task "$task=$DATA_ROOT/$task")
    done
    run_py -m research.visual_context_horizon.prepare_workloads "${args[@]}"
  fi
fi

if [ "$DRY_RUN" = 1 ]; then
  log "dry run complete; no files were written"
else
  log "prepare complete"
  log "  drafter export:   $DRAFT_DIR"
  log "  runtime manifest: $RUNTIME_MANIFEST"
  log "  workload manifest:$WORKLOAD_MANIFEST"
fi
