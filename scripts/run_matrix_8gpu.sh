#!/usr/bin/env bash
# Launch one wave (or print its exact GPU assignment).
#
#   bash scripts/run_matrix_8gpu.sh --wave wave0_smoke|wave1_main|wave2_bonus
#                                   [--dry-run] [--resume] [--run-dir DIR]
#                                   [--config FILE] [--retry-failed]
#                                   [--skip-missing-drafters]
#
# wave0_smoke: three smoke lanes that must print [WAVE0 PASS].
# wave1_main:  the fixed eight-lane map (Exp1 x 4, Exp2 x 2, Exp3, Exp4).
# wave2_bonus: OCRBench/InfoVQA, APC and block-width slices balanced over 8 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

WAVE=""
DRY=0
RESUME=0
RUN_DIR=""
CONFIG=""
RETRY=0
SKIP_MISSING=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wave) shift; WAVE="${1:-}" ;;
    --dry-run) DRY=1 ;;
    --resume) RESUME=1 ;;
    --run-dir) shift; RUN_DIR="${1:-}" ;;
    --config) shift; CONFIG="${1:-}" ;;
    --retry-failed) RETRY=1 ;;
    --skip-missing-drafters) SKIP_MISSING=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[ -n "$WAVE" ] || die "--wave is required (wave0_smoke, wave1_main, wave2_bonus)"

need_cmd "$PYTHON"
export VCH_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon${PYTHONPATH:+:$PYTHONPATH}"
if [ "$RETRY" -eq 1 ]; then export VCH_RETRY_FAILED=1; fi

ARGS=(--wave "$WAVE")
if [ "$DRY" -eq 1 ]; then ARGS+=(--dry-run); fi
if [ "$RESUME" -eq 1 ]; then ARGS+=(--resume); fi
if [ -n "$RUN_DIR" ]; then ARGS+=(--run-dir "$RUN_DIR"); fi
if [ -n "$CONFIG" ]; then ARGS+=(--config "$CONFIG"); fi
if [ "$SKIP_MISSING" -eq 1 ]; then ARGS+=(--skip-missing-drafters); fi

if [ "$DRY" -eq 0 ] && [ "$RESUME" -eq 0 ]; then
  "$PYTHON" - "$WAVE" "${CONFIG:-}" <<'PY'
import json, sys
from matrix import config as cfg
wave_name = sys.argv[1]
config = cfg.load_config(sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None)
paths = cfg.paths(config)
name = config["workloads"].get("manifest_name", "workloads_manifest.json")
workloads = paths["manifests"] / name
if not workloads.is_file():
    sys.exit("missing %s; run scripts/prepare_8gpu.sh" % workloads)
exports_path = paths["exports"] / "exports_manifest.json"
if not exports_path.is_file():
    sys.exit("missing %s; run scripts/prepare_8gpu.sh" % exports_path)
manifest = json.load(open(exports_path))
wave = config["waves"][wave_name]
methods = set()
for lane in wave.get("lanes") or []:
    methods.update(lane.get("methods") or [])
if not methods:
    methods.update(config["matrix"]["draft_methods"])
missing = []
for method in sorted(methods):
    entry = config["targets"][cfg.VISION_TARGET]["drafters"].get(method)
    if not entry or method == "ar":
        continue
    if not entry.get("runnable", True):
        continue
    record = manifest.get("exports", {}).get(cfg.VISION_TARGET, {}).get(method, {})
    if not record.get("model_safetensors_sha256"):
        missing.append(method)
if missing:
    sys.exit("exports without weights for wave %s (run prepare_8gpu.sh): %s"
             % (wave_name, ", ".join(missing)))
for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
    if not entry.get("runnable", True):
        print("note: drafter %s is blocked and will be skipped: %s"
              % (method, (entry.get("blocked") or "")[:80]))
PY
fi

mkdir -p "$REPO_ROOT/logs"
log "wave $WAVE"
set +e
"$PYTHON" -m matrix.orchestrator "${ARGS[@]}" 2>&1 | tee -a "$REPO_ROOT/logs/orchestrator_${WAVE}.log"
status=${PIPESTATUS[0]}
set -e
[ "$status" -eq 0 ] || die "wave $WAVE exited $status (summary.json and lane logs remain)"
if [ "$DRY" -eq 0 ]; then
  log "done; inspect with: bash scripts/status_8gpu.sh --wave $WAVE"
fi
