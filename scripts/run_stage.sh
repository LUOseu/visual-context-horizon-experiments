#!/usr/bin/env bash
# Run one frozen experiment stage.
#
#   bash scripts/run_stage.sh smoke        AR/SD pipeline check (2 samples, budgets 256/1280, 1 repeat)
#   bash scripts/run_stage.sh pilot_v1280  complete the pilot's final 1280 budget (24 per task, 3 repeats)
#   bash scripts/run_stage.sh diagnostics  pilot-scale profiling + verifier counters
#   bash scripts/run_stage.sh full         100 per task, 5 budgets, 3 repeats, then analysis + figures
#
# Environment: GPU_ID=0, TARGET, CHECKPOINT, DATA_ROOT (see scripts/lib.sh).
# All frozen parameters come from methods/research/visual_context_horizon/configs/dflash.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

STAGE="${1:-}"
case "$STAGE" in
  smoke|pilot_v1280|diagnostics|full) ;;
  -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
  "") die "usage: bash scripts/run_stage.sh <smoke|pilot_v1280|diagnostics|full>" ;;
  *) die "unknown stage: $STAGE (use smoke, pilot_v1280, diagnostics or full)" ;;
esac

require_prepared
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be a single non-negative integer (TP=1); got '$GPU_ID'"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$RUN_ROOT/$STAGE-$TS"
LOG="$LOG_DIR/$STAGE-$TS.log"
mkdir -p "$LOG_DIR" "$RUN_DIR"

COMMON=(
  --target "$TARGET"
  --draft "$DRAFT_DIR"
  --runtime-manifest "$RUNTIME_MANIFEST"
  --source "$WORKLOAD_MANIFEST"
  --method dflash
  --gpu "$GPU_ID"
  --speculative-tokens 7
  --max-new-tokens 512
  --max-model-len 8192
  --sd-cudagraph-mode PIECEWISE
)

DRIVER=(-m research.visual_context_horizon.run_visual_sweep)
EXTRA=()
SWEEP=""
ANALYSIS=""
FIGURES=""

case "$STAGE" in
  smoke)
    SWEEP="$RUN_DIR/smoke-v1"
    EXTRA=(--phase smoke --count 2 --budgets 256 1280 --repeats 1)
    ;;
  pilot_v1280)
    SWEEP="$RUN_DIR/pilot-v1"
    DRIVER=("$SCRIPT_DIR/run_budget_completion.py" --budget 1280)
    EXTRA=(--phase pilot --budgets 1280 --repeats 3)
    ;;
  diagnostics)
    SWEEP="$RUN_DIR/diagnostic-v1"
    ANALYSIS="$RUN_DIR/diagnostic-v1-analysis"
    EXTRA=(--phase pilot --diagnostic --budgets 256 512 768 1024 1280 --repeats 3)
    ;;
  full)
    SWEEP="$RUN_DIR/full-v1"
    ANALYSIS="$RUN_DIR/full-v1-analysis"
    FIGURES="$RUN_DIR/full-figures"
    EXTRA=(--phase full --budgets 256 512 768 1024 1280 --repeats 3)
    ;;
esac

log "stage:        $STAGE"
log "GPU:          $GPU_ID ($(gpu_name || echo unknown))"
log "sweep output: $SWEEP"
log "log file:     $LOG"

set +e
(
  cd "$VELADRAFT_DIR"
  PYTHONPATH="$(python_path_env)" "$PYTHON" "${DRIVER[@]}" "${COMMON[@]}" "${EXTRA[@]}" --out "$SWEEP"
) 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || die "sweep failed (exit $RC); inspect $LOG and $RUN_DIR"

if [ "$STAGE" = "full" ]; then
  log "running horizon analysis"
  (
    cd "$VELADRAFT_DIR"
    PYTHONPATH="$(python_path_env)" "$PYTHON" -m research.visual_context_horizon.run_horizon_analysis "$SWEEP" --out "$ANALYSIS"
  ) 2>&1 | tee -a "$LOG"
  log "rendering figures"
  (
    cd "$VELADRAFT_DIR"
    PYTHONPATH="$(python_path_env)" "$PYTHON" -m research.visual_context_horizon.plot_results "$ANALYSIS" --out "$FIGURES"
  ) 2>&1 | tee -a "$LOG"
fi

if [ "$STAGE" = "diagnostics" ]; then
  log "summarizing diagnostic CUDA stages and verifier counters"
  (
    cd "$VELADRAFT_DIR"
    PYTHONPATH="$(python_path_env)" "$PYTHON" -m research.visual_context_horizon.analyze_diagnostics "$SWEEP" --out "$ANALYSIS"
  ) 2>&1 | tee -a "$LOG"
fi

PYTHONPATH="$(python_path_env)" "$PYTHON" - "$RUN_DIR" "$STAGE" "$SWEEP" "$ANALYSIS" "$FIGURES" \
  "$REPO_ROOT" "$VELADRAFT_DIR" "$GPU_ID" "$(gpu_name || true)" "$(gpu_uuid || true)" <<'PY'
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

run_dir, stage, sweep, analysis, figures, repo, vendor, gpu_id, gpu_name, gpu_uuid = sys.argv[1:]

def package(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

def git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None

meta = {
    "stage": stage,
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": package("torch"),
    "transformers": package("transformers"),
    "vllm": package("vllm"),
    "gpu_id": gpu_id,
    "gpu_name": gpu_name or None,
    "gpu_uuid": gpu_uuid or None,
    "repo_commit": git_head(repo),
    "veladraft_commit": git_head(vendor),
    "sweep_dir": str(Path(sweep).resolve()),
    "analysis_dir": str(Path(analysis).resolve()) if analysis else None,
    "figures_dir": str(Path(figures).resolve()) if figures else None,
    "status": "complete",
}
Path(run_dir, "RUN_META.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {Path(run_dir, 'RUN_META.json')}")
PY

mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/$STAGE.latest"

log "stage complete"
log "  run directory: $RUN_DIR"
log "  next: bash scripts/pack_and_upload.sh $STAGE"
