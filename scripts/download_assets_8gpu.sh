#!/usr/bin/env bash
# Download and SHA-verify every 8-GPU asset.
#
#   bash scripts/download_assets_8gpu.sh [--models-only|--data-only|--verify-only|--dry-run]
#
# Parallel downloads (JOBS=4 by default) go to the paths resolved from
# configs/vision_overhead_8gpu.yaml.  HF_TOKEN and HF_ENDPOINT are honoured; nothing is
# written into the repository except caches under the configured asset roots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODE=all
JOBS="${JOBS:-4}"
SKIP_QA=0
for arg in "$@"; do
  case "$arg" in
    --models-only) MODE=models ;;
    --data-only) MODE=data ;;
    --verify-only) MODE=verify ;;
    --dry-run) MODE=dry ;;
    --skip-qa-tasks) SKIP_QA=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

need_cmd "$PYTHON"
export VCH_REPO_ROOT="$REPO_ROOT"
PY_MODULE=(env PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon" VCH_REPO_ROOT="$REPO_ROOT" "$PYTHON" -m matrix.download)

ARGS=()
case "$MODE" in
  models) ARGS+=(--models) ;;
  data) ARGS+=(--data) ;;
  verify) ARGS+=(--verify-only) ;;
  dry) ARGS+=(--dry-run) ;;
esac

if [ "$MODE" = dry ]; then
  "${PY_MODULE[@]}" "${ARGS[@]}"
  exit 0
fi

if [ "$MODE" = verify ]; then
  log "verifying every pin without network I/O"
  "${PY_MODULE[@]}" --verify-only --report "$REPO_ROOT/artifacts/download_verify_8gpu.json"
  exit $?
fi

if [ "$MODE" != data ]; then
  log "downloading the pinned Qwen3-VL-8B target and seven drafter checkpoints"
  "${PY_MODULE[@]}" --models --jobs "$JOBS" --report "$REPO_ROOT/artifacts/download_models_8gpu.json"
fi
if [ "$MODE" != models ]; then
  log "downloading HR-Bench, MP-DocVQA and Video-MME pins"
  "${PY_MODULE[@]}" --data --jobs "$JOBS" --report "$REPO_ROOT/artifacts/download_data_8gpu.json"
fi

if [ "$MODE" != models ] && [ "$MODE" != verify ] && [ "$SKIP_QA" -eq 0 ]; then
  require_dir "$VELADRAFT_DIR" "run scripts/setup_8gpu.sh before downloading QA tasks"
  log "restoring the five frozen QA task image trees (legacy pinned downloader)"
  DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/assets/datasets}" \
  CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/assets/cache}" \
    bash "$SCRIPT_DIR/download_assets.sh" --data-only
fi

if [ "$MODE" != models ]; then
  log "re-verifying dataset pins"
  "${PY_MODULE[@]}" --data --verify-only --report "$REPO_ROOT/artifacts/download_data_verify_8gpu.json"
fi
log "download complete"
