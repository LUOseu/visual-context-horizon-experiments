#!/usr/bin/env bash
# Show progress of a wave: cell states, rows, lane exits, telemetry tail.
#
#   bash scripts/status_8gpu.sh --wave wave1_main
#   bash scripts/status_8gpu.sh --run-dir runs/wave1_main_20261127T101500Z [--json]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

WAVE=""
RUN_DIR=""
JSON=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wave) shift; WAVE="${1:-}" ;;
    --run-dir) shift; RUN_DIR="${1:-}" ;;
    --json) JSON="--json" ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

export VCH_REPO_ROOT="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "$RUN_DIR" ]; then
  [ -n "$WAVE" ] || die "--wave or --run-dir is required"
  RUN_DIR="$("$PYTHON" - "$WAVE" <<'PY'
import glob, os, sys
from matrix import config as cfg
root = cfg.paths(cfg.load_config())["runs"]
matches = sorted(glob.glob(os.path.join(str(root), sys.argv[1] + "*")))
print(matches[-1] if matches else "")
PY
)"
  [ -n "$RUN_DIR" ] || die "no run directory for wave $WAVE under the configured runs root"
fi
[ -d "$RUN_DIR" ] || die "no such run directory: $RUN_DIR"
"$PYTHON" -m matrix.status --run-dir "$RUN_DIR" $JSON
