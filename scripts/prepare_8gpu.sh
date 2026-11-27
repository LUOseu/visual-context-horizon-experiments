#!/usr/bin/env bash
# Verify pins, export the runnable drafters, materialize the task manifest, and
# record the runtime manifest.
#
#   bash scripts/prepare_8gpu.sh [--dry-run] [--skip-exports] [--allow-missing-data]
#                               [--methods dflash,dflash2,...]
#                               [--skip-missing-drafters] [--lock-from-local]
#
# Checkpoints must already be under $CACHE_DIR/checkpoints/qwen3vl_8b/<method>/.
# --skip-missing-drafters is for development only; production runs fail closed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

DRY=0
SKIP_EXPORTS=0
ALLOW_MISSING=0
SKIP_MISSING=0
METHODS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --skip-exports|--lock-from-local) SKIP_EXPORTS=1 ;;
    --allow-missing-data) ALLOW_MISSING=1 ;;
    --skip-missing-drafters) SKIP_MISSING=1 ;;
    --methods) shift; METHODS="${1:-}" ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need_cmd "$PYTHON"
require_dir "$VELADRAFT_DIR" "run scripts/setup_8gpu.sh first"
[ "$(git -C "$VELADRAFT_DIR" rev-parse HEAD)" = "e89526b30b028d396005a3ff045238d829d6c24a" ] \
  || die "VelaDraft is not at the pinned e89526b commit"

export VCH_REPO_ROOT="$REPO_ROOT" VCH_VELADRAFT_DIR="$VELADRAFT_DIR"
PY_MODULE=(env PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon" \
  VCH_REPO_ROOT="$REPO_ROOT" VCH_VELADRAFT_DIR="$VELADRAFT_DIR" "$PYTHON" -m matrix.prepare)

CHECKPOINTS="${CHECKPOINTS:-${CACHE_DIR:-$REPO_ROOT/assets/cache}/checkpoints}"
COMMON=(--checkpoints "$CHECKPOINTS")
if [ -n "$METHODS" ]; then COMMON+=(--methods "$METHODS"); fi
if [ "$SKIP_MISSING" -eq 1 ]; then COMMON+=(--skip-missing-drafters); fi

if [ "$DRY" -eq 1 ]; then
  log "would verify target and dataset pins"
  log "would export the five runnable drafters from $CHECKPOINTS"
  log "would materialize the 9-task + text-control manifest under \$OUT_ROOT/manifests"
  log "would write exports_manifest.json and runtime_manifest.json"
  log "dspark and domino are documented as blocked at VelaDraft e89526b"
  exit 0
fi

log "verifying target snapshots and dataset pins"
"${PY_MODULE[@]}" verify --report "$REPO_ROOT/artifacts/prepare_verify_8gpu.json" >/dev/null
if [ "$ALLOW_MISSING" -eq 0 ]; then
  "$PYTHON" - "$REPO_ROOT/artifacts/prepare_verify_8gpu.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
bad = [name for name, value in report.get("targets", {}).items() if not value["ok"]]
missing = report.get("data", {}).get("missing", [])
if bad:
    sys.exit("target snapshots incomplete: " + ", ".join(bad))
if missing:
    sys.exit("dataset pins missing (pass --allow-missing-data to defer):\n  "
             + "\n  ".join(missing[:12]))
PY
fi

if [ "$SKIP_EXPORTS" -eq 0 ]; then
  log "exporting runnable drafters (trained bytes unchanged)"
  "${PY_MODULE[@]}" exports "${COMMON[@]}" >/dev/null
else
  log "skipping drafter export; locking existing export directories"
fi

log "materializing the task workload manifest"
"${PY_MODULE[@]}" workloads >/dev/null

log "locking export hashes"
"${PY_MODULE[@]}" lock >/dev/null
log "recording runtime manifest"
"${PY_MODULE[@]}" runtime >/dev/null

log "prepare steps complete"
PYTHONPATH="$REPO_ROOT/methods/research/visual_context_horizon" VCH_REPO_ROOT="$REPO_ROOT" \
  "$PYTHON" - "$REPO_ROOT" <<'PY'
import json, sys
from matrix import config as cfg
config = cfg.load_config()
paths = cfg.paths(config)
manifest_name = config["workloads"].get("manifest_name", "workloads_manifest.json")
workloads_path = paths["manifests"] / manifest_name
exports_path = paths["exports"] / "exports_manifest.json"
runtime_path = paths["manifests"] / "runtime_manifest.json"
print("  exports:           %s" % paths["exports"])
print("  workload manifest: %s" % workloads_path)
print("  runtime manifest:  %s" % runtime_path)
problems = []
if not workloads_path.is_file():
    problems.append("workload manifest missing")
if not runtime_path.is_file():
    problems.append("runtime manifest missing")
exports = json.loads(exports_path.read_text()) if exports_path.is_file() else {"exports": {}}
for method, entry in config["targets"][cfg.VISION_TARGET]["drafters"].items():
    record = exports.get("exports", {}).get(cfg.VISION_TARGET, {}).get(method, {})
    if not entry.get("runnable", True):
        print("  blocked drafter: %-8s %s" % (method, entry.get("blocked", "")[:70]))
    elif not record.get("model_safetensors_sha256"):
        problems.append("drafter %s has no exported weights" % method)
if problems:
    print("[PREPARE BLOCKED] " + "; ".join(problems))
    sys.exit(1)
print("[PREPARE OK] five runnable drafters exported; dspark/domino blocked upstream")
PY
