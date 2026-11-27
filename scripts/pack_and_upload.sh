#!/usr/bin/env bash
# Package one completed stage, refresh STATUS.md/SHA256SUMS, then push a result
# branch when this checkout has permission (otherwise leave a clean bundle).
#
#   bash scripts/pack_and_upload.sh <pilot_v1280|diagnostics|full> [--run-dir DIR] [--no-push]
#
# Model weights (.pt/.safetensors), images, profiler traces and caches are excluded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

STAGE="${1:-}"
shift || true
RUN_DIR_ARG=""
NO_PUSH=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-dir) [ "$#" -ge 2 ] || die "--run-dir needs a value"; RUN_DIR_ARG="$2"; shift 2 ;;
    --no-push) NO_PUSH=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$STAGE" in
  pilot_v1280|diagnostics|full) ;;
  "" ) die "usage: bash scripts/pack_and_upload.sh <pilot_v1280|diagnostics|full> [--run-dir DIR] [--no-push]" ;;
  -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
  *) die "unknown stage: $STAGE" ;;
esac

if [ -n "$RUN_DIR_ARG" ]; then
  RUN_DIR="$RUN_DIR_ARG"
elif [ -f "$RUN_ROOT/$STAGE.latest" ]; then
  RUN_DIR="$(cat "$RUN_ROOT/$STAGE.latest")"
else
  die "no recorded run for $STAGE; pass --run-dir or run scripts/run_stage.sh $STAGE first"
fi
require_dir "$RUN_DIR" "run directory"

TS="$(date -u +%Y%m%dT%H%M%SZ)"

case "$STAGE" in
  pilot_v1280)
    SWEEP="$RUN_DIR/pilot-v1"
    DEST="$REPO_ROOT/results/pilot-checkpoints/v1280"
    require_file "$SWEEP/v1280.json"
    require_dir "$SWEEP/v1280.evidence"
    "$PYTHON" - "$SWEEP/v1280.json" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())
if row.get("status") != "complete" or not row.get("resources_verified"):
    raise SystemExit("v1280 summary is not a complete, resource-verified run")
if row.get("timed_pairs") != 360 or row.get("repeats") != 3:
    raise SystemExit(f"v1280 summary has {row.get('timed_pairs')} pairs / {row.get('repeats')} repeats; expected 360 / 3")
if Path(row.get("evidence", "")).stem != "v1280":
    raise SystemExit("v1280 summary points at unexpected evidence")
print("v1280 summary validated")
PY
    ;;
  diagnostics)
    SWEEP="$RUN_DIR/diagnostic-v1"
    ANALYSIS="$RUN_DIR/diagnostic-v1-analysis"
    DEST="$REPO_ROOT/results/diagnostics"
    require_dir "$SWEEP"
    require_file "$ANALYSIS/diagnostics.json"
    require_file "$ANALYSIS/diagnostic_stages.csv"
    ;;
  full)
    SWEEP="$RUN_DIR/full-v1"
    ANALYSIS="$RUN_DIR/full-v1-analysis"
    FIGURES="$RUN_DIR/full-figures"
    DEST="$REPO_ROOT/results/full"
    require_dir "$SWEEP"
    require_file "$ANALYSIS/REPORT.md"
    require_file "$ANALYSIS/summary.csv"
    require_file "$ANALYSIS/break_even.json"
    require_dir "$FIGURES"
    ;;
esac

if [ -e "$DEST" ]; then
  die "$DEST already exists; move it aside or rerun with a different result tree"
fi

TAR_EXCLUDES=(
  --exclude='*.pt' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.ckpt'
  --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.gif'
  --exclude='*.trace.json' --exclude='profiles' --exclude='*profiles*'
  --exclude='cache' --exclude='inductor' --exclude='__pycache__' --exclude='*.pyc'
  --exclude='*.tmp' --exclude='*.part'
)

mkdir -p "$DEST"
case "$STAGE" in
  pilot_v1280)
    cp "$SWEEP/v1280.json" "$DEST/summary.json"
    tar -czf "$DEST/native-evidence.tar.gz" -C "$SWEEP" "${TAR_EXCLUDES[@]}" v1280.json v1280.evidence
    ;;
  diagnostics)
    cp "$ANALYSIS/diagnostics.json" "$ANALYSIS/diagnostic_stages.csv" "$DEST/"
    tar -czf "$DEST/native-evidence.tar.gz" -C "$RUN_DIR" "${TAR_EXCLUDES[@]}" diagnostic-v1 diagnostic-v1-analysis
    ;;
  full)
    cp "$ANALYSIS/REPORT.md" "$ANALYSIS/summary.csv" "$ANALYSIS/break_even.json" "$DEST/"
    mkdir -p "$DEST/figures"
    cp "$FIGURES"/* "$DEST/figures/" 2>/dev/null || true
    [ -n "$(ls -A "$DEST/figures" 2>/dev/null || true)" ] || die "no figures were produced in $FIGURES"
    tar -czf "$DEST/native-evidence.tar.gz" -C "$RUN_DIR" "${TAR_EXCLUDES[@]}" full-v1 full-v1-analysis
    ;;
esac
[ -f "$RUN_DIR/RUN_META.json" ] && cp "$RUN_DIR/RUN_META.json" "$DEST/run_meta.json"

case "$STAGE" in
  pilot_v1280)
    printf '# pilot-v1280\n\n[摘要](summary.json) · [原始证据](native-evidence.tar.gz) · [运行元数据](run_meta.json)\n' > "$DEST/ARCHIVE.md" ;;
  diagnostics)
    printf '# diagnostics\n\n[诊断阶段表](diagnostic_stages.csv) · [验证事件汇总](diagnostics.json) · [原始证据](native-evidence.tar.gz) · [运行元数据](run_meta.json)\n' > "$DEST/ARCHIVE.md" ;;
  full)
    printf '# full\n\n[实验报告](REPORT.md) · [汇总指标](summary.csv) · [生成盈亏点](break_even.json) · [图表](figures/) · [原始证据](native-evidence.tar.gz) · [运行元数据](run_meta.json)\n' > "$DEST/ARCHIVE.md" ;;
esac

GPU="$(gpu_name || true)"
if [ -z "$GPU" ] && [ -f "$RUN_DIR/RUN_META.json" ]; then
  GPU="$(json_get "$RUN_DIR/RUN_META.json" gpu_name 2>/dev/null || true)"
fi
[ -n "$GPU" ] || GPU="unknown-gpu"
SLUG="$(slugify "$GPU")"
[ -n "$SLUG" ] || SLUG="unknown-gpu"
BRANCH="results/$STAGE-$SLUG"

# ---- STATUS.md ---------------------------------------------------------------
if [ "$STAGE" = "pilot_v1280" ]; then
  RESULT_LINK="results/pilot-checkpoints/v1280"
  DETAIL="E2E 加速 $(json_get "$DEST/summary.json" speedup_sum_ar_over_sd)×，token 一致 $(json_get "$DEST/summary.json" exact_output_pairs)/360"
else
  RESULT_LINK="results/$STAGE"
  DETAIL="见 [报告]($RESULT_LINK/ARCHIVE.md)"
fi
GPU_ID_LABEL="$GPU_ID"
if [ -f "$RUN_DIR/RUN_META.json" ]; then
  GPU_ID_LABEL="$(json_get "$RUN_DIR/RUN_META.json" gpu_id 2>/dev/null || echo "$GPU_ID")"
fi
STATUS_LINE="- $TS $STAGE on ${GPU}（GPU ${GPU_ID_LABEL}）：$DETAIL · [归档]($RESULT_LINK) · 分支 \`$BRANCH\`"
if ! grep -q '^## 交接运行记录$' "$REPO_ROOT/STATUS.md" 2>/dev/null; then
  printf '\n## 交接运行记录\n\n' >> "$REPO_ROOT/STATUS.md"
fi
printf '%s\n' "$STATUS_LINE" >> "$REPO_ROOT/STATUS.md"

# ---- SHA256SUMS (tracked files, excluding the index itself) -------------------
cd "$REPO_ROOT"
git add -- results STATUS.md
"$PYTHON" - "$REPO_ROOT" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
names = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], check=True,
                       capture_output=True).stdout.split(b"\0")
rows = []
for raw in names:
    if not raw:
        continue
    rel = raw.decode()
    if rel == "SHA256SUMS":
        continue
    path = repo / rel
    if not path.is_file():
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(f"{digest}  ./{rel}\n")
(repo / "SHA256SUMS").write_text("".join(sorted(rows)))
PY
git add -- SHA256SUMS

if git diff --cached --quiet; then
  log "no repository changes to commit"
else
  git commit -q -m "Archive $STAGE results from $GPU"
  log "committed $(git rev-parse --short HEAD)"
fi

# ---- push or bundle ----------------------------------------------------------
PUSHED=0
if [ "$NO_PUSH" = 0 ] && git remote get-url origin >/dev/null 2>&1; then
  if GIT_TERMINAL_PROMPT=0 git push --dry-run origin "HEAD:refs/heads/$BRANCH" >/dev/null 2>&1; then
    log "pushing $BRANCH"
    GIT_TERMINAL_PROMPT=0 git push origin "HEAD:refs/heads/$BRANCH"
    PUSHED=1
    log "pushed: $(git remote get-url origin | sed 's#\.git$##')/tree/$BRANCH"
  else
    warn "no push permission for $BRANCH; generating a bundle instead"
  fi
else
  warn "push skipped; generating a bundle"
fi

if [ "$PUSHED" = 0 ]; then
  TMP="$(mktemp -d)"
  PKG_NAME="visual-context-horizon-results-$STAGE-$SLUG"
  PKG="$TMP/$PKG_NAME"
  mkdir -p "$PKG"
  cp -R "$REPO_ROOT/results" "$PKG/results"
  cp "$REPO_ROOT/STATUS.md" "$REPO_ROOT/SHA256SUMS" "$PKG/"
  [ -f "$RUN_DIR/RUN_META.json" ] && cp "$RUN_DIR/RUN_META.json" "$PKG/RUN_META.json"
  "$PYTHON" - "$PKG/PACK_MANIFEST.json" "$STAGE" "$GPU" "$BRANCH" "$(git rev-parse HEAD)" "$TS" <<'PY'
import json, sys
path, stage, gpu, branch, commit, stamp = sys.argv[1:]
open(path, "w").write(json.dumps(dict(
    stage=stage, gpu=gpu, intended_branch=branch, git_commit=commit, packed_utc=stamp,
    extract_at="repository root (tar -xzf ... -C /path/to/visual-context-horizon-experiments)"),
    ensure_ascii=False, indent=2) + "\n")
PY
  cat > "$PKG/README.txt" <<EOF
Results bundle for stage: $STAGE
GPU: $GPU
Git commit in the producer checkout: $(git rev-parse HEAD)
Intended branch: $BRANCH

Extract at the repository root:
  tar -xzf $PKG_NAME.tar.gz -C /path/to/visual-context-horizon-experiments --strip-components=1

Then verify with: sha256sum -c SHA256SUMS
EOF
  BUNDLE="$REPO_ROOT/results-$STAGE-$SLUG-$TS.tar.gz"
  tar -czf "$BUNDLE" -C "$TMP" "$PKG_NAME"
  rm -rf "$TMP"
  log "bundle ready: $BUNDLE"
  log "send this file back; no GitHub permission is required"
fi

log "packaged $STAGE: $DEST"
