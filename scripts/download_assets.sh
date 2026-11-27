#!/usr/bin/env bash
# Download the pinned target model, drafter checkpoint and evaluation data.
#
#   bash scripts/download_assets.sh [--models-only|--data-only|--verify-only]
#
# Layout (override with TARGET / CHECKPOINT / DATA_ROOT / CACHE_DIR):
#   $TARGET      Qwen3-VL-8B-Instruct files pinned by configs/target_files.json
#   $CHECKPOINT  dflash_final.pt (1.26 GiB, sha256 86b40b55...)
#   $DATA_ROOT   {textvqa,docvqa,chartqa}/{images,*_v2_*.jsonl}
#                {general_vqa,long_form}/images (+ VQAv2 prompts/targets)
#   $CACHE_DIR   resumable download cache (safe to delete after a verified run)
#
# Every file is checked against a pinned SHA256; partial downloads resume.
# Data restoration needs the assembled VelaDraft checkout, so run scripts/setup.sh
# before --data-only unless you only need the model files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODE=all
while [ "$#" -gt 0 ]; do
  case "$1" in
    --models-only) MODE=models ;;
    --data-only) MODE=data ;;
    --verify-only) MODE=verify ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need_cmd curl
require_file "$CONFIG_JSON"
require_file "$TARGET_FILES_JSON"

TARGET_REPO="$(json_get "$CONFIG_JSON" target)"
TARGET_REV="$(json_get "$CONFIG_JSON" target_revision)"
DRAFTER_REPO="$(json_get "$CONFIG_JSON" drafter)"
DRAFTER_REV="$(json_get "$CONFIG_JSON" drafter_revision)"
CHECKPOINT_SHA="$(json_get "$CONFIG_JSON" checkpoint_sha256)"
CHECKPOINT_PATH="ckpt/ckpt_final.pt"
HF_ENDPOINT="https://huggingface.co"

hf_url() {
  printf '%s/%s/resolve/%s/%s?download=true' "$HF_ENDPOINT" "$1" "$2" "$3"
}

# fetch_verified URL DEST SHA LABEL — resume into DEST.part, verify, then rename.
fetch_verified() {
  local url="$1" dest="$2" expected="$3" label="${4:-$2}" part attempt
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    if [ "$(sha256_file "$dest")" = "$expected" ]; then
      log "verified $label"
      return 0
    fi
    warn "existing file does not match its pin; re-downloading $label"
    rm -f "$dest"
  fi
  part="$dest.part"
  attempt=1
  while [ "$attempt" -le 5 ]; do
    if [ -f "$part" ] && [ "$(sha256_file "$part")" = "$expected" ]; then
      mv "$part" "$dest"
      log "verified $label"
      return 0
    fi
    log "downloading $label (attempt $attempt)"
    if curl -fL ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        --retry 2 --retry-delay 2 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
        -C - -o "$part" "$url"; then
      if [ "$(sha256_file "$part")" = "$expected" ]; then
        mv "$part" "$dest"
        log "verified $label"
        return 0
      fi
      warn "SHA256 mismatch for $label; restarting the download"
      rm -f "$part"
    else
      warn "download interrupted for $label"
    fi
    sleep $((1 << attempt))
    attempt=$((attempt + 1))
  done
  die "could not retrieve a verified copy of $label"
}

verify_models() {
  local name sha missing=0
  while IFS=$'\t' read -r name sha; do
    [ -n "$name" ] || continue
    if [ ! -f "$TARGET/$name" ]; then
      warn "missing model file: $TARGET/$name"
      missing=1
    elif [ "$(sha256_file "$TARGET/$name")" != "$sha" ]; then
      warn "model file hash mismatch: $TARGET/$name"
      missing=1
    fi
  done < <("$PYTHON" - "$TARGET_FILES_JSON" <<'PY'
import json, sys
for name, sha in json.load(open(sys.argv[1])).items():
    print(f"{name}\t{sha}")
PY
)
  if [ -f "$CHECKPOINT" ]; then
    [ "$(sha256_file "$CHECKPOINT")" = "$CHECKPOINT_SHA" ] || { warn "checkpoint hash mismatch: $CHECKPOINT"; missing=1; }
  else
    warn "missing checkpoint: $CHECKPOINT"
    missing=1
  fi
  [ "$missing" = 0 ] || die "model assets are incomplete; rerun without --verify-only"
  log "model assets verified (14 target files + checkpoint)"
}

download_models() {
  local name sha
  mkdir -p "$TARGET"
  while IFS=$'\t' read -r name sha; do
    [ -n "$name" ] || continue
    fetch_verified "$(hf_url "$TARGET_REPO" "$TARGET_REV" "$name")" "$TARGET/$name" "$sha" "$TARGET_REPO/$name"
  done < <("$PYTHON" - "$TARGET_FILES_JSON" <<'PY'
import json, sys
for name, sha in json.load(open(sys.argv[1])).items():
    print(f"{name}\t{sha}")
PY
)
  mkdir -p "$(dirname "$CHECKPOINT")"
  fetch_verified "$(hf_url "$DRAFTER_REPO" "$DRAFTER_REV" "$CHECKPOINT_PATH")" \
    "$CHECKPOINT" "$CHECKPOINT_SHA" "$DRAFTER_REPO/$CHECKPOINT_PATH"
}

verify_data() {
  "$PYTHON" - "$REPO_ROOT" "$DATA_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

repo, root = Path(sys.argv[1]), Path(sys.argv[2])
selected = json.loads((repo / "workloads/selected-500.json").read_text())["tasks"]
problems = []

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def check_rows(task, targets):
    for row in targets:
        path = root / task / "images" / row["image"]
        if not path.is_file():
            problems.append(f"missing image: {path}")
        elif digest(path) != row["image_sha256"]:
            problems.append(f"image hash mismatch: {path}")

for task in ("textvqa", "docvqa", "chartqa"):
    for suffix in ("prompts", "targets"):
        path = root / task / f"{task}_v2_{suffix}.jsonl"
        if not path.is_file():
            problems.append(f"missing metadata: {path}")
    path = root / task / f"{task}_v2_targets.jsonl"
    if path.is_file():
        check_rows(task, [json.loads(line) for line in path.read_text().splitlines() if line.strip()])

for suffix in ("prompts", "targets"):
    path = root / "general_vqa" / f"vqav2_v2_{suffix}.jsonl"
    if not path.is_file():
        problems.append(f"missing metadata: {path}")
path = root / "general_vqa/vqav2_v2_targets.jsonl"
if path.is_file():
    check_rows("general_vqa", [json.loads(line) for line in path.read_text().splitlines() if line.strip()])

for row in selected["long_form"]:
    path = root / "long_form" / "images" / Path(row["image_relative"]).name
    if not path.is_file():
        problems.append(f"missing image: {path}")
    elif digest(path) != row["image_sha256"]:
        problems.append(f"image hash mismatch: {path}")

if problems:
    for item in problems[:20]:
        print(f"ERROR: {item}", file=sys.stderr)
    raise SystemExit(f"data assets are incomplete ({len(problems)} problem(s))")
print("data assets verified (5 tasks, images and metadata)")
PY
}

download_data() {
  local task name sha
  [ -d "$METHOD_DST" ] || die "run scripts/setup.sh before downloading data (restoration imports VelaDraft)"
  require_file "$REPO_ROOT/workloads/selected-500.json"
  mkdir -p "$CACHE_DIR/qa" "$DATA_ROOT"

  for task in textvqa docvqa chartqa; do
    require_file "$REPO_ROOT/data/qa/$task/manifest.json"
    while IFS=$'\t' read -r name sha; do
      [ -n "$name" ] || continue
      fetch_verified "$(hf_url "$(json_get "$REPO_ROOT/data/qa/$task/manifest.json" dataset)" \
        "$(json_get "$REPO_ROOT/data/qa/$task/manifest.json" revision)" \
        "$(json_get "$REPO_ROOT/data/qa/$task/manifest.json" folder)/$name")" \
        "$CACHE_DIR/qa/$task/$name" "$sha" "$task/$name"
    done < <("$PYTHON" - "$REPO_ROOT/data/qa/$task/manifest.json" <<'PY'
import json, sys
for name, sha in sorted(json.load(open(sys.argv[1]))["shards"].items()):
    print(f"{name}\t{sha}")
PY
)
  done

  log "restoring archived QA image bytes"
  (
    cd "$VELADRAFT_DIR"
    PYTHONPATH="$(python_path_env)" "$PYTHON" -m research.visual_context_horizon.restore_frozen_qa \
      "$REPO_ROOT/data/qa" "$DATA_ROOT" --cache "$CACHE_DIR/qa" \
      --task textvqa --task docvqa --task chartqa
  )

  log "materializing general_vqa and long_form from official COCO JPEGs"
  "$PYTHON" "$REPO_ROOT/scripts/prepare_public_coco.py" \
    --selected "$REPO_ROOT/workloads/selected-500.json" \
    --out "$DATA_ROOT" --cache "$CACHE_DIR/coco"
}

case "$MODE" in
  models) download_models ;;
  data) download_data ;;
  verify) verify_models; verify_data ;;
  all) download_models; download_data; verify_data ;;
esac

log "assets ready"
log "  TARGET=$TARGET"
log "  CHECKPOINT=$CHECKPOINT"
log "  DATA_ROOT=$DATA_ROOT"
