#!/usr/bin/env bash
# Shared environment and helpers for the handoff scripts.
# shellcheck shell=bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

VELADRAFT_URL="https://github.com/LUOseu/VelaDraft.git"
VELADRAFT_COMMIT="e89526b30b028d396005a3ff045238d829d6c24a"
VELADRAFT_DIR="${VELADRAFT_DIR:-$REPO_ROOT/vendor/veladraft}"

PYTHON="${PYTHON:-python3}"
export PYTHON

TARGET="${TARGET:-$REPO_ROOT/assets/Qwen3-VL-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/assets/dflash_final.pt}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/assets/datasets}"
DRAFT_DIR="${DRAFT_DIR:-$REPO_ROOT/artifacts/dflash_export}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-$REPO_ROOT/artifacts/runtime_manifest.json}"
WORKLOAD_MANIFEST="${WORKLOAD_MANIFEST:-$REPO_ROOT/workloads/manifest.json}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/assets/cache}"

GPU_ID="${GPU_ID:-0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

CONFIG_JSON="$REPO_ROOT/methods/research/visual_context_horizon/configs/dflash.json"
TARGET_FILES_JSON="$REPO_ROOT/methods/research/visual_context_horizon/configs/target_files.json"
PATCH_FILE="$REPO_ROOT/methods/native-eval.patch"
METHOD_SRC="$REPO_ROOT/methods/research/visual_context_horizon"
METHOD_DST="$VELADRAFT_DIR/research/visual_context_horizon"

export VELADRAFT_DIR DRAFT_DIR RUNTIME_MANIFEST WORKLOAD_MANIFEST CACHE_DIR GPU_ID RUN_ROOT LOG_DIR

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [ -f "$1" ] || die "required file not found: $1${2:+ ($2)}"
}

require_dir() {
  [ -d "$1" ] || die "required directory not found: $1${2:+ ($2)}"
}

json_get() {
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]])' "$1" "$2"
}

python_path_env() {
  printf '%s%s' "$VELADRAFT_DIR" "${PYTHONPATH:+:$PYTHONPATH}"
}

gpu_name() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID%%,*}" 2>/dev/null | head -1 | sed 's/[[:space:]]*$//'
  fi
}

gpu_uuid() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_ID%%,*}" 2>/dev/null | head -1 | sed 's/[[:space:]]*$//'
  fi
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | LC_ALL=C tr -cs 'a-z0-9' '-' | sed 's/^-//; s/-$//'
}

require_prepared() {
  require_dir "$VELADRAFT_DIR" "run scripts/setup.sh first"
  [ -d "$METHOD_DST" ] || die "method directory missing in VelaDraft checkout; run scripts/setup.sh"
  require_dir "$TARGET" "run scripts/download_assets.sh or export TARGET"
  require_file "$CHECKPOINT" "run scripts/download_assets.sh or export CHECKPOINT"
  require_dir "$DATA_ROOT" "run scripts/download_assets.sh or export DATA_ROOT"
  require_file "$WORKLOAD_MANIFEST" "run scripts/prepare.sh first"
  require_file "$RUNTIME_MANIFEST" "run scripts/prepare.sh first"
  require_dir "$DRAFT_DIR" "run scripts/prepare.sh first"
}
