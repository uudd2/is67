#!/usr/bin/env bash
set -Eeuo pipefail

DATA_ROOT="${DATA_ROOT:-/media/dm/Elements/VLANeXt_migration/data/calvin}"
EVAL_ROOT="${EVAL_ROOT:-$DATA_ROOT/CALVIN_eval}"
CALVIN_ROOT="${CALVIN_ROOT:-$EVAL_ROOT/calvin}"
DATASET_ROOT="${DATASET_ROOT:-$EVAL_ROOT/task_ABC_D}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_BIN="${HF_BIN:-/home/dm/miniconda3/bin/hf}"
HF_REPO="wangdong24/calvin_data"
HF_REVISION="737642c40a31d54c8ed95b95007a10f0a8924158"
HF_CONFIG_PATH="task_ABCD_D/validation/.hydra/merged_config.yaml"
HF_CACHE_DIR="$EVAL_ROOT/.calvin_data_config"

CALVIN_GIT_URL="${CALVIN_GIT_URL:-https://github.com/mees/calvin.git}"
GIT_PROXY_PREFIX="${GIT_PROXY_PREFIX:-https://ghfast.top/}"
PARTIAL_ROOT="$EVAL_ROOT/.calvin.partial"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

clone_calvin() {
  local clone_url="$1"
  rm -rf "$PARTIAL_ROOT"
  git clone --recurse-submodules "$clone_url" "$PARTIAL_ROOT"
  mv "$PARTIAL_ROOT" "$CALVIN_ROOT"
}

mkdir -p "$EVAL_ROOT" "$DATASET_ROOT/validation/.hydra" "$HF_CACHE_DIR"

if [[ ! -x "$HF_BIN" ]]; then
  echo "Hugging Face CLI not found: $HF_BIN" >&2
  exit 1
fi

log "Downloading the CALVIN scene-D environment config through $HF_ENDPOINT"
if ! HF_ENDPOINT="$HF_ENDPOINT" "$HF_BIN" download \
  "$HF_REPO" \
  "$HF_CONFIG_PATH" \
  --revision "$HF_REVISION" \
  --local-dir "$HF_CACHE_DIR"; then
  log "HF mirror does not have the small environment config; retrying from the official Hub"
  HF_ENDPOINT="https://huggingface.co" "$HF_BIN" download \
    "$HF_REPO" \
    "$HF_CONFIG_PATH" \
    --revision "$HF_REVISION" \
    --local-dir "$HF_CACHE_DIR"
fi

cp \
  "$HF_CACHE_DIR/$HF_CONFIG_PATH" \
  "$DATASET_ROOT/validation/.hydra/merged_config.yaml"

if [[ -d "$CALVIN_ROOT/.git" ]]; then
  log "CALVIN repository already exists; refreshing submodules"
  git -C "$CALVIN_ROOT" submodule sync --recursive
  if ! git -C "$CALVIN_ROOT" \
    -c "url.${GIT_PROXY_PREFIX}https://github.com/.insteadOf=https://github.com/" \
    submodule update --init --recursive; then
    log "Git proxy failed; retrying submodules from the official source"
    git -C "$CALVIN_ROOT" submodule update --init --recursive
  fi
else
  log "Cloning CALVIN through the GitHub proxy"
  if ! clone_calvin "${GIT_PROXY_PREFIX}${CALVIN_GIT_URL}"; then
    log "Git proxy failed; retrying from the official GitHub source"
    clone_calvin "$CALVIN_GIT_URL"
  fi
fi

CONFIG_FILE="$DATASET_ROOT/validation/.hydra/merged_config.yaml"
if [[ ! -s "$CONFIG_FILE" ]]; then
  echo "Missing CALVIN evaluation config: $CONFIG_FILE" >&2
  exit 1
fi
if ! grep -q 'calvin_scene_D' "$CONFIG_FILE"; then
  echo "The downloaded config does not describe CALVIN scene D." >&2
  exit 1
fi
if [[ ! -d "$CALVIN_ROOT/calvin_env" || ! -d "$CALVIN_ROOT/calvin_models" ]]; then
  echo "CALVIN repository or its submodules are incomplete: $CALVIN_ROOT" >&2
  exit 1
fi

log "CALVIN D evaluation assets downloaded successfully"
printf 'CALVIN_ROOT=%s\n' "$CALVIN_ROOT"
printf 'CALVIN_DATASET_PATH=%s\n' "$DATASET_ROOT"
printf 'CALVIN_ENV_CONFIG=%s\n' "$CONFIG_FILE"
