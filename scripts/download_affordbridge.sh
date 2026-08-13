#!/usr/bin/env bash
set -euo pipefail

HF_CLI="/home/dm/miniconda3/envs/VLANeXt/bin/hf"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
TARGET_DIR="/media/dm/Elements/AffordBridge"
LOG_DIR="/home/dm/QWENLA/VLANeXt_migration/VLANeXt/logs"
LOCK_FILE="/tmp/affordbridge_download.lock"

mkdir -p "$TARGET_DIR" "$LOG_DIR"
export HF_ENDPOINT

{
  echo "[$(date '+%F %T')] starting AffordBridge download to $TARGET_DIR via $HF_ENDPOINT"
  flock -n 9 || {
    echo "[$(date '+%F %T')] another AffordBridge download is already running"
    exit 0
  }

  set +e
  timeout --preserve-status 6h "$HF_CLI" download aiozai/AffordBridge \
    --repo-type dataset \
    --local-dir "$TARGET_DIR" \
    --max-workers 2
  status=$?
  set -e
  echo "[$(date '+%F %T')] finished with status $status"
  exit "$status"
} 9>"$LOCK_FILE" >>"$LOG_DIR/affordbridge_download.log" 2>&1
