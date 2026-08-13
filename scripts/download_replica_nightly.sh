#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/dm/QWENLA/VLANeXt_migration/VLANeXt/Replica-Dataset"
TARGET_DIR="/media/dm/Elements/replica_v1"
LOG_DIR="/home/dm/QWENLA/VLANeXt_migration/VLANeXt/logs"
LOCK_FILE="/tmp/replica_download_nightly.lock"

mkdir -p "$TARGET_DIR" "$LOG_DIR"

cd "$REPO_DIR"

{
  echo "[$(date '+%F %T')] starting Replica download to $TARGET_DIR"
  flock -n 9 || {
    echo "[$(date '+%F %T')] another Replica download is already running"
    exit 0
  }

  timeout --preserve-status 6h ./download.sh "$TARGET_DIR"
  status=$?
  echo "[$(date '+%F %T')] finished with status $status"
  exit "$status"
} 9>"$LOCK_FILE" >>"$LOG_DIR/replica_download_nightly.log" 2>&1
