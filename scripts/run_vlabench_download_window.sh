#!/usr/bin/env bash
set -euo pipefail

START_HOUR=${START_HOUR:-0}
END_HOUR=${END_HOUR:-6}
DATA_ROOT=${DATA_ROOT:-"$HOME/datasets"}
PROJECT_ROOT=${PROJECT_ROOT:-"/home/dm/QWENLA/VLANeXt_migration/VLANeXt"}
LOG_DIR="$PROJECT_ROOT/data_download_logs"
LOG_FILE="$LOG_DIR/vlabench_unified_$(date +%Y%m%d).log"
PID_FILE="$LOG_DIR/vlabench_unified.pid"

mkdir -p "$LOG_DIR"

hour=$(date +%H)
hour=$((10#$hour))
if (( hour < START_HOUR || hour >= END_HOUR )); then
  echo "$(date '+%F %T') outside download window ${START_HOUR}:00-${END_HOUR}:00; exiting." >> "$LOG_FILE"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE" || true)
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "$(date '+%F %T') download already running with PID=$old_pid; exiting." >> "$LOG_FILE"
    exit 0
  fi
fi

end_epoch=$(date -d "today ${END_HOUR}:00" +%s)
now_epoch=$(date +%s)
seconds_left=$((end_epoch - now_epoch))
if (( seconds_left <= 60 )); then
  echo "$(date '+%F %T') less than 60s left in window; exiting." >> "$LOG_FILE"
  exit 0
fi

echo "$$" > "$PID_FILE"
echo "$(date '+%F %T') starting VLABench download for up to ${seconds_left}s." >> "$LOG_FILE"

set +e
timeout --preserve-status "${seconds_left}s" bash "$PROJECT_ROOT/scripts/download_vlabench_unified.sh" >> "$LOG_FILE" 2>&1
status=$?
set -e

rm -f "$PID_FILE"
echo "$(date '+%F %T') download command exited with status=$status." >> "$LOG_FILE"

if [[ "$status" == "124" || "$status" == "143" ]]; then
  exit 0
fi
exit "$status"
