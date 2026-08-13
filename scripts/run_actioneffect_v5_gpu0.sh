#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dm/QWENLA/VLANeXt_migration/VLANeXt"
CONFIG="config/libero_long_train_vita_hiermq54_actioneffect_v5_effectproj512_statecond_rawq256_k4hard2_tokenmix02_compactlog_mainview_q4_cross0_global24_bs16_config.yaml"
LOG="$ROOT/train_actioneffect_v5_tokenmix02_gpu0.log"
PID_FILE="$ROOT/train_actioneffect_v5_tokenmix02_gpu0.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "v5 training is already running as PID $old_pid" >&2
    exit 1
  fi
fi

cd "$ROOT"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

nohup env \
  PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR=/tmp/matplotlib \
  CUDA_VISIBLE_DEVICES=0 \
  python -m scripts.train --config "$CONFIG" \
  > "$LOG" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
echo "Started ActionEffect v5 on GPU0: PID $pid"
echo "Log: $LOG"
