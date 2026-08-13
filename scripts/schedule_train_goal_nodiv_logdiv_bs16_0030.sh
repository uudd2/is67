#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt

TARGET="2026-07-07 00:30:00"
GPU_ID=0
MEM_THRESHOLD_MB=5000
MAX_WAIT_CHECKS=144
LOG_FILE="train_libero_goal_vita_hiermq54_nodiv_logdiv_bs16_gpu0.log"
CONFIG="config/libero_goal_train_vita_hiermq54_vlm12_0vis8_0text4_1_4_8_12_dim1024_flow8_ae6_dct_nodiv_logdiv_fixedgate01_tokengate08_finetune_vlm01_bs16_config.yaml"

target_epoch=$(date -d "$TARGET" +%s)
now_epoch=$(date +%s)
sleep_seconds=$((target_epoch - now_epoch))

echo "[scheduler] created_at=$(date '+%F %T %Z') target=$TARGET gpu=$GPU_ID sleep_seconds=$sleep_seconds"
if [ "$sleep_seconds" -gt 0 ]; then
  sleep "$sleep_seconds"
fi

for i in $(seq 1 "$MAX_WAIT_CHECKS"); do
  used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  echo "[scheduler] $(date '+%F %T') gpu=$GPU_ID used_mb=$used_mb threshold_mb=$MEM_THRESHOLD_MB check=$i/$MAX_WAIT_CHECKS"
  if [ "$used_mb" -lt "$MEM_THRESHOLD_MB" ]; then
    break
  fi
  sleep 300
done

used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [ "$used_mb" -ge "$MEM_THRESHOLD_MB" ]; then
  echo "[scheduler] GPU $GPU_ID still busy after waiting; aborting."
  exit 2
fi

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

echo "[scheduler] launching goal training at $(date '+%F %T %Z') on GPU $GPU_ID"
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m scripts.train --config "$CONFIG" >> "$LOG_FILE" 2>&1
