#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
target_epoch=$(date -d 'tomorrow 01:30' +%s)
now_epoch=$(date +%s)
sleep_seconds=$((target_epoch - now_epoch))
if (( sleep_seconds > 0 )); then
  echo "Waiting ${sleep_seconds}s; V6.1-1 starts at $(date -d @${target_epoch} '+%F %T %Z')."
  sleep "${sleep_seconds}"
fi

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
  --config config/libero_long_train_vita_structuredmq54_edar_lite_v6_1_1_onestep_stage_b_bs16_config.yaml \
  > train_v6_1_1_onestep_gpu1.log 2>&1
