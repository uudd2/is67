#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES=1

python -m scripts.train \
  --config config/libero_long_train_vita_hiermq54_edar_lite_v6_dinov2large_stage_b_bs16_config.yaml
