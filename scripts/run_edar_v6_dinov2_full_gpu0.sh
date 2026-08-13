#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES=0

python scripts/cache_edar_dino_features.py \
  --data-path /media/dm/Elements/VLANeXt_migration/data/LIBERO_modified/libero_10_no_noops/1.0.0 \
  --dataset-name libero_10_no_noops \
  --dino-path pretrained/dinov2-large \
  --cache-dir /media/dm/Elements/VLANeXt_migration/data/edar_cache/libero_10_dinov2large_256_grid8 \
  --batch-size 32

python -m scripts.train_edar_lite \
  --config config/libero_long_train_edar_lite_v6_dinov2large_stage_a_config.yaml

python -m scripts.train \
  --config config/libero_long_train_vita_hiermq54_edar_lite_v6_dinov2large_stage_b_bs16_config.yaml
