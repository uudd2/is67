#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dm/QWENLA/VLANeXt_migration/VLANeXt
SQUARE_INFO=/media/dm/Elements/VLANeXt_migration/data/RoboMimic_TFDS/robomimic_ph/square_ph_image/1.0.1/dataset_info.json

echo "Waiting for RoboMimic Square TFDS: $SQUARE_INFO"
until [[ -f "$SQUARE_INFO" ]]; do
  sleep 30
done

echo "Square TFDS is ready. Starting three-task training on GPU0."
cd "$ROOT"
exec env \
  PYTHONNOUSERSITE=1 \
  MPLCONFIGDIR=/tmp/matplotlib \
  CUDA_VISIBLE_DEVICES=0 \
  /home/dm/miniconda3/envs/VLANeXt/bin/python -m scripts.train \
  --config config/robomimic_lift_can_square_train_vita_hiermq54_bs16_config.yaml
