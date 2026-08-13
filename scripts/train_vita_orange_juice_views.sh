#!/usr/bin/env bash
set -euo pipefail

VIEW=${1:?Usage: bash scripts/train_vita_orange_juice_views.sh <front|wrist|front_wrist> [GPU_ID] [STEPS]}
GPU_ID=${2:-0}
STEPS=${3:-100000}

VITA_ROOT=/home/dm/QWENLA/VLANeXt_migration/VITA

case "${VIEW}" in
  front)
    SESSION=orange_juice_vita_single_image
    IMAGE_KEYS='[observation.images.image]'
    ;;
  wrist)
    SESSION=orange_juice_vita_wrist_image
    IMAGE_KEYS='[observation.images.wrist_image]'
    ;;
  front_wrist)
    SESSION=orange_juice_vita_front_wrist
    IMAGE_KEYS='[observation.images.image,observation.images.wrist_image]'
    ;;
  *)
    echo "Unknown view: ${VIEW}. Expected front, wrist, or front_wrist." >&2
    exit 2
    ;;
esac

cd "${VITA_ROOT}"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate vita

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" python flare/train.py \
  policy=vita \
  task=libero_object_orange_juice_tfds \
  session="${SESSION}" \
  device=cuda:0 \
  train.steps="${STEPS}" \
  train.batch_size=128 \
  train.num_workers=4 \
  train.log_freq=100 \
  train.save_freq=5000 \
  wandb.enable=true \
  val.num_episodes=20 \
  val.val_offline_freq=1000 \
  val.val_online_freq=0 \
  policy.flow_net.name=simple_flow_net \
  "task.image_keys=${IMAGE_KEYS}"

