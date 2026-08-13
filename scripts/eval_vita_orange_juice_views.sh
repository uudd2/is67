#!/usr/bin/env bash
set -euo pipefail

VIEW=${1:?Usage: bash scripts/eval_vita_orange_juice_views.sh <front|wrist|front_wrist> [STEP] [GPU_ID] [NUM_TRIALS]}
STEP=${2:-100000}
GPU_ID=${3:-0}
NUM_TRIALS=${4:-50}

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

CKPT=$(printf "%s/flare_outputs/libero_object_orange_juice_tfds/vita/%s/checkpoints/step_%010d" "${VITA_ROOT}" "${SESSION}" "${STEP}")

if [ ! -d "${CKPT}" ]; then
  echo "Checkpoint directory not found: ${CKPT}" >&2
  exit 1
fi

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate vita

export PYTHONPATH=/home/dm/QWENLA/VLANeXt_migration/VLANeXt/third_party/LIBERO:$PYTHONPATH
export LIBERO_CONFIG_PATH=/home/dm/.libero
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m scripts.vita_libero_eval \
  --checkpoint "${CKPT}" \
  --task-suite libero_object \
  --task-id 9 \
  --num-trials "${NUM_TRIALS}" \
  --image-keys "${IMAGE_KEYS}"
