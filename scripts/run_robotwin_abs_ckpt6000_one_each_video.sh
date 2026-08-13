#!/usr/bin/env bash
set -euo pipefail

VLANEXT_ROOT=/home/dm/QWENLA/VLANeXt_migration/VLANeXt
CKPT=/home/dm/QWENLA/VLANeXt_migration/VLANeXt/checkpoints/VLANeXt_vita_hiermq54/robotwin_abs_vlm12_0vis8_0text4_1_4_8_12_dim1024_flow8_ae6_dct_div_fixedgate01_tokengate08_finetune_vlm01_aloha_clean50/checkpoint_6000.pt

GPU_ID=${GPU_ID:-0}
MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
ABS_QPOS_STEP_CLIP_NORM=${ABS_QPOS_STEP_CLIP_NORM:-0.10}
RUN_DIR=${RUN_DIR:-${VLANEXT_ROOT}/robotwin_abs_ckpt6000_one_each_video_$(date +%Y%m%d_%H%M%S)}

mkdir -p "${RUN_DIR}"

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt_RoboTwinEval

cd "${VLANEXT_ROOT}"
source scripts/robotwin_vlanext_eval_env.sh

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export PYTHONDONTWRITEBYTECODE=1

python scripts/eval_robotwin_one_each_video.py \
  --checkpoint "${CKPT}" \
  --output-dir "${RUN_DIR}" \
  --test-num 1 \
  --diffusion-steps 6 \
  --exec-horizon 8 \
  --abs-qpos-step-clip-norm "${ABS_QPOS_STEP_CLIP_NORM}"
