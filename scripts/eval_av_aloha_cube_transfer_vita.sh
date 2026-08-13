#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:?Usage: $0 CHECKPOINT_PATH [HYDRA_OVERRIDES...]}"
shift
EVAL_OUTPUT_DIR="${AV_ALOHA_EVAL_DIR:-/home/dm/QWENLA/VLANeXt_migration/VLANeXt/av_aloha_eval_videos/$(basename "$CHECKPOINT_PATH")}"
mkdir -p "$EVAL_OUTPUT_DIR"

cd /home/dm/QWENLA/VLANeXt_migration/VITA

export FLARE_DATASETS_DIR="/media/dm/Elements/VLANeXt_migration/data/AV_ALOHA/converted"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH="/home/dm/QWENLA/VLANeXt_migration/VITA:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

/home/dm/miniconda3/envs/vita/bin/python \
  /home/dm/QWENLA/VLANeXt_migration/VLANeXt/scripts/eval_av_aloha_disk.py \
  policy=vita \
  policy.flow_net.name=simple_flow_net \
  task=cube_transfer \
  session=av_aloha_cube_transfer_vita \
  checkpoint_path="$CHECKPOINT_PATH" \
  eval_dir="$EVAL_OUTPUT_DIR" \
  device=cuda:0 \
  train.num_workers=2 \
  val.eval_n_episodes=50 \
  val.eval_n_envs=10 \
  val.num_viz_videos=4 \
  "$@"
