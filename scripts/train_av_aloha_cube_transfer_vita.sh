#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VITA

export FLARE_DATASETS_DIR="/media/dm/Elements/VLANeXt_migration/data/AV_ALOHA/converted"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/dm/QWENLA/VLANeXt_migration/VITA:${PYTHONPATH:-}"

/home/dm/miniconda3/envs/vita/bin/python \
  /home/dm/QWENLA/VLANeXt_migration/VLANeXt/scripts/train_av_aloha_disk.py \
  policy=vita \
  policy.flow_net.name=simple_flow_net \
  task=cube_transfer \
  session=av_aloha_cube_transfer_vita \
  device=cuda:0 \
  train.num_workers=2 \
  val.val_online_freq=0 \
  "$@"
