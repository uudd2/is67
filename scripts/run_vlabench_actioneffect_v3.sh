#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dm/QWENLA/VLANeXt_migration/VLANeXt"
CONDA_SH="/home/dm/miniconda3/etc/profile.d/conda.sh"
CONFIG="config/vlabench_train_vita_hiermq54_actioneffect_v3_mainview_q4_global24_bs16_config.yaml"
GPU_ID="${1:-0}"

cd "$ROOT_DIR"
source "$CONDA_SH"
conda activate VLANeXt

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec python -m scripts.train --config "$CONFIG"
