#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [GPU_ID] [extra vlabench_bench_eval.py arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="/home/dm/QWENLA/VLANeXt_migration/VLANeXt"
CHECKPOINT="$1"
GPU_ID=0
shift 1
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  GPU_ID="$1"
  shift 1
fi

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt_VLABenchEval

cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export VLABENCH_ROOT="$PROJECT_ROOT/third_party/VLABench/VLABench"

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/vlabench_bench_eval.py \
  --checkpoint "$CHECKPOINT" \
  "$@"
