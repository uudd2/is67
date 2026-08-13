#!/usr/bin/env bash

# Usage:
#   conda activate VLANeXt_RoboTwinEval
#   source scripts/robotwin_vlanext_eval_env.sh
#
# This keeps the VLANeXt torch/transformers stack, while exposing only the
# RoboTwin simulation dependencies through a selective overlay.

export VLANEXT_ROOT=/home/dm/QWENLA/VLANeXt_migration/VLANeXt
export ROBOTWIN_ROOT=/home/dm/QWENLA/VLANeXt_migration/RoboTwin

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_vlanext_robotwin}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}

export PATH="$VLANEXT_ROOT/runtime/robotwin_sim_overlay_bin:$PATH"
export PYTHONPATH="$VLANEXT_ROOT/runtime/robotwin_sim_overlay:$ROBOTWIN_ROOT/envs/curobo/src:$ROBOTWIN_ROOT:$VLANEXT_ROOT:${PYTHONPATH:-}"
