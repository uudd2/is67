#!/usr/bin/env bash

# Common environment for local VLABench / LeRobot dataset checks.
# Usage:
#   source scripts/vlabench_eval_env.sh

export VLABENCH_DATA_ROOT="${VLABENCH_DATA_ROOT:-/home/dm/datasets/vlabench_unified}"
export HF_HOME="${HF_HOME:-/home/dm/datasets/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# Headless MuJoCo rendering can be enabled per command with:
#   MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 ...
