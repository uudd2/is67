#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR=${1:?Usage: bash scripts/schedule_robotwin_hard_eval_ckpts.sh CKPT_DIR [GPU_ID] [TEST_NUM] [SEED] [TAG_PREFIX]}
GPU_ID=${2:-0}
TEST_NUM=${3:-100}
SEED=${4:-0}
TAG_PREFIX=${5:-robotwin_official_hard}

# Conservative gates based on the 1506/12000 @ 8h40m estimate on 2026-06-01 21:54:57.
# The script still waits for the checkpoint file to actually exist, so these are only
# "do not start before" guards.
NOT_BEFORE_2000=${NOT_BEFORE_2000:-"2026-06-02 00:55:00"}
NOT_BEFORE_3000=${NOT_BEFORE_3000:-"2026-06-02 06:45:00"}

POLL_SECONDS=${POLL_SECONDS:-300}
EVAL_SCRIPT=/home/dm/QWENLA/VLANeXt_migration/VLANeXt/scripts/eval_robotwin_official_hard.sh

wait_until_time() {
  local target_time="$1"
  local target_ts
  target_ts=$(date -d "${target_time}" +%s)

  while true; do
    local now_ts
    now_ts=$(date +%s)
    if [ "${now_ts}" -ge "${target_ts}" ]; then
      break
    fi
    echo "[$(date '+%F %T')] waiting until ${target_time}"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_ckpt() {
  local ckpt="$1"
  while [ ! -f "${ckpt}" ]; do
    echo "[$(date '+%F %T')] waiting for checkpoint: ${ckpt}"
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date '+%F %T')] found checkpoint: ${ckpt}"
}

run_eval() {
  local step="$1"
  local ckpt="${CKPT_DIR}/checkpoint_${step}.pt"
  local tag="${TAG_PREFIX}_ckpt${step}"

  if [ "${step}" = "2000" ]; then
    wait_until_time "${NOT_BEFORE_2000}"
  elif [ "${step}" = "3000" ]; then
    wait_until_time "${NOT_BEFORE_3000}"
  fi

  wait_for_ckpt "${ckpt}"

  echo "[$(date '+%F %T')] starting RoboTwin official Hard eval: ${ckpt}"
  bash "${EVAL_SCRIPT}" "${ckpt}" "${GPU_ID}" "${TEST_NUM}" "${SEED}" "${tag}"
  echo "[$(date '+%F %T')] finished RoboTwin official Hard eval: ${ckpt}"
}

run_eval 2000
run_eval 3000

