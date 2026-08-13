#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH=${1:?Usage: bash scripts/eval_robotwin_official_hard.sh CHECKPOINT_PATH [GPU_ID] [TEST_NUM] [SEED] [CKPT_TAG]}
GPU_ID=${2:-0}
TEST_NUM=${3:-100}
SEED=${4:-0}
CKPT_TAG=${5:-vlanext_official_hard}

ROBOTWIN_ROOT=/home/dm/QWENLA/VLANeXt_migration/RoboTwin
SOCKET_DIR=${ROBOTWIN_ROOT}/policy/VLANeXt

# RoboTwin 2.0 official Hard setting is demo_randomized.
TASK_CONFIG=demo_randomized
INSTRUCTION_TYPE=unseen
DIFFUSION_STEPS=5
EXEC_HORIZON=8
POLICY_ENV=VLANeXt

TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

cd "${SOCKET_DIR}"

for task in "${TASKS[@]}"; do
  echo "[$(date '+%F %T')] official hard eval: ${task}"
  bash eval_socket.sh \
    "${task}" \
    "${TASK_CONFIG}" \
    "${CKPT_TAG}" \
    "${CHECKPOINT_PATH}" \
    "${SEED}" \
    "${GPU_ID}" \
    "${TEST_NUM}" \
    "${INSTRUCTION_TYPE}" \
    "${DIFFUSION_STEPS}" \
    "${EXEC_HORIZON}" \
    "${POLICY_ENV}"
done

