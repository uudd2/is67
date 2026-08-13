#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Usage: $0 GPU_ID TASK [TASK ...]}"
shift

ROOT="/home/dm/QWENLA/VLANeXt_migration/VLANeXt"
VITA_OUTPUT="/home/dm/QWENLA/VLANeXt_migration/VITA/flare_outputs"
LOG_DIR="${ROOT}/av_aloha_train_logs"
mkdir -p "${LOG_DIR}"

run_task() {
  local task="$1"
  local task_config="${task}"
  local task_name="avaloha_${task}"
  local session="av_aloha_${task}_vita"
  local output_dir="${VITA_OUTPUT}/${task_name}/vita/${session}"
  local final_checkpoint="${output_dir}/checkpoints/step_0000100000/training_state.pt"
  local log_file="${LOG_DIR}/${task}_gpu${GPU_ID}.log"
  local -a overrides

  if [[ -f "${final_checkpoint}" ]]; then
    echo "[skip] ${task}: final checkpoint already exists"
    return
  fi

  case "${task}" in
    cube_transfer|hook_package|pour_test_tube|slot_insertion|thread_needle)
      overrides=("task=${task_config}")
      ;;
    peg_insertion)
      overrides=(
        "task=cube_transfer"
        "task.name=${task_name}"
        "task.dataset_repo_id=iantc104/av_aloha_sim_peg_insertion"
        "task.dataset_root=/media/dm/Elements/VLANeXt_migration/data/AV_ALOHA/converted/iantc104/av_aloha_sim_peg_insertion"
        "task.env_name=peg-insertion-v1"
      )
      ;;
    *)
      echo "Unknown AV-ALOHA task: ${task}" >&2
      exit 2
      ;;
  esac

  if find "${output_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' -print -quit 2>/dev/null | grep -q .; then
    overrides+=("resume=true")
  fi

  echo "[start] ${task} on physical GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    bash "${ROOT}/scripts/train_av_aloha_cube_transfer_vita.sh" \
      "${overrides[@]}" \
      "session=${session}" \
      2>&1 | tee "${log_file}"
  echo "[done] ${task}"
}

for task in "$@"; do
  run_task "${task}"
done
