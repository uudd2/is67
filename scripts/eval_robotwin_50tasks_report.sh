#!/usr/bin/env bash
set -uo pipefail

ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/home/dm/QWENLA/VLANeXt_migration/RoboTwin}
CKPT=${CKPT:-/home/dm/QWENLA/VLANeXt_migration/VLANeXt/checkpoints/VLANeXt_robotwin_ablation/nofilm_aloha_clean50/checkpoint_4000.pt}
CKPT_TAG=${CKPT_TAG:-nofilm_ckpt4000_clean20}

GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-20}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
INSTRUCTION_TYPE=${INSTRUCTION_TYPE:-unseen}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-5}
EXEC_HORIZON=${EXEC_HORIZON:-8}
SERVER_ENV=${SERVER_ENV:-VLANeXt}
CLIENT_ENV=${CLIENT_ENV:-RoboTwin}

RUN_NAME=${RUN_NAME:-robotwin_50tasks_${CKPT_TAG}_$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-/home/dm/QWENLA/VLANeXt_migration/VLANeXt/eval_logs/${RUN_NAME}}
REPORT_TSV=${REPORT_TSV:-${LOG_ROOT}/summary.tsv}
REPORT_MD=${REPORT_MD:-${LOG_ROOT}/summary.md}

ORIG_EVAL_SOCKET="${ROBOTWIN_ROOT}/policy/VLANeXt/eval_socket.sh"
TMP_EVAL_SOCKET="/tmp/vlanext_eval_socket_client_${CLIENT_ENV}_$$.sh"

mkdir -p "${LOG_ROOT}"

if [[ ! -f "${ORIG_EVAL_SOCKET}" ]]; then
  echo "[error] missing ${ORIG_EVAL_SOCKET}"
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing checkpoint: ${CKPT}"
  exit 1
fi

cp "${ORIG_EVAL_SOCKET}" "${TMP_EVAL_SOCKET}"
perl -0pi -e "s/conda deactivate/conda activate ${CLIENT_ENV}/" "${TMP_EVAL_SOCKET}"
chmod +x "${TMP_EVAL_SOCKET}"
trap 'rm -f "${TMP_EVAL_SOCKET}"' EXIT

mapfile -t TASKS < <(
  find "${ROBOTWIN_ROOT}/precollected_dataset/dataset" \
    -mindepth 2 -maxdepth 2 -type d -name 'aloha-agilex_clean_50' \
    -printf '%h\n' | xargs -n1 basename | sort
)

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "[error] no aloha-agilex_clean_50 tasks found under ${ROBOTWIN_ROOT}/precollected_dataset/dataset"
  exit 1
fi

printf "task\tsuccess\ttotal\trate_percent\tstatus\tresult_file\tlog_file\n" > "${REPORT_TSV}"

{
  echo "# RoboTwin Eval Summary"
  echo
  echo "- checkpoint: \`${CKPT}\`"
  echo "- task_config: \`${TASK_CONFIG}\`"
  echo "- tasks: ${#TASKS[@]}"
  echo "- trials/task: ${TEST_NUM}"
  echo "- seed: ${SEED}"
  echo "- gpu: ${GPU_ID}"
  echo "- instruction_type: \`${INSTRUCTION_TYPE}\`"
  echo "- diffusion_steps: ${DIFFUSION_STEPS}"
  echo "- exec_horizon: ${EXEC_HORIZON}"
  echo
  echo "| task | success | total | rate | status |"
  echo "|---|---:|---:|---:|---|"
} > "${REPORT_MD}"

cd "${ROBOTWIN_ROOT}/policy/VLANeXt" || exit 1

sum_success=0
sum_total=0
finished_tasks=0

for task in "${TASKS[@]}"; do
  log_file="${LOG_ROOT}/${task}.log"
  echo "[eval] ${task} (${TEST_NUM} episodes) -> ${log_file}"

  bash "${TMP_EVAL_SOCKET}" \
    "${task}" \
    "${TASK_CONFIG}" \
    "${CKPT_TAG}" \
    "${CKPT}" \
    "${SEED}" \
    "${GPU_ID}" \
    "${TEST_NUM}" \
    "${INSTRUCTION_TYPE}" \
    "${DIFFUSION_STEPS}" \
    "${EXEC_HORIZON}" \
    "${SERVER_ENV}" 2>&1 | tee "${log_file}"

  exit_code=${PIPESTATUS[0]}
  clean_log=$(perl -pe 's/\e\[[0-9;]*m//g' "${log_file}")
  result_file=$(printf "%s\n" "${clean_log}" | grep -oP 'Data has been saved to \K.*' | tail -1 || true)
  final_rate_line=$(printf "%s\n" "${clean_log}" | grep 'Success rate:' | tail -1 || true)

  success=""
  total=""
  rate_percent=""
  status="failed"

  if [[ -n "${final_rate_line}" ]]; then
    success=$(printf "%s\n" "${final_rate_line}" | grep -oP 'Success rate: \K[0-9]+(?=/)' || true)
    total=$(printf "%s\n" "${final_rate_line}" | grep -oP 'Success rate: [0-9]+/\K[0-9]+' || true)
    rate_percent=$(printf "%s\n" "${final_rate_line}" | grep -oP '=> \K[0-9.]+(?=%)' || true)
  fi

  if [[ "${exit_code}" -eq 0 && -n "${success}" && -n "${total}" ]]; then
    status="ok"
    sum_success=$((sum_success + success))
    sum_total=$((sum_total + total))
    finished_tasks=$((finished_tasks + 1))
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task}" "${success}" "${total}" "${rate_percent}" "${status}" "${result_file}" "${log_file}" >> "${REPORT_TSV}"

  md_rate="${rate_percent}"
  [[ -n "${md_rate}" ]] && md_rate="${md_rate}%"
  echo "| ${task} | ${success:-} | ${total:-} | ${md_rate:-} | ${status} |" >> "${REPORT_MD}"
done

overall_rate="NA"
if [[ "${sum_total}" -gt 0 ]]; then
  overall_rate=$(awk -v s="${sum_success}" -v t="${sum_total}" 'BEGIN { printf "%.2f%%", 100*s/t }')
fi

{
  echo
  echo "## Overall"
  echo
  echo "- finished_tasks: ${finished_tasks}/${#TASKS[@]}"
  echo "- success: ${sum_success}/${sum_total}"
  echo "- success_rate: ${overall_rate}"
  echo "- tsv: \`${REPORT_TSV}\`"
} >> "${REPORT_MD}"

echo "[done] finished_tasks=${finished_tasks}/${#TASKS[@]} success=${sum_success}/${sum_total} rate=${overall_rate}"
echo "[done] report tsv: ${REPORT_TSV}"
echo "[done] report md: ${REPORT_MD}"
