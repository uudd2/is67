#!/usr/bin/env bash
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/media/dm/Elements/VLANeXt_migration/data}"
STAGING_DIR="${DATA_ROOT}/bridge_dataset"
FINAL_DIR="${DATA_ROOT}/bridge_orig"
SOURCE_URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/"
MAX_WINDOW_SECONDS="${WINDOW_SECONDS:-21600}"

now_epoch=$(date +%s)
end_epoch=$(date -d "$(date +%F) 06:00:00" +%s)
if (( now_epoch >= end_epoch )); then
  echo "[$(date '+%F %T')] Outside the 00:00-06:00 download window; skipping"
  exit 0
fi
remaining_seconds=$((end_epoch - now_epoch))
if (( remaining_seconds < MAX_WINDOW_SECONDS )); then
  WINDOW_SECONDS=${remaining_seconds}
else
  WINDOW_SECONDS=${MAX_WINDOW_SECONDS}
fi

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

if [[ -f "${FINAL_DIR}/.download_complete" ]]; then
  echo "[$(date '+%F %T')] BridgeData V2 is already complete: ${FINAL_DIR}"
  exit 0
fi

if [[ -e "${FINAL_DIR}" && ! -e "${STAGING_DIR}" ]]; then
  echo "[$(date '+%F %T')] Refusing to overwrite existing incomplete final directory: ${FINAL_DIR}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Starting/resuming BridgeData V2 download"
echo "staging=${STAGING_DIR}"
echo "window_seconds=${WINDOW_SECONDS}"

timeout --signal=INT --kill-after=60s "${WINDOW_SECONDS}" \
  wget --continue --recursive --no-host-directories --cut-dirs=4 \
    --no-parent --reject='index.html*' "${SOURCE_URL}"
status=$?

if [[ ${status} -eq 0 ]]; then
  if [[ -e "${FINAL_DIR}" ]]; then
    echo "[$(date '+%F %T')] Final directory already exists; leaving completed data in ${STAGING_DIR}" >&2
    exit 1
  fi
  mv "${STAGING_DIR}" "${FINAL_DIR}"
  touch "${FINAL_DIR}/.download_complete"
  echo "[$(date '+%F %T')] Download complete: ${FINAL_DIR}"
  exit 0
fi

if [[ ${status} -eq 124 || ${status} -eq 130 ]]; then
  echo "[$(date '+%F %T')] Download window closed; files retained for the next run"
  exit 0
fi

echo "[$(date '+%F %T')] wget stopped with status ${status}; next scheduled run will retry" >&2
exit "${status}"
