#!/usr/bin/env bash
set -Eeuo pipefail

disk_root="/media/ding-101/91d6d462-81a7-41ee-86fb-74cb0f959c05"
hf_bin="${HF_BIN:-/home/ding-101/miniconda3/envs/VLANeXt/bin/hf}"

download_root="${disk_root}/VLANeXt_downloads"
hf_home="${download_root}/hf_home"
log_dir="${download_root}/logs"

libero_plus_assets_dir="${disk_root}/LIBERO_plus_assets"
libero_plus_data_dir="${disk_root}/LIBERO_plus_datasets/libero_plus_rlds"

mkdir -p "${hf_home}" "${log_dir}"
mkdir -p "${libero_plus_assets_dir}" "${libero_plus_data_dir}"

export HF_HOME="${hf_home}"
export HF_HUB_CACHE="${hf_home}/hub"

log_file="${log_dir}/night_downloads.log"

run_download() {
  local label="$1"
  shift
  echo "[$(date '+%F %T %Z %z')] START ${label}"
  "$@"
  echo "[$(date '+%F %T %Z %z')] DONE  ${label}"
}

{
  echo "[$(date '+%F %T %Z %z')] Night download job started"
  echo "HF binary: ${hf_bin}"
  echo "HF_HOME: ${HF_HOME}"
  echo "Disk root: ${disk_root}"

  run_download "LIBERO-plus assets.zip" \
    "${hf_bin}" download Sylvest/LIBERO-plus assets.zip \
    --repo-type dataset \
    --local-dir "${libero_plus_assets_dir}"

  run_download "LIBERO-plus RLDS dataset" \
    "${hf_bin}" download Sylvest/libero_plus_rlds \
    --repo-type dataset \
    --local-dir "${libero_plus_data_dir}"

  echo "[$(date '+%F %T %Z %z')] Night download job finished"
  echo "Downloaded files summary:"
  find "${libero_plus_assets_dir}" "${libero_plus_data_dir}" -maxdepth 2 -type f -printf '%s %p\n' | sort -nr | head -50
} 2>&1 | tee -a "${log_file}"
