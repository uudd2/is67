#!/usr/bin/env bash
set -euo pipefail

VITA_ROOT="/home/dm/QWENLA/VLANeXt_migration/VITA"
DATA_ROOT="/media/dm/Elements/VLANeXt_migration/data/AV_ALOHA"
CONVERTED_ROOT="${DATA_ROOT}/converted"
HF_CACHE="${DATA_ROOT}/huggingface_cache"
PYTHON="/home/dm/miniconda3/envs/vita/bin/python"

mkdir -p "${CONVERTED_ROOT}" "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export PYTHONUNBUFFERED=1

cd "${VITA_ROOT}"

"${PYTHON}" - "${CONVERTED_ROOT}" <<'PY'
import sys
from pathlib import Path

from gym_av_aloha.datasets.av_aloha_dataset import (
    create_av_aloha_dataset_from_lerobot,
)

output_root = Path(sys.argv[1])
remove_keys = [
    "observation.images.wrist_cam_left",
    "observation.images.wrist_cam_right",
    "observation.images.worms_eye_cam",
    "observation.images.overhead_cam",
]

datasets = {
    "iantc104/av_aloha_sim_peg_insertion": 100,
    "iantc104/av_aloha_sim_cube_transfer": 200,
    "iantc104/av_aloha_sim_thread_needle": 200,
    "iantc104/av_aloha_sim_pour_test_tube": 100,
    "iantc104/av_aloha_sim_hook_package": 100,
    "iantc104/av_aloha_sim_slot_insertion": 100,
}

for repo_id, num_episodes in datasets.items():
    destination = output_root / repo_id
    print(f"[AV-ALOHA] converting {repo_id} -> {destination}", flush=True)
    create_av_aloha_dataset_from_lerobot(
        episodes={repo_id: list(range(num_episodes))},
        repo_id=repo_id,
        root=destination,
        remove_keys=remove_keys,
        image_size=(240, 320),
    )

print("[AV-ALOHA] all datasets finished", flush=True)
PY
