#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/media/dm/Elements/VLANeXt_migration/data/calvin}"
TRAIN_DIR="$DATA_ROOT/calvin-task-ABC-D-lerobot"
EVAL_ROOT="$DATA_ROOT/CALVIN_eval"
CALVIN_CODE="$EVAL_ROOT/calvin"
EVAL_CONFIG_DIR="$EVAL_ROOT/task_ABC_D/validation/.hydra"

mkdir -p "$TRAIN_DIR" "$EVAL_CONFIG_DIR"

echo "[$(date '+%F %T')] Downloading LeRobot CALVIN ABC-D training data..."
/home/dm/miniconda3/bin/hf download \
  fywang/calvin-task-ABC-D-lerobot \
  --repo-type dataset \
  --local-dir "$TRAIN_DIR" \
  --max-workers 4

echo "[$(date '+%F %T')] Downloading the minimal CALVIN D evaluation config..."
wget -q --show-progress \
  -O "$EVAL_CONFIG_DIR/merged_config.yaml" \
  "https://huggingface.co/wangdong24/calvin_data/resolve/737642c40a31d54c8ed95b95007a10f0a8924158/task_ABCD_D/validation/.hydra/merged_config.yaml?download=true"

echo "[$(date '+%F %T')] Downloading the official CALVIN simulator code and assets..."
if [[ -d "$CALVIN_CODE/.git" ]]; then
  git -C "$CALVIN_CODE" submodule update --init --recursive
else
  git clone --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_CODE"
fi

echo "[$(date '+%F %T')] Download complete."
echo "Training data: $TRAIN_DIR"
echo "Evaluation dataset path: $EVAL_ROOT/task_ABC_D"
echo "CALVIN code: $CALVIN_CODE"
