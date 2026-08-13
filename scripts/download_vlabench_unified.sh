#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"$HOME/datasets"}
DATASET_NAME="lerobot/vlabench_unified"
LOCAL_DIR="$DATA_ROOT/vlabench_unified"
PYTHON_BIN=${PYTHON_BIN:-"$HOME/miniconda3/bin/python"}

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

mkdir -p "$DATA_ROOT"

"$PYTHON_BIN" -m pip install -U "huggingface_hub[cli]"
"$PYTHON_BIN" -m pip install -U hf_transfer || true
export HF_HUB_ENABLE_HF_TRANSFER=1

if command -v hf >/dev/null 2>&1; then
  hf download \
    "$DATASET_NAME" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR"
else
  huggingface-cli download \
    "$DATASET_NAME" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download
fi

echo "Download finished."
echo "Dataset path: $LOCAL_DIR"
