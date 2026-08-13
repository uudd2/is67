#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dm/QWENLA/VLANeXt_migration/VLANeXt"
GOAL_CHECKPOINT="$PROJECT_ROOT/checkpoints/VLANeXt_vita_hiermq54/actioneffect_v3_auxonly_mainview_q4_cross0_global24_nodct_bs16_libero_goal/checkpoint_160000.pt"
TRAIN_CONFIG="config/calvin_abcd_d_train_vita_hiermq54_actioneffect_v3_auxonly_freezevistext_train12_bs16_config.yaml"
TRAIN_LOG="$PROJECT_ROOT/train_calvin_abcd_d_freezevistext_train12_bs16_gpu0.log"
TRAIN_PID_FILE="$PROJECT_ROOT/train_calvin_abcd_d_freezevistext_train12_bs16_gpu0.pid"

echo "[$(date '+%F %T')] Waiting for goal checkpoint: $GOAL_CHECKPOINT"
while [[ ! -f "$GOAL_CHECKPOINT" ]]; do
    sleep 300
done

echo "[$(date '+%F %T')] Goal checkpoint found; waiting 180 seconds for GPU0 cleanup."
sleep 180

cd "$PROJECT_ROOT"
source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

echo "$$" > "$TRAIN_PID_FILE"
echo "[$(date '+%F %T')] Starting CALVIN ABCD-D training on GPU0."
exec env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_VISIBLE_DEVICES=0 \
    python -m scripts.train --config "$TRAIN_CONFIG" \
    >> "$TRAIN_LOG" 2>&1
