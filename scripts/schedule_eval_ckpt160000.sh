#!/usr/bin/env bash
set -euo pipefail

cd /home/dm/QWENLA/VLANeXt_migration/VLANeXt

TARGET="2026-07-07 00:00:00"
target_epoch=$(date -d "$TARGET" +%s)
now_epoch=$(date +%s)
sleep_seconds=$((target_epoch - now_epoch))

echo "[scheduler] created_at=$(date '+%F %T %Z') target=$TARGET sleep_seconds=$sleep_seconds"
if [ "$sleep_seconds" -gt 0 ]; then
  sleep "$sleep_seconds"
fi

ckpt="/home/dm/QWENLA/VLANeXt_migration/VLANeXt/checkpoints/VLANeXt_vita_hiermq54/vlm12_0vis8_0text4_1_4_8_12_dim1024_flow8_ae6_dct_div_fixedgate01_tokengate08_finetune_vlm01_bs16_libero_long/checkpoint_160000.pt"
for _ in $(seq 1 120); do
  if [ -f "$ckpt" ]; then
    break
  fi
  echo "[scheduler] $(date '+%F %T') waiting for checkpoint: $ckpt"
  sleep 60
done

if [ ! -f "$ckpt" ]; then
  echo "[scheduler] missing checkpoint after wait: $ckpt"
  exit 2
fi

source /home/dm/miniconda3/etc/profile.d/conda.sh
conda activate VLANeXt

PYTHONPATH=/home/dm/QWENLA/VLANeXt_migration/VLANeXt/third_party/LIBERO:$PYTHONPATH \
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib \
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python -m scripts.libero_bench_eval \
  --config config/libero_long_hiermq54_vlm12_split_bs16_ckpt160000_eval_config.yaml
