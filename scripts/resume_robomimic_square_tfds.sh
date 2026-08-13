#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-/media/dm/Elements/VLANeXt_migration/data/RoboMimic_TFDS}
DOWNLOAD_DIR="$DATA_DIR/downloads/robomimic_ph"
URL="http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/ph/image.hdf5"
EXPECTED_SIZE=2603429840
EXPECTED_SHA256="8901b2e4546c59d57436e2dcad4ba643c806ae3db0133b4bd1859236a0f4eae0"

PARTIAL_DIR=$(find "$DOWNLOAD_DIR" -maxdepth 1 -type d -name '*squa*.hdf5.tmp.*' | head -n 1)
if [[ -z "$PARTIAL_DIR" ]]; then
  echo "No partial Square download found under $DOWNLOAD_DIR" >&2
  exit 1
fi

PARTIAL_FILE="$PARTIAL_DIR/image.hdf5"
CACHE_FILE="${PARTIAL_DIR%%.tmp.*}"

echo "Resuming Square image dataset: $PARTIAL_FILE"
wget --continue --output-document="$PARTIAL_FILE" "$URL"

ACTUAL_SIZE=$(stat -c '%s' "$PARTIAL_FILE")
if [[ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]]; then
  echo "Unexpected size: $ACTUAL_SIZE (expected $EXPECTED_SIZE)" >&2
  exit 1
fi
echo "$EXPECTED_SHA256  $PARTIAL_FILE" | sha256sum --check -

mv "$PARTIAL_FILE" "$CACHE_FILE"
rmdir "$PARTIAL_DIR"
cat > "${CACHE_FILE}.INFO" <<EOF
{"dataset_names":["robomimic_ph"],"original_fname":"image.hdf5","url_info":{"checksum":"$EXPECTED_SHA256","filename":"image.hdf5","size":$EXPECTED_SIZE},"urls":["$URL"]}
EOF

exec /home/dm/miniconda3/envs/VLANeXt/bin/python \
  /home/dm/QWENLA/VLANeXt_migration/VLANeXt/scripts/download_robomimic_tfds.py \
  --data-dir "$DATA_DIR"
