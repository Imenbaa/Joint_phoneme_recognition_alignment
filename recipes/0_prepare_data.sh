#!/bin/bash
# Step 0: preprocess the audiofolder dataset (16 kHz Arrow dataset) and build the
# phone vocabulary. Shared by every later step.
#
#   DATA_DIR=/path/to/audiofolder WORK_DIR=/path/to/work bash recipes/0_prepare_data.sh [extra train_ctc.py flags]
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR to the audiofolder dataset}"
: "${WORK_DIR:?set WORK_DIR to the folder that receives data and models}"
cd "$(dirname "$0")/.."

python train_ctc.py --preprocess_only \
  --data_dir "$DATA_DIR" \
  --preprocessed_path "$WORK_DIR/data/preprocessed_16k" \
  --vocab_path "$WORK_DIR/data/vocab.json" \
  "$@"
