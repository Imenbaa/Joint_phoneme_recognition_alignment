#!/bin/bash
# Step 2: CTC phone recogniser with 10 ms frames (last conv stride 1), fine-tuned
# from the 20 ms model of step 1. 10 epochs, lr 3e-5, effective batch 64
# (16 x 4 on one GPU); the best epoch on validation PER is saved to
# $WORK_DIR/ctc_10ms.
#
#   WORK_DIR=/path/to/work [CUDA_VISIBLE_DEVICES=0] bash recipes/2_ctc_10ms.sh [extra train_ctc.py flags]
set -euo pipefail
: "${WORK_DIR:?set WORK_DIR (see 0_prepare_data.sh)}"
cd "$(dirname "$0")/.."

python train_ctc.py \
  --init_model "$WORK_DIR/ctc_20ms" \
  --last_conv_stride 1 \
  --preprocessed_path "$WORK_DIR/data/preprocessed_16k" \
  --vocab_path "$WORK_DIR/data/vocab.json" \
  --output_dir "$WORK_DIR/ctc_10ms" \
  --epochs 10 --lr 3e-5 --batch_size 16 --grad_acc 4 \
  "$@"
