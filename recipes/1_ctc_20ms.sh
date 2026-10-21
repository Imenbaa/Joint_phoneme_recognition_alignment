#!/bin/bash
# Step 1: CTC phone recogniser with 20 ms frames, fine-tuned from the pretrained
# self-supervised wav2vec2. 30 epochs, lr 3e-5, effective batch 128
# (32 per GPU x GRAD_ACC x NGPU), frozen conv front-end; the best epoch on
# validation PER is saved to $WORK_DIR/ctc_20ms.
#
#   WORK_DIR=/path/to/work [NGPU=2] [INIT_MODEL=...] bash recipes/1_ctc_20ms.sh [extra train_ctc.py flags]
set -euo pipefail
: "${WORK_DIR:?set WORK_DIR (see 0_prepare_data.sh)}"
NGPU=${NGPU:-2}
GRAD_ACC=${GRAD_ACC:-$((4 / NGPU))}
INIT_MODEL=${INIT_MODEL:-LeBenchmark/wav2vec2-FR-7K-large}
cd "$(dirname "$0")/.."

torchrun --standalone --nproc_per_node="$NGPU" train_ctc.py \
  --init_model "$INIT_MODEL" \
  --preprocessed_path "$WORK_DIR/data/preprocessed_16k" \
  --vocab_path "$WORK_DIR/data/vocab.json" \
  --output_dir "$WORK_DIR/ctc_20ms" \
  --epochs 30 --lr 3e-5 --batch_size 32 --grad_acc "$GRAD_ACC" \
  "$@"
