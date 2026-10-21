#!/bin/bash
# Step 3: joint recognition + alignment model at 10 ms, from the CTC model of step 2:
# L_CTC + L_FS (ForwardSum, beta-binomial prior annealed away) + L_m (masked
# reconstruction, nearest-code EMA quantizer) + L_energy, latent optional SIL
# from step 2000. 43750 steps, effective batch 32 (4 x 8 on one GPU), seed 42.
#
# ALPHA sets how strongly L_FS / L_m train the top 2 encoder layers
# (--encoder_alpha = --lm_encoder_alpha, ramped in over steps 2000-5000):
#   ALPHA=0.1 (default) or ALPHA=1.0.
#
#   DATA_DIR=/path/to/audiofolder WORK_DIR=/path/to/work [ALPHA=0.1] \
#   [EVAL_MANIFEST=/path/to/manifest.json] [CUDA_VISIBLE_DEVICES=0] \
#   bash recipes/3_aligner_10ms_sil.sh [extra train_aligner.py flags]
#
# DATA_DIR is read for the word boundaries (phonemes_single in
# train/metadata.csv) that place the SIL slots. EVAL_MANIFEST is optional:
# alignment metrics on it are logged to TensorBoard at every evaluation.
# Training resumes from the newest checkpoint in the output folder.
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR to the audiofolder dataset}"
: "${WORK_DIR:?set WORK_DIR (see 0_prepare_data.sh)}"
ALPHA=${ALPHA:-0.1}
OUT=${OUT:-$WORK_DIR/aligner_10ms_sil_alpha$ALPHA}
EVAL=()
if [ -n "${EVAL_MANIFEST:-}" ]; then EVAL=(--eval_manifest "$EVAL_MANIFEST"); fi
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

python train_aligner.py \
  --audio_dir "$DATA_DIR" \
  --preprocessed_path "$WORK_DIR/data/preprocessed_16k" --sampling_rate 16000 \
  --vocab_path "$WORK_DIR/data/vocab.json" \
  --pretrained_model "$WORK_DIR/ctc_10ms" \
  --out_dir "$OUT" \
  ${EVAL[@]+"${EVAL[@]}"} \
  --max_steps 43750 --seed 42 \
  --no_position_features \
  --ctc_weight 1.0 --align_weight 1.0 --contrastive_weight 1.0 \
  --align_warmup_steps 1000 \
  --encoder_alpha "$ALPHA" --lm_encoder_alpha "$ALPHA" --encoder_top_layers 2 \
  --encoder_alpha_start_steps 2000 --encoder_alpha_ramp_steps 3000 \
  --prior_schedule anneal --prior_anneal_start_frac 0.15 --prior_anneal_end_frac 0.5 \
  --prior_type beta_binomial --pool_mode window --window_sigma_frames 10 \
  --convbank_kernels 1,5,9 --prior_scaling_factor 0.5 \
  --backbone_lr 1e-5 --alignment_lr 1e-4 --weight_decay 1e-4 --warmup_steps 0 \
  --max_grad_norm 1.0 --layerdrop 0.0 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 --fp16 \
  --logging_steps 100 --viz_every_n_steps 500 \
  --optimizer_last_only \
  --sil --sil_positions word --sil_eval_positions all \
  --sil_logp -2.0 --sil_min_frames 4 --sil_init lowenergy --sil_start_steps 2000 \
  --lm_att_source fs_posterior --lm_mask_same_code \
  --quantizer_assign nearest --lm_energy_weight 1.0 \
  "$@"
