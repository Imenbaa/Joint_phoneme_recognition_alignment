# A joint phoneme recognition and alignment model

Joint **phone recognition and phone alignment** for French speech, with one wav2vec2 model.
A CTC head transcribes the speech into phones. A cross-attention aligner, trained in the same
network, places each phone in time, with optional silences. The same model gives the phone
sequence and its segmentation, at 10 ms resolution.

This repository has the code to train the models from scratch and to evaluate them as aligners.

## Contents

- [Models](#models)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Repository layout](#repository-layout)

## Models

Training runs in three stages. Each stage starts from the previous one.

| Stage | Model | Starts from | Frame | Recipe |
|---|---|---|---|---|
| 1 | **CTC 20 ms**: wav2vec2 phone recogniser | [LeBenchmark/wav2vec2-FR-7K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large) | 20 ms | `recipes/1_ctc_20ms.sh` |
| 2 | **CTC 10 ms**: same, last conv layer with stride 1 | CTC 20 ms | 10 ms | `recipes/2_ctc_10ms.sh` |
| 3 | **Joint aligner 10 ms + latent silence** | CTC 10 ms | 10 ms | `recipes/3_aligner_10ms_sil.sh` |

Stage 3 has two variants. They differ in how strongly the alignment losses train the encoder:
`ALPHA=0.1` (default) and `ALPHA=1.0`.

### CTC phone recogniser (stages 1 and 2)

`Wav2Vec2ForCTC` fine-tuned with CTC on phone sequences. The conv front-end is frozen, and the
epoch with the best validation phone error rate (PER) is kept.

Stage 2 changes the stride of the last conv layer from 2 to 1. Frames are then 10 ms instead
of 20 ms. Every pretrained weight still loads, and the model is fine-tuned again at the new
rate.

### Joint recognition + alignment model (stage 3)

`reco_align.model.Wav2Vec2ForCTCPoolContrastive`, built on the CTC 10 ms model:

```
waveform ─► wav2vec2 encoder ─┬─► CTC head ──────────────────────────► phones        (L_CTC)
                              │
                              ├─► ConvBank (kernels 1,5,9) ─► queries ┐
                              │                                       ├─► attention energy  (L_FS)
                              └─► phone keys ─────────────────────────┘        │
                                  (CTC-posterior × Gaussian window              ▼
                                   pooling of encoder frames)          masked reconstruction (L_m)
```

- **Phone keys.** Each phone of the target sequence gets one vector. It is a weighted average of
  encoder frames: the weights are that phone's CTC posterior times a Gaussian window (σ = 10
  frames) centred on the phone. No separate text encoder is needed.
- **L_FS: ForwardSum alignment loss.** It is computed on the frame × phone attention energy, with
  a beta-binomial prior that is annealed away between 15 % and 50 % of training.
- **L_m: masked reconstruction loss.**
  - The encoder frames are masked (p ~ U[0.1, 0.5]).
  - They are reconstructed from the masked frames and the alignment-weighted phone keys.
  - The reconstruction is scored with InfoNCE against codes from an EMA vector quantizer of the
    encoder output (nearest-code assignment, 320 codes).
  - A good reconstruction needs the right phone at the right frame, so L_m also trains the
    alignment.
- **L_energy.** A small head predicts each frame's waveform log-energy from the alignment content
  alone. It is added to L_m. A loud frame assigned to silence is then predicted quiet, and the
  error corrects the alignment.
- **Latent optional silence (SIL)** starts at step 2000, in the style of Kaldi/MFA.
  - SIL slots are inserted at word boundaries: `[SIL] p1 p2 [SIL] p3 … [SIL]`. They are optional
    and all share one learned key.
  - Entering a slot costs log p = −2, and a used SIL lasts at least 4 frames (40 ms).
  - The ForwardSum dynamic programming sums over paths with and without each SIL
    (`reco_align/sil_fs.py`).
  - L_m takes its attention from the forward-backward occupancy.
  - At inference, a Viterbi pass over the same topology gives phone boundaries and pauses.
- **Gradient routing.**
  - L_CTC trains the whole encoder.
  - L_FS and L_m reach only the top 2 transformer layers, through the queries.
  - Their gradient is scaled by `ALPHA` (`--encoder_alpha`, `--lm_encoder_alpha`), ramped in
    between steps 2000 and 5000.
  - Phone keys and reconstruction targets are always detached.

At inference the model aligns its own CTC decode, so it needs no transcription.

## Installation

Python ≥ 3.10 and a CUDA GPU for training.

```bash
git clone https://github.com/Imenbaa/Joint_phoneme_recognition_alignment.git
cd Joint_phoneme_recognition_alignment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions used to train the models
pip install -e .                    # the reco_align package
```

The scripts at the repository root (`train_ctc.py`, `train_aligner.py`, `evaluate.py`) also run
from the repository root without `pip install -e .`.

## Data

### Training corpus

The models were trained on the French broadcast-news corpora **ESTER 1, ESTER 2 and EPAC**
(distributed by ELRA), with phone transcriptions. Utterances containing an unknown word were
removed. Any French (or other) corpus with phone-level transcriptions can be used in the same
format.

### Format

The code reads a Hugging Face [`audiofolder`](https://huggingface.co/docs/datasets/audio_dataset#audiofolder):

```
DATA_DIR/
  train/        metadata.csv + audio files
  dev/          (or validation/)
  test/
```

`metadata.csv` has one row per utterance:

| column | required | content |
|---|---|---|
| `file_name` | yes | audio file, relative to the split folder |
| `phonemes_single_v2` | yes | the phone sequence, one symbol per phone, separated by spaces: `p a t ʁ i s j a` |
| `phonemes_single` | stage 3 | the same phones written as one string per word, words separated by spaces: `patʁisja maʁt1`. It places the SIL slots at word boundaries, so each phone symbol must be a single character, identical in both columns |
| `phonemes` | no | if present, rows containing `\|ø\|n\|k\|` (unknown word) are dropped |

Any other column (for example `transcription`) is ignored. Producing the phone transcriptions
(lexicon / grapheme-to-phoneme conversion) is not part of this repository. The phone vocabulary
is built from the data in stage 0.

## Training

Every recipe reads two environment variables. Settings can be overridden with extra flags,
which are passed through to the Python script.

- `WORK_DIR`: receives the preprocessed data and all models.
- `DATA_DIR`: the audiofolder above.

```bash
export DATA_DIR=/path/to/audiofolder WORK_DIR=/path/to/work

bash recipes/0_prepare_data.sh            # -> $WORK_DIR/data/{preprocessed_16k, vocab.json}
NGPU=2 bash recipes/1_ctc_20ms.sh         # -> $WORK_DIR/ctc_20ms
bash recipes/2_ctc_10ms.sh                # -> $WORK_DIR/ctc_10ms
bash recipes/3_aligner_10ms_sil.sh        # -> $WORK_DIR/aligner_10ms_sil_alpha0.1
ALPHA=1.0 bash recipes/3_aligner_10ms_sil.sh   # -> $WORK_DIR/aligner_10ms_sil_alpha1.0
```

Choose the GPU with `CUDA_VISIBLE_DEVICES`. Stage 1 uses `torchrun` with `NGPU` processes. The
gradient accumulation adapts so that the effective batch stays 128. TensorBoard logs go to
`<output>/runs`.

### Stage 0: data preparation (`train_ctc.py --preprocess_only`)

The audio is resampled to 16 kHz and normalised, and the phone strings are converted to ids.
The result is an Arrow dataset, plus a vocabulary file `vocab.json` (`[PAD]` = CTC blank,
`[UNK]`, then the phones). All later stages read this dataset.

### Stages 1–2: CTC (`train_ctc.py`)

| | stage 1 | stage 2 |
|---|---|---|
| `--init_model` | `LeBenchmark/wav2vec2-FR-7K-large` | `$WORK_DIR/ctc_20ms` |
| `--last_conv_stride` | 2 (20 ms) | 1 (10 ms) |
| `--epochs` | 30 | 10 |
| `--lr` | 3e-5 (linear decay, 500 warm-up steps) | 3e-5 |
| batch | 32 × 2 GPUs × grad_acc 2 = 128 | 16 × grad_acc 4 = 64 |
| precision | bf16 on Ampere GPUs, else fp16 | same |

Other settings, the same for both stages:
- frozen conv front-end (`--train_cnn` unfreezes it);
- weight decay 0.005, gradient clipping 2.0;
- gradient checkpointing;
- the epoch with the best validation PER is kept.

Run `python train_ctc.py --help` for every option.

### Stage 3: joint aligner (`train_aligner.py`)

Main settings of `recipes/3_aligner_10ms_sil.sh`:

| group | flags |
|---|---|
| length, batch | `--max_steps 43750`, batch 4 × grad_acc 8 = 32, `--fp16`, `--seed 42` |
| learning rates | encoder + CTC head `--backbone_lr 1e-5`, aligner `--alignment_lr 1e-4`, weight decay 1e-4, no warm-up, clipping 1.0 |
| losses | `--ctc_weight 1 --align_weight 1 --contrastive_weight 1`; L_FS ramps in over `--align_warmup_steps 1000` |
| aligner | `--pool_mode window --window_sigma_frames 10 --convbank_kernels 1,5,9`; attention dimension = encoder size / 2 |
| prior | `--prior_type beta_binomial --prior_scaling_factor 0.5 --prior_schedule anneal` from 0.15 to 0.5 of training |
| L_m | `--quantizer_assign nearest --lm_mask_same_code --no_position_features`, 50 negatives, temperature 0.1 |
| L_energy | `--lm_energy_weight 1.0` |
| routing | `--encoder_alpha ALPHA --lm_encoder_alpha ALPHA --encoder_top_layers 2`, ramp `--encoder_alpha_start_steps 2000 --encoder_alpha_ramp_steps 3000` |
| silence | `--sil --sil_positions word --sil_logp -2.0 --sil_min_frames 4 --sil_init lowenergy --sil_start_steps 2000 --lm_att_source fs_posterior` |
| bookkeeping | checkpoint every 1000 steps (`--optimizer_last_only` keeps `optimizer.pt` in the newest one only); validation PER every 1000 steps |

Run `python train_aligner.py --help` for every option and its meaning.

Notes:
- **Held-out alignment metrics.** `EVAL_MANIFEST=/path/to/manifest.json` (see
  [Evaluation](#evaluation)) also scores each checkpoint as an aligner on a held-out manifest.
  The scores are logged to TensorBoard under `heldout/` and saved in `<output>/heldout/`. They
  are not used to select checkpoints.
- **Resuming.** Training resumes from the newest checkpoint in the output folder, so a run that
  is stopped can be restarted with the same command.
- **Expected warning.** The trainer warns that 7 trainable tensors are in no optimizer group.
  These are the wav2vec2 pretraining heads (`quantizer`, `project_hid`, `project_q`), which this
  model does not use.

## Evaluation

`evaluate.py` scores checkpoints as phone aligners on a **manifest** of long recordings with
reference phone intervals:

```json
{"files": {
  "rec01": {
    "wav": "/path/to/rec01.wav",
    "chunks": [{"start": 0.0, "end": 14.2}, {"start": 15.1, "end": 29.8}],
    "ref_intervals": [{"phoneme": "b", "start": 0.31, "end": 0.38}, ...],
    "style": "spontaneous",
    "pause_intervals": [[3.10, 3.42], ...],
    "excluded_intervals": [[20.0, 21.5], ...]
  }
}}
```

| field | required | meaning |
|---|---|---|
| `wav` | yes | the recording; any sampling rate, mono or stereo |
| `chunks` | yes | the segments the model sees, in seconds (for example from a voice activity detector) |
| `ref_intervals` | yes | the reference phones and their intervals |
| `style` | no | groups the results by speaking style |
| `pause_intervals` | no | reference pauses; enables the silence scores |
| `excluded_intervals` | no | regions ignored by the silence scores |

```bash
python evaluate.py --manifest manifest.json --out_dir eval/ \
    $WORK_DIR/aligner_10ms_sil_alpha0.1/checkpoint-43000
python evaluate.py --manifest manifest.json --out_dir eval/ --method ctcfa_contig <checkpoint>
```

- **What is aligned.** The model aligns its own CTC greedy decode, not the reference phones.
  Hypothesis and reference are matched by edit distance.
- **`--method attention`** (default):
  - for a latent-SIL model, Viterbi decoding over phones and optional SIL;
  - otherwise, monotonic DTW on the attention.
  - For SIL models the plain DTW decode of the same pass is also reported, with the suffix
    `_dtw`.
- **`--method ctcfa` / `ctcfa_contig`.** The same decoded phones are force-aligned on the CTC
  head with `torchaudio.functional.forced_align`. Phones either end where their CTC run ends
  (`ctcfa`) or where the next phone starts (`ctcfa_contig`). This is a baseline that does not
  use the attention.
- **Frame timing** is read from the model config (stride and receptive field): frame f of a chunk
  starting at t₀ gives a boundary at t₀ + f·stride + (RF − stride)/2. That is +7.5 ms at 10 ms
  and +2.5 ms at 20 ms. The same rule is used for every method.

**Metrics** (`reco_align/evaluation/metrics.py`):
- boundary precision, recall and F1 within 10, 20 and 50 ms (main score: F1@20);
- median, mean and 90th percentile of the absolute boundary error, and rate of errors > 50 ms;
- signed shift of phone starts and ends;
- PER;
- duration fidelity: slope and correlation of hypothesis versus reference phone durations
  (`_sp`: phones with no pause on either side);
- for SIL models with `pause_intervals`: frame-level SIL precision/recall/F1 and pause-level F1
  at IoU ≥ 0.5.

The script writes these files to `--out_dir`:
- per-file, per-style and global CSVs;
- the alignments (`.pkl`);
- one JSON line per checkpoint in `summary.jsonl`.

## Tests

CPU only. They need no data or pretrained weights:

```bash
python tests/test_sil_fs.py   # latent-SIL DP vs brute force, gradients, batching, == ForwardSumLoss without SIL
python tests/test_model.py    # full model on a tiny random wav2vec2: losses, gradient routing, SIL, L_energy, decoders
```

## Repository layout

```
train_ctc.py              stages 0-2: data preparation, CTC fine-tuning (20 or 10 ms)
train_aligner.py          stage 3: joint recognition + alignment model
evaluate.py               alignment evaluation on a manifest
recipes/                  one script per stage, with the settings of the trained models
reco_align/
  modules.py              ForwardSum loss, ConvBank, cross-attention, prior helpers
  pool_alignment.py       CTC + aligner with pooled phone keys, latent SIL, gradient routing
  model.py                + masked reconstruction (L_m), EMA quantizer, L_energy
  sil_fs.py               latent optional SIL: topology, forward-backward, Viterbi
  evaluation/
    attention_eval.py     manifest loop, DTW decode, frame timing
    sil_eval.py           SIL Viterbi decode and silence scores
    ctcfa_eval.py         CTC forced-alignment baseline
    metrics.py            boundary F1, errors, PER
tests/
```

## Generative AI Use Disclosure

The code in this repository was refactored, organised and refined with the aid of
[Claude Code](https://claude.com/claude-code) (Anthropic).

## License

MIT, see `LICENSE`.
