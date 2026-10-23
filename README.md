# A joint phoneme recognition and alignment model

Joint **phone recognition and phone alignment** for French speech, with one wav2vec2 model.
A CTC head transcribes the speech into phones. A cross-attention aligner, trained in the same
network, places each phone in time, with optional silences. The same model gives the phone
sequence and its segmentation, at 10 ms resolution.

This repository has the code to train the models from scratch, to evaluate them as aligners, and
to transcribe recordings into Praat TextGrids (phones, boundaries, pauses and the model's top-5
candidates for each phone), with no transcript needed.

**Phoneticians: start with the [step-by-step guide](GUIDE.md)**: install with two commands, then
transcribe a corpus with one command (with an optional HTML report of the plots), or from Praat.

## Contents

- [Models](#models)
- [Installation](#installation)
- [Transcribing recordings](#transcribing-recordings)
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

`jointrecalign.model.Wav2Vec2ForCTCPoolContrastive`, built on the CTC 10 ms model:

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
    (`jointrecalign/sil_fs.py`).
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

Python ≥ 3.10. A CUDA GPU for training; transcription also runs on CPU.

**To transcribe recordings only**, see the [step-by-step guide](GUIDE.md) (no Python
knowledge needed), or, in an existing Python environment:

```bash
pip install git+https://github.com/Imenbaa/Joint_phoneme_recognition_alignment.git
```

Both install the `JointRecAlign` command.

**To train or evaluate**, clone the repository:

```bash
git clone https://github.com/Imenbaa/Joint_phoneme_recognition_alignment.git
cd Joint_phoneme_recognition_alignment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions used to train the models
pip install -e .                    # the jointrecalign package
```

The scripts at the repository root (`train_ctc.py`, `train_aligner.py`, `evaluate.py`,
`transcribe.py`) also run from the repository root without `pip install -e .`.
`pip install -e .` also installs the `JointRecAlign` command.

## Transcribing recordings

`JointRecAlign` (or `python transcribe.py` from the repository root) gives, for each
recording, the phones the model hears, their boundaries, the pauses, and the model's five best
candidates for each phone. It runs on GPU or CPU.

```bash
JointRecAlign validate recordings_folder/                 # check the audio files (no model)
JointRecAlign transcribe recordings_folder/ [-o out/]     # check, then transcribe
JointRecAlign transcribe recordings_folder/ --report      # + one HTML page per recording, index.html
JointRecAlign transcribe recordings_folder/ --reference textgrids/   # + comparison with reference TextGrids
JointRecAlign install-praat-plugin                        # a "JointRecAlign: transcribe..." button for Sounds in Praat
```

`transcribe` checks every file first (readable, at least 0.5 s, not silent; warnings for a
sampling rate under 16 kHz, several channels, clipping, very low level, two recordings that
would write the same result files), skips the files that cannot be transcribed, goes on when
one fails, and ends with a summary, also written to `JointRecAlign.log`. The exit code is 1 when
a recording was not transcribed.

**The model** ([`aligner_10ms_sil_alpha0.1.zip`](https://github.com/Imenbaa/Joint_phoneme_recognition_alignment/releases/download/v0.1.0/aligner_10ms_sil_alpha0.1.zip),
1.2 GB, [release v0.1.0](https://github.com/Imenbaa/Joint_phoneme_recognition_alignment/releases/tag/v0.1.0),
licence CC BY-NC 4.0) is downloaded and checked the first time (or with `JointRecAlign download`)
into a cache folder: `~/.cache/JointRecAlign` (Linux), `~/Library/Caches/JointRecAlign` (macOS),
`%LOCALAPPDATA%\JointRecAlign` (Windows); `JOINTRECALIGN_HOME` changes it. Any stage-3 checkpoint
you train yourself (joint aligner with latent silence, see [Training](#training)) works with
`--model`, e.g. `--model $WORK_DIR/aligner_10ms_sil_alpha0.1/checkpoint-43000`.

Audio: WAV, FLAC, MP3, OGG or AIFF, any sampling rate, mono or stereo (resampled to 16 kHz and
downmixed). A folder means every audio file in it. A recording that cannot be read is reported
and the others are still transcribed.

For each recording, in a folder `JointRecAlign/` next to the recording (or in `-o`), and
`JointRecAlign.log`:

| file | content |
|---|---|
| `<name>.TextGrid` | Praat interval tiers, all covering the whole recording |
| `<name>.csv` | one row per phone: `start`, `end`, `phone`, then `top1`, `p1` ... `top5`, `p5` |
| `<name>.html` | with `--report`: waveform, phones and pauses, the aligner's attention with the alignment path, the CTC posteriors of every phone, the top 5 of each frame, on one time axis that scrolls and zooms; self-contained, no server (`index.html` lists the recordings) |

| tier | label of each interval |
|---|---|
| `phones` | the recognised phone (empty between phones) |
| `confidence` | probability of that phone |
| `top2` ... `top5` | the next candidates, as `symbol probability` (e.g. `p 0.050`) |
| `pauses` | `SIL` on every pause the model detected |

| option | meaning |
|---|---|
| `-o folder` | where the results go (default: `JointRecAlign/` next to each recording) |
| `--report` | also the HTML pages (about 0.6 MB per minute of audio) |
| `--copy_audio` | with `--report`: copy each recording into `<results>/audio/`, so the pages still play it after the results folder is moved or copied to another computer |
| `--model folder` | another model (default: the released one, downloaded the first time) |
| `--symbols ipa` (default) / `model` | `ipa` writes the nasal vowels and œ in IPA (`ɑ̃ ɛ̃ ɔ̃ œ`); `model` keeps the model's one-character codes (`@ 1 § 9`) |
| `--phone_map map.json` | your own labels, applied last, e.g. `{"ə": "ø"}` for a corpus that writes the schwa `ø` |
| `--max_chunk 10` | longest piece of audio the model sees at once, in seconds |
| `--vocab_path` | phone vocabulary (default: `vocab.json` in the checkpoint) |
| `--device cuda` / `cpu` | default: `cuda` when available |

**Comparison with a reference** (`--reference FOLDER`, `jointrecalign/reference.py`): for
`<name>.wav`, the phone tier of `FOLDER/<name>.TextGrid` (`--ref_tier`, default the tier named
`phones`/`phonemes`/..., else the first interval tier) is compared with the transcription with the
metrics of the evaluation (`evaluate.py`):

| figure | meaning |
|---|---|
| PER | substitutions + deletions + insertions over the reference phones |
| F1 20 ms / 50 ms | boundary detection: every phone onset and offset; each reference boundary is matched one-to-one to the nearest recognised boundary within 20 / 50 ms |
| median error | median \|error\| of the starts and ends of the phones recognised with the reference label |
| pause P / R / F1 | reference pauses matched one-to-one by a recognised pause (time not covered by a phone) with intersection over union ≥ 0.5 |

Empty intervals and `--pause_labels` (default `SIL sil sp spn _ # <p:>` ...) are reference pauses,
`--skip_labels` (e.g. `noise,overlap`) are left out, and `--ref_map map.json` rewrites the other
labels into the model's phones (e.g. `{"ø": "ə"}` for a corpus that writes the schwa `ø`). The
summary lists the reference labels that are not phones of the model; they count as errors. Results:
`comparison.csv` (one row per recording and a `TOTAL` row; corpus figures pool the counts of all
recordings), the scores in the summary and, with `--report`, in the pages: reference lane under the
recognised phones (amber: substituted, red: inserted / deleted), reference boundaries and pauses
in red on the attention, reference phones as red boxes on the posteriors and the top 5, and a
"show reference" switch. A reference TextGrid is never overwritten (`-o` into the reference folder
is refused when the names collide).

How it works (`jointrecalign/transcribe.py`):
1. The recording is cut into pieces of at most 10 s, back to back, so no audio is dropped.
   [Silero VAD](https://github.com/snakers4/silero-vad) finds the pauses; each cut goes in the
   middle of the latest pause within 10 s (or, with no pause, where speech is least likely).
2. For each piece: the CTC head gives the phones; a Viterbi pass over
   `[SIL] p1 [SIL] p2 ... [SIL]` on the aligner gives their boundaries and the pauses, as in the
   evaluation. A pause cut in two by a piece boundary is joined back into one.
3. The candidates of a phone are the CTC posteriors at the frame where that phone is most
   probable, over the phone symbols only, renormalised to sum to 1.

Read the probabilities with care: they are the model's own and are **not calibrated**. The top-1
is often given about 0.99 even when it is wrong, but the **order** of the candidates is
informative: when the top-1 is wrong, the right phone is often 2nd or 3rd.

The model cannot see the first and last ~10 ms of each piece, so a gap of 15 to 25 ms with no
label is left at each cut (about every 10 s) and at the start and end of the recording.

## Data

### Training corpus

The models were trained on French speech with phone transcriptions. Utterances containing an
unknown word were removed. Any French (or other) corpus with phone-level transcriptions can be used
in the same format.

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
    "chunks": [{"start": 0.0, "end": 9.8}, {"start": 15.1, "end": 29.8}],
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
| `chunks` | yes | the segments the model sees, in seconds (see [Building a manifest](#building-a-manifest)) |
| `ref_intervals` | yes | the reference phones and their intervals |
| `style` | no | groups the results by speaking style |
| `pause_intervals` | no | reference pauses; enables the silence scores |
| `excluded_intervals` | no | regions ignored by the silence scores |

### Building a manifest

`make_manifest.py` builds a manifest from a folder of recordings and Praat TextGrids that have a
phone tier. The chunks come from the [WhisperX](https://github.com/m-bain/whisperX) voice
activity detector (pyannote segmentation, onset 0.5 / offset 0.363). Speech is cut and merged
into chunks of at most 10 s, with pauses kept inside a chunk. WhisperX pins its own `torch`
version, so install it in a separate environment:

```bash
python -m venv .venv-manifest && source .venv-manifest/bin/activate
pip install -r requirements-manifest.txt

python make_manifest.py --audio_dir wav/ --textgrid_dir textgrids/ --tier phones \
    --phone_map phone_map.json --skip_labels "<noise>" "<unk>" \
    --style=_read=read,_spont=spontaneous --out manifest.json
```

| option | meaning |
|---|---|
| `--textgrid_suffix` | TextGrid name = audio id + suffix (default `.TextGrid`). Write `--textgrid_suffix=-x.TextGrid` when it starts with `-` |
| `--phone_map` | JSON `{"tier label": "phone"}` that rewrites the reference labels into the model's phone set (`vocab.json`). Unmapped labels are kept |
| `--pause_labels` | labels of pauses (default: empty, `SIL`, `sil`, `spn`, `SP`, `<SIL>`, `_`). They become `pause_intervals` |
| `--skip_labels` | other non-phone labels (noise, unknown words, ...). They become `excluded_intervals` |
| `--style` | `SUBSTRING=STYLE` rules on the file id, comma-separated, for per-style results |
| `--max_chunk_dur` | maximum chunk length in seconds (default 10) |

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

**Metrics** (`jointrecalign/evaluation/metrics.py`):
- boundary precision, recall and F1 within 10, 20 and 50 ms (main score: F1@20);
- median, mean and 90th percentile of the absolute boundary error, and rate of errors > 50 ms;
- signed shift of phone starts and ends;
- PER;
- duration fidelity: slope and correlation of hypothesis versus reference phone durations
  (`_sp`: phones with no pause on either side);
- for SIL models with `pause_intervals`: frame-level SIL precision/recall/F1 and pause-level F1
  at IoU ≥ 0.5, over each whole file: a pause cut by a chunk boundary counts once, and audio left
  out between chunks counts as silence (`sil_eval.evaluate(..., sil_mode="chunk")` gives the
  older per-chunk scores, which count such a pause as two false silences).

The script writes these files to `--out_dir`:
- per-file, per-style and global CSVs;
- the alignments (`.pkl`);
- one JSON line per checkpoint in `summary.jsonl`.

## Tests

CPU only. They need no data or pretrained weights:

```bash
python tests/test_sil_fs.py   # latent-SIL DP vs brute force, gradients, batching, == ForwardSumLoss without SIL
python tests/test_model.py    # full model on a tiny random wav2vec2: losses, gradient routing, SIL, L_energy, decoders
python tests/test_transcribe.py   # chunking, pause joining, candidates, TextGrid/CSV, end to end on a tiny model
python tests/test_reference.py    # TextGrid reading, reference labels, comparison scores
```

The end-to-end part of `test_transcribe.py` needs `silero-vad` (in `requirements.txt`) and is
skipped without it.

## Repository layout

```
train_ctc.py              stages 0-2: data preparation, CTC fine-tuning (20 or 10 ms)
train_aligner.py          stage 3: joint recognition + alignment model
evaluate.py               alignment evaluation on a manifest
make_manifest.py          evaluation manifest from audio + TextGrids (WhisperX VAD chunks)
transcribe.py             recordings -> TextGrids (= JointRecAlign)
GUIDE.md                  step-by-step guide for phoneticians
recipes/                  one script per stage, with the settings of the trained models
jointrecalign/
  transcribe.py           Silero chunking, decoding, top-5 candidates, TextGrid / CSV writers
  transcribe_cli.py       the JointRecAlign command
  report.py, report_assets/  the HTML report
  reference.py            comparison with reference TextGrids
  download.py             download of the released model into a cache folder
  praat.py, praat_plugin/ the Praat plugin
  modules.py              ForwardSum loss, ConvBank, cross-attention, prior helpers
  pool_alignment.py       CTC + aligner with pooled phone keys, latent SIL, gradient routing
  model.py                + masked reconstruction (L_m), EMA quantizer, L_energy
  sil_fs.py               latent optional SIL: topology, forward-backward, Viterbi
  evaluation/
    attention_eval.py     manifest loop, DTW decode, frame timing
    sil_eval.py           SIL Viterbi decode and silence scores
    ctcfa_eval.py         CTC forced-alignment baseline
    vad_chunking.py       WhisperX VAD chunking (used by make_manifest.py)
    metrics.py            boundary F1, errors, PER
tests/
```

## Generative AI Use Disclosure

The code in this repository was refactored, organised and refined with the aid of
[Claude Code](https://claude.com/claude-code) (Anthropic).

## License

Code: MIT, see `LICENSE`. Released model weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use, with attribution).
