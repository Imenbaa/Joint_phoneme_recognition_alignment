#!/usr/bin/env python3
"""
Fine-tune a pretrained wav2vec2 model as a CTC phone recogniser.

Used for both CTC stages of the recipe:
  - 20 ms frames (default conv stride), from a self-supervised wav2vec2;
  - 10 ms frames (--last_conv_stride 1), from the 20 ms CTC model. Every
    pretrained weight still loads; only the hop of the last conv layer changes.

Data: a Hugging Face "audiofolder" (--data_dir) with train/validation/test
splits, each with a metadata.csv holding `file_name` and `phonemes_single_v2`
(space-separated phone symbols). The first run preprocesses it into an Arrow
dataset (--preprocessed_path) and writes the phone vocabulary (--vocab_path);
later runs, and train_aligner.py, reuse both. --preprocess_only stops there.

Multi-GPU: launch with torchrun (DDP). Preprocess once on a single process first.
The best epoch on validation PER is saved to --output_dir.
"""
import argparse
import os
import tempfile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None,
                   help="audiofolder dataset; read only when --preprocessed_path does not exist")
    p.add_argument("--preprocessed_path", required=True)
    p.add_argument("--vocab_path", required=True)
    p.add_argument("--output_dir", default=None, help="required unless --preprocess_only")
    p.add_argument("--init_model", default="LeBenchmark/wav2vec2-FR-7K-large",
                   help="pretrained wav2vec2 (HF Hub id or local folder)")
    p.add_argument("--last_conv_stride", type=int, default=2,
                   help="2 = standard 20 ms frames; 1 = 10 ms frames")
    p.add_argument("--input_sr", type=int, default=16000,
                   help="input sampling rate (32000: upsampled input, 10 ms frames)")
    p.add_argument("--epochs", type=float, default=30)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=32, help="per device")
    p.add_argument("--grad_acc", type=int, default=2)
    p.add_argument("--train_cnn", action="store_true",
                   help="also train the conv front-end (frozen by default)")
    p.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing")
    p.add_argument("--num_proc", type=int, default=16, help="processes for dataset map/filter")
    p.add_argument("--num_workers", type=int, default=12, help="dataloader workers per process")
    p.add_argument("--eval_max", type=int, default=0,
                   help="validation utterances used for the per-epoch PER; 0 = all")
    p.add_argument("--cache_dir", default=None,
                   help="Arrow map caches during preprocessing (default: system temp dir)")
    p.add_argument("--preprocess_only", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="resume from the newest checkpoint in --output_dir")
    a = p.parse_args()
    if not a.preprocess_only and not a.output_dir:
        p.error("--output_dir is required unless --preprocess_only")
    return a


args = parse_args()
model_name = args.init_model
LAST_CONV_STRIDE = args.last_conv_stride
EPOCHS = args.epochs
INPUT_SR = args.input_sr
LR = args.lr
FREEZE_CNN = not args.train_cnn
data_dir = args.data_dir
output_dir = args.output_dir
vocab_path = args.vocab_path
preprocessed_path = args.preprocessed_path
NUM_PROC = args.num_proc
EVAL_MAX = args.eval_max
GRAD_CKPT = not args.no_grad_ckpt
TRAIN_BS = args.batch_size
GRAD_ACC = args.grad_acc
PREPROCESS_ONLY = args.preprocess_only
CACHE_DIR = args.cache_dir or tempfile.gettempdir()
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(vocab_path)), exist_ok=True)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN = int(os.environ.get("RANK", "0")) == 0

import inspect
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Union

import editdistance
import numpy as np
import torch
from datasets import Audio, load_dataset, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# Ampere: let cuDNN/cuBLAS use TF32 for fp32 ops. Free speed on A100.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# bf16 on Ampere and newer, fp16 otherwise, nothing on CPU.
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
USE_FP16 = torch.cuda.is_available() and not USE_BF16

if IS_MAIN:
    print(
        f"CUDA: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()} | "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
        f"world_size={WORLD_SIZE} | bf16={USE_BF16} fp16={USE_FP16}",
        flush=True,
    )

# =========================
# Steps 1-6: Preprocessing (cached)
# =========================

if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset from disk...", flush=True)
    dataset = load_from_disk(preprocessed_path)
    _sr_tag = os.path.join(preprocessed_path, "input_sr.txt")
    _cached_sr = int(open(_sr_tag).read()) if os.path.exists(_sr_tag) else 16000
    if _cached_sr != INPUT_SR:
        raise RuntimeError(f"{preprocessed_path} holds {_cached_sr} Hz audio but "
                           f"--input_sr {INPUT_SR}. Point --preprocessed_path to a new folder.")

else:
    if WORLD_SIZE > 1:
        raise RuntimeError(
            f"No preprocessed dataset at {preprocessed_path} and this is a "
            f"{WORLD_SIZE}-process run. Preprocess once on a single process first:\n"
            f"    python {os.path.basename(__file__)} --preprocess_only ...\n"
            "then launch training with torchrun."
        )

    print("Preprocessing from scratch...", flush=True)

    # Step 1: Load dataset
    print("Step 1: Loading dataset", flush=True)
    if not data_dir:
        raise SystemExit(f"{preprocessed_path} does not exist: pass --data_dir to build it")
    dataset = load_dataset("audiofolder", data_dir=data_dir)
    if "transcription" in dataset["train"].column_names:
        dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=INPUT_SR))

    # Step 2: Filter UNK (utterances whose optional `phonemes` column marks an
    # unknown word as |ø|n|k|)
    print("Step 2: Filtering UNK", flush=True)

    def has_no_unk(example):
        return "|ø|n|k|" not in example["phonemes"]

    if "phonemes" in dataset["train"].column_names:
        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].filter(has_no_unk, num_proc=NUM_PROC)

    # Step 4: Build vocab
    print("Step 4: Build vocab", flush=True)
    vocab_set = set()
    for split in ["train", "validation", "test"]:
        for seq in dataset[split]["phonemes_single_v2"]:
            vocab_set.update(seq.split())

    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1}
    vocab_dict.update({v: i + 2 for i, v in enumerate(vocab_list)})

    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f)

    print(f"Vocab size: {len(vocab_dict)} — {list(vocab_dict.keys())}", flush=True)

    # Step 5: Processor (needed inside preprocess_batch)
    print("Step 5: Create processor", flush=True)
    _tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="",
    )
    _feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=INPUT_SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    # Step 6: Preprocessing
    print("Step 6: Preprocessing", flush=True)

    def preprocess_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == INPUT_SR for sr in sampling_rates), "Unexpected sampling rate"

        inputs = _feature_extractor(
            audio_arrays,
            sampling_rate=INPUT_SR,
            return_tensors="np",
            padding=False,
        )
        # Direct tokenizer call: as_target_processor() is deprecated and adds
        # a context-manager round trip per batch.
        labels = _tokenizer(batch["phonemes_single_v2"]).input_ids

        return {
            # float32 instead of float64: halves the size of the Arrow dataset
            # and therefore the bytes read per epoch.
            "input_values": [np.asarray(x, dtype=np.float32) for x in inputs.input_values],
            "labels": labels,
            "length": [len(a) for a in audio_arrays],
        }

    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True,
            batch_size=32,
            num_proc=NUM_PROC,
            remove_columns=dataset[split].column_names,
            cache_file_name=os.path.join(
                CACHE_DIR, f"map_ctc_{INPUT_SR}_{split}.arrow"
            ),
        )

    # Save to disk for future runs
    print("Saving preprocessed dataset to disk...", flush=True)
    dataset.save_to_disk(preprocessed_path)
    with open(os.path.join(preprocessed_path, "input_sr.txt"), "w") as f:
        f.write(str(INPUT_SR))

if PREPROCESS_ONLY:
    print("--preprocess_only: dataset written, exiting before training.", flush=True)
    sys.exit(0)

# =========================
# Always load processor
# =========================
print("Loading processor...", flush=True)
if not os.path.exists(vocab_path):
    raise FileNotFoundError(
        f"{vocab_path} is missing but the preprocessed dataset exists. "
        "The label ids in the cache were built from that vocab — rebuild both "
        "together (delete the preprocessed directory and rerun)."
    )

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="",
)
# Attention mask on: LeBenchmark-large uses layer-norm feature extraction, so
# zero padding without a mask changes the outputs (cost 0.55 PER points at 32 kHz).
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=INPUT_SR,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

# Cap the dev set used for per-epoch monitoring. Deterministic slice, so the
# number stays comparable across runs. --eval_max 0 restores the full dev set.
eval_dataset = dataset["validation"]
if EVAL_MAX and len(eval_dataset) > EVAL_MAX:
    eval_dataset = eval_dataset.select(range(EVAL_MAX))
    print(
        f"Eval on a fixed subset of {EVAL_MAX} / {len(dataset['validation'])} "
        "dev utterances (--eval_max 0 for the full set).",
        flush=True,
    )

# =========================
# Step 7: Data collator
# =========================
print("Step 7: Data collator", flush=True)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    pad_to_multiple_of: int = 8  # keeps tensor cores fed under bf16/fp16

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# =========================
# Step 8: Metrics (PER)
# =========================
print("Step 8: Metrics", flush=True)


def preprocess_logits_for_metrics(logits, labels):
    """Argmax on the GPU, per batch.

    Without this the Trainer accumulates [n_utts, n_frames, vocab] float logits
    across the whole dev set before compute_metrics ever runs — tens of GB moved
    host-side each epoch, all of it discarded by the argmax below.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def ctc_collapse(token_list):
    collapsed = []
    prev = None
    for t in token_list:
        if t != prev and t != "[PAD]":
            collapsed.append(t)
        prev = t
    return collapsed


def compute_metrics(pred):
    # Already argmaxed by preprocess_logits_for_metrics.
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    convert = processor.tokenizer.convert_ids_to_tokens
    pad_id = processor.tokenizer.pad_token_id

    all_per = []
    for pred_id_seq, label_id_seq in zip(pred_ids, label_ids):
        # -100 = the Trainer's padding when it stacks eval batches of different
        # lengths; convert(-100) gives [UNK], which added one insertion per utterance.
        pred_tokens = ctc_collapse([convert(int(i)) for i in pred_id_seq if i != -100])
        label_tokens = [
            convert(int(i)) for i in label_id_seq if i not in (-100, pad_id)
        ]
        all_per.append(
            editdistance.eval(pred_tokens, label_tokens) / max(1, len(label_tokens))
        )

    return {"PER": float(np.mean(all_per))}


# =========================
# Step 9: Model
# =========================
print("Step 9: Model", flush=True)

model_kwargs = dict(
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    layerdrop=0.0,                # <-- required for DDP
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
if LAST_CONV_STRIDE != 2:
    model_kwargs["conv_stride"] = [5, 2, 2, 2, 2, 2, LAST_CONV_STRIDE]
# Scaled dot-product attention where the installed transformers supports it.
if "attn_implementation" in inspect.signature(Wav2Vec2ForCTC.from_pretrained).parameters:
    model_kwargs["attn_implementation"] = "sdpa"

model = Wav2Vec2ForCTC.from_pretrained(model_name, **model_kwargs)
if FREEZE_CNN:
    model.freeze_feature_encoder()
model.config.input_sampling_rate = INPUT_SR      # saved in config.json, read by the eval
if IS_MAIN:
    print(f"conv front-end {'frozen' if FREEZE_CNN else 'TRAINED'}", flush=True)
    print(f"input {INPUT_SR} Hz, conv_stride={model.config.conv_stride} -> "
          f"{model.config.inputs_to_logits_ratio / INPUT_SR * 1000:.0f} ms per frame",
          flush=True)
model.config.ctc_zero_infinity = True

# =========================
# Step 10: Training
# =========================
print("Step 10: Training", flush=True)

args_kwargs = dict(
    output_dir=output_dir,
    group_by_length=True,
    length_column_name="length",
    remove_unused_columns=False,   # <-- keep the `length` column for the sampler
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=TRAIN_BS,
    gradient_accumulation_steps=GRAD_ACC,
        save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="PER",
    greater_is_better=False,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_steps=500,
    weight_decay=0.005,
    logging_steps=100,
    gradient_checkpointing=GRAD_CKPT,
    max_grad_norm=2.0,
    bf16=USE_BF16,
    fp16=USE_FP16,
    dataloader_num_workers=args.num_workers,
    dataloader_pin_memory=True,
    ddp_find_unused_parameters=False,
    # TensorBoard event files go to <output_dir>/runs
    report_to=["tensorboard"] if IS_MAIN else [],
)

# use_reentrant=False is required for gradient checkpointing under DDP.
if GRAD_CKPT:
    args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

# Persistent dataloader workers: respawning 12 processes every epoch is pure
# overhead. Only available in newer transformers.
_ta_params = inspect.signature(TrainingArguments.__init__).parameters
if "dataloader_persistent_workers" in _ta_params:
    args_kwargs["dataloader_persistent_workers"] = True
if "dataloader_prefetch_factor" in _ta_params:
    args_kwargs["dataloader_prefetch_factor"] = 4

# Renamed from evaluation_strategy to eval_strategy in recent transformers.
if "eval_strategy" in _ta_params:
    args_kwargs["eval_strategy"] = "epoch"
else:
    args_kwargs["evaluation_strategy"] = "epoch"

# Drop anything this transformers version does not know about.
args_kwargs = {k: v for k, v in args_kwargs.items() if k in _ta_params}

training_args = TrainingArguments(**args_kwargs)


class CTCTrainer(Trainer):
    def _get_eval_sampler(self, eval_dataset):
        return None


trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    train_dataset=dataset["train"],
    eval_dataset=eval_dataset,
)

trainer.train(resume_from_checkpoint=args.resume or None)
trainer.save_model(output_dir)
if IS_MAIN:
    processor.save_pretrained(output_dir)

# =========================
# Save logs
# =========================
if IS_MAIN:
    logs_path = os.path.join(output_dir, "logs.json")
    with open(logs_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)