#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the joint recognition + alignment model (Wav2Vec2ForCTCPoolContrastive)
from a wav2vec2 CTC checkpoint (train_ctc.py).

    loss = ctc_weight * L_CTC + align_weight * L_FS + contrastive_weight * L_m

  L_CTC : ground-truth phones on the CTC head.
  L_FS  : ForwardSum loss on the cross-attention energy (beta-binomial prior),
          aligning the ground-truth phones; ramps in over --align_warmup_steps.
  L_m   : masked-frame InfoNCE against EMA-quantized targets, reconstructed
          from [X_masked ; att @ phone_keys ; pos/dur features]
          (+ --lm_energy_weight * L_energy).

Phone keys are CTC-posterior-weighted averages of encoder frames, windowed
around each phone (--pool_mode window, --window_sigma_frames).

Gradient routing (--encoder_alpha): L_CTC trains the whole encoder; L_FS and
L_m reach only the top --encoder_top_layers transformer layers, through the
queries, with their gradient scaled by --encoder_alpha / --lm_encoder_alpha
(ramped in from --encoder_alpha_start_steps over --encoder_alpha_ramp_steps).
Without --encoder_alpha, the --detach_* flags set the routing.

Latent optional silence (--sil): from --sil_start_steps on, L_FS uses optional
SIL states ([SIL] p1 [SIL] p2 ... [SIL], slots per --sil_positions) with one
learned key, an entry cost and >= --sil_min_frames frames, and L_m reads its
attention from the forward-backward occupancy (--lm_att_source fs_posterior).

Evaluation
----------
PER on the validation split from the CTC head's greedy decode, every
--eval_steps. With --eval_manifest, the checkpoint is also scored as an
aligner on that manifest (see jointrecalign/evaluation) and the boundary metrics
are logged to TensorBoard under heldout/. They are only logged: no checkpoint
is selected on them.

Training resumes automatically from the newest checkpoint in --out_dir.
"""

import os
import sys
import json
import inspect
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers.trainer_utils import get_last_checkpoint

from jointrecalign.model import Wav2Vec2ForCTCPoolContrastive, initialize_quantizer


# =========================
# Data pipeline
# =========================
def build_or_load_dataset(args):
    sr_tag = os.path.join(args.preprocessed_path, "input_sr.txt")
    if os.path.exists(args.preprocessed_path):
        print(f"Loading preprocessed dataset from {args.preprocessed_path}")
        cached = int(open(sr_tag).read()) if os.path.exists(sr_tag) else 16000
        if cached != args.sampling_rate:
            raise SystemExit(f"{args.preprocessed_path} holds {cached} Hz audio but "
                             f"--sampling_rate {args.sampling_rate}: use a new --preprocessed_path")
        return load_from_disk(args.preprocessed_path)

    if not args.audio_dir:
        raise SystemExit(f"{args.preprocessed_path} does not exist: pass --audio_dir to build it")
    print("Preprocessing from scratch...")
    print("Step 1: Loading dataset")
    dataset = load_dataset("audiofolder", data_dir=args.audio_dir)
    if "transcription" in dataset["train"].column_names:
        dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    print("Step 2: Filtering UNK and very short seq")
    min_samples = int(args.min_audio_seconds * args.sampling_rate)
    min_phones = args.min_phones

    def is_valid(example):
        has_unk = "|ø|n|k|" in example.get("phonemes", "")
        too_short = len(example["audio"]["array"]) < min_samples
        few_phones = len(example["phonemes_single_v2"].split()) < min_phones
        return not has_unk and not too_short and not few_phones

    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(is_valid, num_proc=args.num_proc)
        after = len(dataset[split])
        print(f"  {split}: {before} -> {after} ({before - after} removed)")

    print("Step 3: Build vocab")
    vocab_set = set()
    for split in dataset.keys():
        for seq in dataset[split]["phonemes_single_v2"]:
            vocab_set.update(seq.split())
    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1}
    vocab_dict.update({v: i + 2 for i, v in enumerate(vocab_list)})
    os.makedirs(os.path.dirname(args.vocab_path) or ".", exist_ok=True)
    with open(args.vocab_path, "w") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    print(f"  Vocab size: {len(vocab_dict)} -> {args.vocab_path}")

    tok = Wav2Vec2CTCTokenizer(
        vocab_file=args.vocab_path,
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="",
    )
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=args.sampling_rate,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

    print("Step 4: Preprocessing")

    def preprocess_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == args.sampling_rate for sr in sampling_rates)
        inputs = fe(
            audio_arrays, sampling_rate=args.sampling_rate,
            return_tensors="np", padding=False, return_attention_mask=True,
        )
        with proc.as_target_processor():
            labels = proc(batch["phonemes_single_v2"]).input_ids
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "length": [len(x) for x in inputs.input_values],
        }

    for split in list(dataset.keys()):
        cache_file = (
            os.path.join(args.cache_dir, f"map_FS_{split}.arrow")
            if args.cache_dir else None
        )
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True, batch_size=32, num_proc=args.num_proc,
            remove_columns=dataset[split].column_names,
            cache_file_name=cache_file,
        )
    print(f"Saving preprocessed dataset to {args.preprocessed_path}")
    dataset.save_to_disk(args.preprocessed_path)
    with open(sr_tag, "w") as f:
        f.write(str(args.sampling_rate))
    return dataset


def add_word_slots(ds, metadata_csv, vocab_path):
    """Add `sil_slots` (word-boundary SIL slot indices, edges included) to a
    preprocessed split, without re-reading audio. Rows are matched on their
    label string; word boundaries come from `phonemes_single` (words separated
    by spaces, one character per phone = the v2 tokens)."""
    import csv
    inv = {i: t for t, i in json.load(open(vocab_path)).items()}
    table, conflicts = {}, 0
    for r in csv.DictReader(open(metadata_csv)):
        v2, ps = r.get("phonemes_single_v2"), r.get("phonemes_single")
        if not v2 or not ps:
            continue
        words = ps.split()
        key = "".join(words)
        if list(key) != v2.split():
            continue
        slots = [0] + np.cumsum([len(w) for w in words]).tolist()   # last = N
        if key in table and table[key] != slots:
            conflicts += 1                       # same phones, other word split: keep the common slots
            table[key] = sorted(set(table[key]) & set(slots))
            continue
        table[key] = slots
    col, miss = [], 0
    for lab in ds["labels"]:
        key = "".join(inv[int(i)] for i in lab)
        sl = table.get(key)
        if sl is None:
            miss += 1
            sl = []                               # -> 'all' for this row (warned in the model)
        col.append(sl)
    print(f"[sil] word slots: {len(ds) - miss}/{len(ds)} rows matched "
          f"({conflicts} label strings with two word splits -> common slots)")
    if miss > 0.01 * len(ds):
        raise SystemExit(f"[sil] {miss} rows without word boundaries: wrong --sil_metadata_csv?")
    return ds.add_column("sil_slots", col)


# =========================
# Collator
# =========================
@dataclass
class DataCollatorCTCAlignment:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    apply_spec_augment: bool = False
    mask_time_prob_min: float = 0.05
    mask_time_prob_max: float = 0.40
    mask_time_length: int = 2
    feat_extract_output_lengths_fn: Any = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_values"] != 0).long()

        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if "sil_slots" in features[0]:
            L = max(1, max(len(f["sil_slots"]) for f in features))
            ss = torch.full((len(features), L), -1, dtype=torch.long)
            for i, f in enumerate(features):
                if len(f["sil_slots"]):
                    ss[i, :len(f["sil_slots"])] = torch.as_tensor(f["sil_slots"], dtype=torch.long)
            batch["sil_slots"] = ss

        if self.apply_spec_augment:
            bsz, raw_len = batch["input_values"].shape
            seq_len = int(self.feat_extract_output_lengths_fn(raw_len))
            mask_prob = float(
                np.random.uniform(self.mask_time_prob_min, self.mask_time_prob_max)
            )
            batch["mask_time_indices"] = torch.tensor(
                _compute_mask_indices(
                    (bsz, seq_len),
                    mask_prob=mask_prob,
                    mask_length=self.mask_time_length,
                ),
                dtype=torch.long,
            )

        return batch


# =========================
# PER
# =========================
def _levenshtein(hyp, ref):
    if len(hyp) == 0:
        return len(ref)
    if len(ref) == 0:
        return len(hyp)
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (h != r)))
        prev = cur
    return prev[-1]


def make_compute_metrics(blank_id):
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        total_err, total_len = 0, 0
        for p_row, l_row in zip(pred_ids, label_ids):
            seq = [int(x) for x in np.asarray(p_row).ravel() if int(x) != -100]
            collapsed, prev = [], -1
            for t in seq:
                if t != prev:
                    collapsed.append(t)
                    prev = t
            hyp = [t for t in collapsed if t != blank_id]
            ref = [int(x) for x in np.asarray(l_row).ravel() if int(x) != -100]
            total_err += _levenshtein(hyp, ref)
            total_len += len(ref)
        return {"per": total_err / max(total_len, 1)}
    return compute_metrics


# =========================
# Trainer
# =========================
class CTCAlignmentTrainer(Trainer):
    def __init__(self, *args, backbone_lr=1e-5, alignment_lr=1e-4,
                 weight_decay=1e-4,
                 ctc_weight=1.0, align_weight=1.0, contrastive_weight=1.0,
                 align_warmup_steps=0,
                 align_start_steps=0, contrastive_start_steps=0,
                 contrastive_warmup_steps=0, reseed_codebook_at_lm_start=True,
                 encoder_alpha=None, encoder_alpha_start_steps=0,
                 encoder_alpha_ramp_steps=0,
                 lm_encoder_alpha=None, encoder_alpha_stop_steps=None,
                 encoder_alpha_stop_ramp_steps=0,
                 gt_targets_only=True, prior_constant=True,
                 pred_target_start_frac=0.35, pred_target_end_frac=0.75,
                 prior_anneal_start_frac=0.45, prior_anneal_end_frac=0.85,
                 viz_every_n_steps=500, viz_max_samples=2,
                 tokenizer_for_viz=None, processor_for_save=None,
                 probe_dataset=None, probe_collator=None,
                 use_sil=False, sil_start_steps=0, sil_init="random",
                 **kwargs):
        self._backbone_lr = backbone_lr
        self._alignment_lr = alignment_lr
        self._wd = weight_decay
        self._ctc_w = ctc_weight
        self._align_w = align_weight
        self._contr_w = contrastive_weight
        self._align_warmup_steps = align_warmup_steps
        # Delayed starts (default 0: L_FS ramps from step 0, L_m at full
        # weight from step 0).
        self._align_start = int(align_start_steps)
        self._contr_start = int(contrastive_start_steps)
        self._contr_warmup_steps = int(contrastive_warmup_steps)
        self._reseed_at_lm_start = bool(reseed_codebook_at_lm_start)
        self._reseeded = False
        # None = legacy routing (detach flags); float = joint routing, L_FS
        # reaches the encoder through the queries scaled by this value.
        self._enc_alpha = None if encoder_alpha is None else float(encoder_alpha)
        self._enc_alpha_start = int(encoder_alpha_start_steps)
        self._enc_alpha_ramp = int(encoder_alpha_ramp_steps)
        self._lm_alpha = None if lm_encoder_alpha is None else float(lm_encoder_alpha)
        self._enc_alpha_stop = None if encoder_alpha_stop_steps is None else int(encoder_alpha_stop_steps)
        self._enc_alpha_stop_ramp = int(encoder_alpha_stop_ramp_steps)
        self._gt_only = bool(gt_targets_only)
        self._prior_constant = bool(prior_constant)
        self._pred_start = pred_target_start_frac
        self._pred_end = pred_target_end_frac
        self._prior_start = prior_anneal_start_frac
        self._prior_end = prior_anneal_end_frac
        self._viz_every = int(viz_every_n_steps or 0)
        self._viz_max_samples = max(1, int(viz_max_samples))
        self._tokenizer_for_viz = tokenizer_for_viz
        self._processor_for_save = processor_for_save
        self._tb_writer = None
        self._probe_dataset = probe_dataset
        self._probe_collator = probe_collator
        self._probe_batch = None
        self._metric_sums = {}
        self._metric_count = 0
        self._use_sil = bool(use_sil)
        self._sil_start = int(sil_start_steps)
        self._sil_init = sil_init
        self._sil_sums, self._sil_counts = {}, {}
        super().__init__(*args, **kwargs)

    # ---------- schedules ----------
    def _frac_ramp(self, start_frac, end_frac):
        """0 before start_frac of training, 1 after end_frac, linear between."""
        total = int(self.state.max_steps)
        if total <= 1:
            return 0.0
        step = int(self.state.global_step)
        s = start_frac * total
        e = end_frac * total
        if step <= s:
            return 0.0
        if step >= e:
            return 1.0
        return float((step - s) / max(e - s, 1.0))

    @staticmethod
    def _delayed_ramp(step, start, warmup):
        """0 before `start`, then linear 0 -> 1 over `warmup` steps (1 if warmup=0).
        With start=0 this is exactly the original ramp min(1, step / warmup)."""
        if step < start:
            return 0.0
        if warmup <= 0:
            return 1.0
        return min(1.0, (step - start) / float(warmup))

    def _current_forward_kwargs(self):
        step = self.state.global_step
        align_ramp = self._delayed_ramp(step, self._align_start, self._align_warmup_steps)
        contr_ramp = self._delayed_ramp(step, self._contr_start, self._contr_warmup_steps)
        if self._gt_only:
            pred_target_prob = 0.0  # always align against ground-truth phones
        else:
            pred_target_prob = self._frac_ramp(self._pred_start, self._pred_end)
        kw = dict(
            ctc_weight=self._ctc_w,
            align_weight=align_ramp * self._align_w,
            contrastive_weight=contr_ramp * self._contr_w,
            # 0 -> ground-truth phones, 1 -> CTC-predicted phones
            pred_target_prob=pred_target_prob,
            # 1 -> full prior, 0 -> no prior
            prior_weight=(1.0 if self._prior_constant
                          else 1.0 - self._frac_ramp(self._prior_start, self._prior_end)),
        )
        if self._enc_alpha is not None:
            shape = self._delayed_ramp(step, self._enc_alpha_start, self._enc_alpha_ramp)
            if self._enc_alpha_stop is not None:        # optional switch-off
                shape *= 1.0 - self._delayed_ramp(step, self._enc_alpha_stop,
                                                  self._enc_alpha_stop_ramp)
            kw["encoder_alpha"] = self._enc_alpha * shape
            kw["lm_encoder_alpha"] = (self._lm_alpha or 0.0) * shape
        if self._use_sil:
            kw["sil_active"] = step >= self._sil_start
        return kw

    # ---------- logging ----------
    def _ensure_tb_writer(self):
        if self._tb_writer is not None:
            return self._tb_writer
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            return None
        log_dir = self.args.logging_dir or os.path.join(self.args.output_dir, "runs")
        os.makedirs(log_dir, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=log_dir)
        return self._tb_writer

    # ---------- alignment visualisation ----------
    def _get_probe_batch(self):
        if self._probe_batch is not None:
            return self._probe_batch
        if self._probe_dataset is None or self._probe_collator is None:
            return None
        n = min(self._viz_max_samples, len(self._probe_dataset))
        feats = [self._probe_dataset[i] for i in range(n)]
        self._probe_batch = self._probe_collator(feats)
        return self._probe_batch

    def _render_alignment(self, att, phone_ids, T, P, tag, step, subtitle):
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import PIL.Image
        from librosa.sequence import dtw as _dtw

        writer = self._ensure_tb_writer()
        if writer is None:
            return

        argmax_path = att.argmax(axis=-1)
        try:
            _, dtw_align = _dtw(C=-att.T, step_sizes_sigma=np.array([[1, 1], [1, 0]]))
            dtw_path = np.full(T, -1, dtype=np.int32)
            for t_, p_ in dtw_align:
                if 0 <= t_ < T and dtw_path[t_] < p_:
                    dtw_path[t_] = p_
        except Exception:
            dtw_path = None

        labels = []
        for pid in phone_ids[:P]:
            if self._tokenizer_for_viz is not None:
                labels.append(self._tokenizer_for_viz.convert_ids_to_tokens(int(pid)))
            else:
                labels.append(str(int(pid)))

        fig, ax = plt.subplots(
            figsize=(max(6.0, min(20.0, T * 0.04)), max(3.0, min(18.0, P * 0.15))),
            dpi=90,
        )
        im = ax.imshow(att.T, aspect="auto", origin="upper",
                       interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        ax.plot(np.arange(T), argmax_path, color="white", lw=0.6, alpha=0.6,
                label="argmax")
        if dtw_path is not None:
            m = dtw_path >= 0
            ax.plot(np.arange(T)[m], dtw_path[m], color="tab:red", lw=0.9,
                    alpha=0.85, label="DTW")
        ax.set_yticks(np.arange(P))
        ax.set_yticklabels(labels, fontsize=max(4, min(8, 400 / max(P, 1))))
        ax.set_xlabel("frame")
        ax.set_ylabel("phone")
        ax.set_title(f"step={step}  T={T} P={P}  {subtitle}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = np.array(PIL.Image.open(buf).convert("RGB"))
        writer.add_image(tag, np.transpose(img, (2, 0, 1)), global_step=step)

    def _log_probe_alignment(self):
        batch = self._get_probe_batch()
        if batch is None:
            return
        writer = self._ensure_tb_writer()
        if writer is None:
            return

        model = self.model
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                inputs = self._prepare_inputs({k: v for k, v in batch.items()})
                out = model(
                    **inputs,
                    ctc_weight=1.0, align_weight=0.0, contrastive_weight=0.0,
                    pred_target_prob=0.0,   # ground-truth phones: fixed axis
                    prior_weight=0.0,       # learned alignment only
                )
        finally:
            model.train(was_training)

        step = int(self.state.global_step)
        energy = out.energy_raw if out.energy_raw is not None else out.energy
        pm = out.predicted_phone_attention_mask
        att_all = (energy.float() + (1 - pm).unsqueeze(1).float() * -10000.0).softmax(-1)
        att_all = att_all.cpu().numpy()
        il = out.input_lengths.cpu().numpy()
        tl = out.text_lengths.cpu().numpy()
        pid = out.predicted_phone_ids.cpu().numpy()

        for b in range(att_all.shape[0]):
            T, P = int(il[b]), int(tl[b])
            if T <= 0 or P <= 0:
                continue
            self._render_alignment(
                att_all[b, :T, :P], pid[b], T, P,
                tag=f"probe/utt_{b}", step=step,
                subtitle="fixed probe | gt phones | no prior",
            )
        writer.flush()

    def _log_batch_alignment(self, outputs):
        energy = getattr(outputs, "energy", None)
        if energy is None:
            return
        kw = self._current_forward_kwargs()
        pm = outputs.predicted_phone_attention_mask
        att_all = (energy.detach().float()
                   + (1 - pm).unsqueeze(1).float() * -10000.0).softmax(-1).cpu().numpy()
        il = outputs.input_lengths.cpu().numpy()
        tl = outputs.text_lengths.cpu().numpy()
        pid = outputs.predicted_phone_ids.cpu().numpy()
        is_pred = outputs.align_target_is_pred.cpu().numpy()
        step = int(self.state.global_step)
        for b in range(min(self._viz_max_samples, att_all.shape[0])):
            T, P = int(il[b]), int(tl[b])
            if T <= 0 or P <= 0:
                continue
            src = "pred" if bool(is_pred[b]) else "gt"
            self._render_alignment(
                att_all[b, :T, :P], pid[b], T, P,
                tag=f"batch/sample_{b}", step=step,
                subtitle=f"{src} phones | prior_w={kw['prior_weight']:.2f}",
            )

    def _alignment_scalars(self, outputs):
        er = getattr(outputs, "energy_raw", None)
        if er is None or outputs.text_lengths is None:
            return {}
        with torch.no_grad():
            pm = outputs.predicted_phone_attention_mask
            att = (er.float() + (1 - pm).unsqueeze(1).float() * -10000.0).softmax(-1)
            am = att.argmax(-1)
            il = outputs.input_lengths
            tl = outputs.text_lengths
            sharp = ent = mono = cov = 0.0
            n = 0
            for b in range(att.size(0)):
                T, P = int(il[b].item()), int(tl[b].item())
                if T < 2 or P < 1:
                    continue
                a = att[b, :T, :P]
                sharp += float(a.max(-1).values.mean())
                e = -(a.clamp_min(1e-9).log() * a).sum(-1).mean()
                ent += float(e / max(np.log(max(P, 2)), 1e-9))
                d = am[b, 1:T] - am[b, :T - 1]
                mono += float((d >= 0).float().mean())
                cov += float(torch.unique(am[b, :T]).numel() / P)
                n += 1
            if n == 0:
                return {}
        return {
            "align/sharpness": sharp / n,
            "align/entropy_norm": ent / n,
            "align/monotonic": mono / n,
            "align/coverage": cov / n,
        }

    def _accumulate(self, outputs, inputs):
        for key in ("ctc_loss", "align_loss", "contrastive_loss", "lm_energy_loss", "codevector_perplexity"):
            v = getattr(outputs, key, None)
            if v is None:
                continue
            if torch.is_tensor(v) and v.numel() == 1:
                self._metric_sums[key] = self._metric_sums.get(key, 0.0) + float(v.detach().item())

        ctc_logits = getattr(outputs, "ctc_logits", None)
        if ctc_logits is not None:
            with torch.no_grad():
                max_probs = ctc_logits.softmax(dim=-1).max(dim=-1).values
                sharpness = float(max_probs.mean().item())
            self._metric_sums["posterior_sharpness"] = (
                self._metric_sums.get("posterior_sharpness", 0.0) + sharpness
            )

        self._metric_count += 1

        for key in ("sil_frame_frac", "sil_used_frac", "sil_mean_dur", "sil_logp_value",
                    "sil_code_top_share", "sil_code_ppl", "sil_energy_db"):
            v = getattr(outputs, key, None)
            if torch.is_tensor(v) and v.numel() == 1:
                self._sil_sums[key] = self._sil_sums.get(key, 0.0) + float(v.detach().item())
                self._sil_counts[key] = self._sil_counts.get(key, 0) + 1

    @torch.no_grad()
    def _reseed_codebook(self, model, outputs):
        """Re-draw the EMA codevectors from the CURRENT encoder frames of this batch.
        Used once, when L_m switches on after a delay: the codebook was seeded at
        step 0 from encoder frames that CTC has since reshaped."""
        m = getattr(model, "module", model)
        X = getattr(m, "_last_frame_hidden", None)
        if X is None or outputs.input_lengths is None:
            return
        X = X.detach().float()
        valid = (torch.arange(X.size(1), device=X.device)[None, :]
                 < outputs.input_lengths.to(X.device)[:, None])
        frames = X[valid]
        q = m.ema_quantizer
        n = q.num_vars
        if frames.size(0) >= n:
            idx = torch.randperm(frames.size(0), device=frames.device)[:n]
        else:
            idx = torch.randint(0, frames.size(0), (n,), device=frames.device)
        q.codevectors.copy_(frames[idx].to(q.codevectors.dtype))
        q.ema_embed.copy_(frames[idx].to(q.ema_embed.dtype))
        q.cluster_size.fill_(1.0)
        print(f"[train] step {self.state.global_step}: L_m switches on -> EMA codebook "
              f"re-seeded from {frames.size(0)} current encoder frames")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kw = self._current_forward_kwargs()
        m = getattr(model, "module", model)
        if kw.get("sil_active") and int(m.sil_initialized) == 0:
            if self._sil_init == "lowenergy":
                n, thr = m.init_sil_key_lowenergy(inputs["input_values"], inputs.get("attention_mask"))
                print(f"[sil] step {self.state.global_step}: sil_key initialised from the mean "
                      f"query of {n} lowest-energy frames (<= {thr:.1f} dB re p95)")
            else:
                m.sil_initialized.fill_(1)
                print(f"[sil] step {self.state.global_step}: SIL on (random sil_key)")
        if not self._use_sil:
            inputs = {k: v for k, v in inputs.items() if k != "sil_slots"}
        outputs = model(**inputs, **kw)
        step = int(self.state.global_step)
        if (self._reseed_at_lm_start and not self._reseeded and self._contr_start > 0
                and self._contr_start <= step <= self._contr_start + 1):
            self._reseed_codebook(model, outputs)
            self._reseeded = True
        self._accumulate(outputs, inputs)
        if (self._viz_every > 0
                and self.state.global_step > 0
                and self.state.global_step % self._viz_every == 0):
            try:
                self._log_probe_alignment()
                self._log_batch_alignment(outputs)
            except Exception as e:
                print(f"[viz] alignment-figure log failed at step "
                      f"{self.state.global_step}: {e}")
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ---------- evaluation ----------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels", None)
        with torch.no_grad():
            outputs = model(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask", None),
                labels=labels,
                skip_alignment=True,
                ctc_weight=1.0,
                align_weight=0.0,
                contrastive_weight=0.0,
            )
            loss = outputs.ctc_loss
            loss = loss.detach() if loss is not None else None
            pred_ids = outputs.ctc_logits.argmax(dim=-1)
            arange = torch.arange(pred_ids.size(1), device=pred_ids.device).unsqueeze(0)
            pred_ids = pred_ids.masked_fill(
                arange >= outputs.input_lengths.unsqueeze(1), -100
            )
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, pred_ids, labels)

    def log(self, logs, *args, **kwargs):
        if self._metric_count > 0:
            for k, total in self._metric_sums.items():
                logs[k] = total / self._metric_count
            kw = self._current_forward_kwargs()
            logs["sched/ctc_weight"] = kw["ctc_weight"]
            logs["sched/align_weight"] = kw["align_weight"]
            logs["sched/contrastive_weight"] = kw["contrastive_weight"]
            logs["sched/pred_target_prob"] = kw["pred_target_prob"]
            logs["sched/prior_weight"] = kw["prior_weight"]
            if "encoder_alpha" in kw:
                m = getattr(self.model, "module", self.model)
                eff = getattr(m, "_last_encoder_alpha", None)
                logs["sched/encoder_alpha"] = kw["encoder_alpha"] if eff is None else eff
                eff_lm = getattr(m, "_last_lm_alpha", None)
                logs["sched/lm_encoder_alpha"] = kw["lm_encoder_alpha"] if eff_lm is None else eff_lm
            # Is the reconstruction head using pos/dur at all?
            usage = getattr(self.model, "position_feature_usage", None)
            if callable(usage):
                try:
                    for k, v in usage().items():
                        logs[f"posfeat/{k}"] = v
                except Exception:
                    pass
            self._metric_sums = {}
            self._metric_count = 0
            if self._use_sil:
                logs["sched/sil_active"] = float(bool(kw.get("sil_active")))
                for k_, tot in self._sil_sums.items():
                    logs["sil/" + k_[4:]] = tot / max(self._sil_counts.get(k_, 1), 1)
                ff = logs.get("sil/frame_frac")
                if ff is not None and ff > 0.15 and self.state.global_step >= self._sil_start + 1000:
                    print(f"[sil] WARNING step {self.state.global_step}: SIL holds {ff:.0%} of the "
                          f"frames -> garbage state? (raise --sil_logp or --sil_min_frames)")
                self._sil_sums, self._sil_counts = {}, {}
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=False, state_dict=state_dict)
        if self._processor_for_save is not None:
            self._processor_for_save.save_pretrained(output_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        m = self.model
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "sil_key", "sil_logp")
        align_param_ids = {id(p) for p in m.get_alignment_parameters()}
        backbone_param_ids = {id(p) for p in m.get_backbone_parameters()}

        def split(params):
            decay, nd = [], []
            for n, p in params:
                if not p.requires_grad:
                    continue
                (nd if any(k in n for k in no_decay) else decay).append(p)
            return decay, nd

        align_named = [(n, p) for n, p in m.named_parameters() if id(p) in align_param_ids]
        backbone_named = [(n, p) for n, p in m.named_parameters() if id(p) in backbone_param_ids]
        align_decay, align_nd = split(align_named)
        bb_decay, bb_nd = split(backbone_named)

        n_grouped = len(align_decay) + len(align_nd) + len(bb_decay) + len(bb_nd)
        n_train = sum(1 for _, p in m.named_parameters() if p.requires_grad)
        if n_grouped != n_train:
            print(f"[train] WARNING: {n_train - n_grouped} trainable tensors are in "
                  f"no optimizer group and will never update. Check "
                  f"get_alignment_parameters() / get_backbone_parameters().")

        groups = [
            {"params": align_decay, "lr": self._alignment_lr, "weight_decay": self._wd},
            {"params": align_nd,    "lr": self._alignment_lr, "weight_decay": 0.0},
            {"params": bb_decay,    "lr": self._backbone_lr,  "weight_decay": self._wd},
            {"params": bb_nd,       "lr": self._backbone_lr,  "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(groups, lr=self._alignment_lr)
        return self.optimizer


class StopAfterStepsCallback(TrainerCallback):
    """End training at a given step while keeping state.max_steps (and so the
    fraction-based schedules) those of the full run."""
    def __init__(self, stop_step):
        self.stop_step = int(stop_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
            control.should_save = True
        return control


class OptimizerLastOnlyCallback(TrainerCallback):
    """After each save, delete optimizer.pt from every checkpoint but the newest.
    Model weights stay in all of them (evaluation); only the newest can resume."""
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        cks = sorted((d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")
                      and d.split("-")[-1].isdigit()), key=lambda d: int(d.split("-")[-1]))
        for d in cks[:-1]:
            f = os.path.join(args.output_dir, d, "optimizer.pt")
            if os.path.exists(f):
                os.remove(f)
        return control


class EvalManifestCallback(TrainerCallback):
    """Score the checkpoint as an aligner on an evaluation manifest and log the
    boundary metrics to TensorBoard (heldout/*). Logging only."""
    def __init__(self, manifest_path, feature_extractor, tokenizer, trainer,
                 every_n_evals=1, max_files=None, csv_dir=None):
        self.manifest = json.load(open(manifest_path))
        self.fe, self.tok, self.trainer = feature_extractor, tokenizer, trainer
        self.every, self.max_files, self.csv_dir = every_n_evals, max_files, csv_dir
        self._n = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self._n += 1
        if (self._n - 1) % self.every:
            return
        # frame stride and input rate are read from the model config
        from jointrecalign.evaluation.attention_eval import run_manifest, duration_fidelity
        from jointrecalign.evaluation.metrics import compute_metrics, align_sequences

        step = int(state.global_step)
        m = getattr(self.trainer.model, "module", self.trainer.model)
        if getattr(m, "use_sil", False):
            try:
                from jointrecalign.evaluation import sil_eval
                res, _, _ = sil_eval.evaluate(self.trainer.model, self.fe, self.tok, self.manifest,
                                              device="cuda", max_files=self.max_files)
                w = self.trainer._ensure_tb_writer()
                if w is not None:
                    for k, v in res.items():
                        w.add_scalar(f"heldout/{k}", v, step)
                    w.flush()
                if self.csv_dir:
                    json.dump(res, open(os.path.join(self.csv_dir, f"heldout_sil_{step}.json"), "w"), indent=1)
                nan = float("nan")
                print(f"[eval] step {step}  SIL decode: F1@10={res['F1@10']:.1f} F1@20={res['F1@20']:.1f} "
                      f"F1@50={res['F1@50']:.1f} P90={res['P90']:.1f} PER={res['PER']:.2f} | "
                      f"SIL frame P/R={res.get('sil_frame_P', nan):.0f}/{res.get('sil_frame_R', nan):.0f} "
                      f"pause F1={res.get('sil_pause_F1', nan):.0f} | DTW F1@20={res.get('F1@20_dtw', nan):.1f}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[eval] SIL eval failed at step {step}: {e}")
            return
        try:
            store = run_manifest(self.trainer.model, self.fe, self.tok,
                                 self.manifest, device="cuda",
                                 max_files=self.max_files)
            csv = (os.path.join(self.csv_dir, f"heldout_{step}.csv")
                   if self.csv_dir else "/dev/null")
            df = compute_metrics(store, csv, None, f1_tolerances=(0.01, 0.02, 0.05))
            g = df[df.group == "GLOBAL"].iloc[0]
            logs = {
                "heldout/F1@20ms":     float(g["F1@20ms (%)"]),
                "heldout/F1@50ms":     float(g["F1@50ms (%)"]),
                "heldout/MedianBE_ms": float(g["MedianBE (ms)"]),
                "heldout/PER":         float(g["PER (%)"]),
            }
            if "F1@10ms (%)" in g.index:
                logs["heldout/F1@10ms"] = float(g["F1@10ms (%)"])
            logs.update({f"heldout/{k}": v for k, v
                         in duration_fidelity(store, align_sequences).items()})
            w = self.trainer._ensure_tb_writer()
            if w is not None:
                for k, v in logs.items():
                    w.add_scalar(k, v, step)
                w.flush()
            print(f"[eval] step {step}  F1@20={logs['heldout/F1@20ms']:.1f}  "
                  f"slope={logs.get('heldout/dur_slope', float('nan')):.3f}  "
                  f"PER={logs['heldout/PER']:.2f}")
        except Exception as e:
            print(f"[eval] eval failed at step {step}: {e}")


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--audio_dir", type=str, default=None,
                   help="Audiofolder dataset (train/validation/test with metadata.csv). Read "
                        "only to build --preprocessed_path when it does not exist, and for "
                        "the word boundaries of --sil_positions word.")
    p.add_argument("--preprocessed_path", type=str, required=True,
                   help="Preprocessed Arrow dataset (built by train_ctc.py --preprocess_only, "
                        "or here from --audio_dir when missing).")
    p.add_argument("--vocab_path", type=str, required=True,
                   help="Phone vocabulary JSON written with the preprocessed dataset.")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--min_audio_seconds", type=float, default=1.5)
    p.add_argument("--min_phones", type=int, default=5)
    p.add_argument("--num_proc", type=int, default=6)
    # Model
    p.add_argument("--pretrained_model", type=str, required=True,
                   help="wav2vec2 CTC checkpoint to start from (train_ctc.py output).")
    p.add_argument("--align_hidden_size", type=int, default=None,
                   help="Cross-attention dimension. Default: wav2vec2 hidden // 2.")
    p.add_argument("--window_sigma_frames", type=float, default=5.0)
    p.add_argument("--convbank_kernels", type=str, default="1,3,5",
                   help="ConvBank kernel sizes in frames. 1,3,5 at 20 ms; use "
                        "1,5,9 for a 10 ms encoder.")
    p.add_argument("--gt_span_source", type=str, default="forced_align",
                   choices=["forced_align", "diagonal"])
    p.add_argument("--pool_mode", type=str, default="window",
                   choices=["global", "global_pos", "window"])
    p.add_argument("--prior_type", type=str, default="beta_binomial",
                   choices=["beta_binomial", "quadratic"])
    # Loss weights: all three at 1.0
    p.add_argument("--ctc_weight", type=float, default=1.0)
    p.add_argument("--align_weight", type=float, default=1.0)
    p.add_argument("--contrastive_weight", type=float, default=1.0)
    p.add_argument("--align_warmup_steps", type=int, default=1000,
                   help="Linear ramp for L_FS: 0 -> align_weight over this many "
                        "steps. Set 0 for a flat weight from step 0.")
    p.add_argument("--align_start_steps", type=int, default=0,
                   help="L_FS weight is 0 before this step, then ramps over "
                        "--align_warmup_steps.")
    p.add_argument("--contrastive_start_steps", type=int, default=0,
                   help="L_m weight is 0 before this step. 0 = L_m at full weight "
                        "from step 0.")
    p.add_argument("--contrastive_warmup_steps", type=int, default=0,
                   help="Linear ramp for L_m after --contrastive_start_steps. "
                        "0 = full weight at once.")
    p.add_argument("--no_reseed_codebook", action="store_true",
                   help="Do not re-seed the EMA codebook when L_m switches on "
                        "after a delay.")
    # Contrastive branch
    p.add_argument("--n_negatives", type=int, default=50)
    p.add_argument("--contrastive_temperature", type=float, default=0.1)
    p.add_argument("--mask_prob_low", type=float, default=0.1)
    p.add_argument("--mask_prob_high", type=float, default=0.5)
    p.add_argument("--num_codevectors_ema", type=int, default=320)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--no_position_features", action="store_true",
                   help="Ablation: feed zeros instead of the pos/dur features.")
    p.add_argument("--quantizer_assign", choices=["gumbel", "nearest"],
                   default="gumbel",
                   help="gumbel: codes picked by a trained weight_proj + hard "
                        "Gumbel-softmax (L_m can move its own targets). nearest: "
                        "closest codevector, no gradient to the assignment.")
    p.add_argument("--init_quantizer_batches", type=int, default=200,
                   help="Batches of real encoder frames used to initialise the EMA "
                        "codevectors before training. 0 disables (codevectors then "
                        "start from noise). Skipped automatically when resuming.")
    # Alignment target
    p.add_argument("--align_target", choices=["gt", "curriculum"], default="gt",
                   help="gt: L_FS always aligns against ground-truth phones. "
                        "curriculum: ramp GT -> CTC greedy decode between "
                        "--pred_target_start_frac and --pred_target_end_frac.")
    p.add_argument("--pred_target_start_frac", type=float, default=1)
    p.add_argument("--pred_target_end_frac", type=float, default=1)
    # Prior
    p.add_argument("--prior_scaling_factor", type=float, default=1.0)
    p.add_argument("--prior_schedule", choices=["constant", "anneal"],
                   default="constant",
                   help="constant: prior_weight=1 for the whole run. anneal: fade "
                        "1 -> 0 between --prior_anneal_start_frac and "
                        "--prior_anneal_end_frac.")
    p.add_argument("--prior_anneal_start_frac", type=float, default=1)
    p.add_argument("--prior_anneal_end_frac", type=float, default=1)
    # Optim
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--alignment_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Training
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=7)
    p.add_argument("--stop_after_steps", type=int, default=None,
                   help="Stop (and save) at this global step. Unlike max_steps it "
                        "leaves the length of the run, and so every schedule "
                        "expressed as a fraction of it, unchanged.")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="stop after this many steps (overrides --num_train_epochs); "
                        "fraction-based schedules (prior anneal) scale to it")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=-1)
    p.add_argument("--optimizer_last_only", action="store_true",
                   help="Keep optimizer.pt only in the newest checkpoint (~2.5 GB each).")
    p.add_argument("--fp16", action="store_true")
    # Gradient routing: both branches attached by default.
    p.add_argument("--detach_align_from_encoder", action="store_true",
                   help="Legacy routing (no --encoder_alpha): block L_FS from reaching "
                        "the wav2vec2 encoder.")
    p.add_argument("--detach_contrastive_from_encoder", action="store_true",
                   help="Legacy routing (no --encoder_alpha): block L_m from reaching "
                        "the wav2vec2 encoder.")
    # Joint routing: L_FS / L_m into the encoder through the queries only.
    p.add_argument("--encoder_alpha", type=float, default=None,
                   help="Joint routing. Keys always detached, L_m never reaches "
                        "the encoder, L_FS reaches it through the queries with its "
                        "gradient scaled by this value (0 = never). Default None: "
                        "legacy routing set by the two --detach_* flags.")
    p.add_argument("--encoder_alpha_start_steps", type=int, default=0,
                   help="alpha is 0 until this step ...")
    p.add_argument("--encoder_alpha_ramp_steps", type=int, default=0,
                   help="... then ramps linearly to --encoder_alpha over this many steps.")
    p.add_argument("--lm_encoder_alpha", type=float, default=None,
                   help="Gradient scale of L_m into the encoder (through its "
                        "attention/queries only; targets stay detached). Same schedule as "
                        "--encoder_alpha. Default None = 0: L_m never reaches the encoder.")
    p.add_argument("--encoder_top_layers", type=int, default=0,
                   help="L_FS / L_m reach only the last k transformer layers "
                        "(+ final layer norm). CTC still trains all layers. 0 = all layers.")
    p.add_argument("--encoder_alpha_stop_steps", type=int, default=None,
                   help="Optional: both alphas start decreasing to 0 at this step ...")
    p.add_argument("--encoder_alpha_stop_ramp_steps", type=int, default=0,
                   help="... over this many steps (pulse schedule). Default: never stop.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for model init (new heads), data order and dropout. "
                        "Default None = previous behaviour (Trainer seed 42, unseeded init).")
    p.add_argument("--layerdrop", type=float, default=0.0)
    p.add_argument("--spec_augment", action="store_true")
    # Visualization / external eval
    p.add_argument("--viz_every_n_steps", type=int, default=500)
    p.add_argument("--viz_max_samples", type=int, default=2)
    p.add_argument("--eval_manifest", type=str, default=None,
                   help="Evaluation manifest (see jointrecalign/evaluation/attention_eval.py): "
                        "alignment metrics logged at every --eval_manifest_every_n_evals-th eval.")
    p.add_argument("--eval_manifest_every_n_evals", type=int, default=1)
    # Latent optional SIL. All off by default.
    p.add_argument("--sil", action="store_true",
                   help="Optional SIL states in L_FS, L_m and decoding (replaces the "
                        "RAD-TTS constant blank once active).")
    p.add_argument("--sil_positions", choices=["word", "edges", "all"], default="word",
                   help="Training SIL slots: word = word boundaries + utterance edges.")
    p.add_argument("--sil_eval_positions", choices=["all", "edges"], default="all",
                   help="Decoding SIL slots (the CTC greedy target has no words).")
    p.add_argument("--sil_logp", type=float, default=-2.0,
                   help="Log-cost of entering a SIL slot (insertion penalty).")
    p.add_argument("--sil_logp_learned", action="store_true",
                   help="Learn the entry cost (initialised from --sil_logp).")
    p.add_argument("--sil_min_frames", type=int, default=1,
                   help="Minimum frames of a used SIL (tied sub-state chain).")
    p.add_argument("--sil_init", choices=["random", "lowenergy"], default="random",
                   help="lowenergy: sil_key from the mean query of the 5%% quietest "
                        "frames of the batch at --sil_start_steps.")
    p.add_argument("--sil_start_steps", type=int, default=0,
                   help="SIL is used from this step on (before: reference L_FS/L_m).")
    p.add_argument("--lm_att_source", choices=["softmax", "fs_posterior"], default="softmax",
                   help="L_m attention with --sil: per-frame softmax over [phones; SIL] "
                        "or the forward-backward occupancy of phones and SIL slots.")
    p.add_argument("--lm_mask_same_code", action="store_true",
                   help="Mask L_m negatives that share the positive's code.")
    p.add_argument("--lm_energy_weight", type=float, default=0.0,
                   help="L_energy: predict each frame's waveform log-energy from the "
                        "alignment content [YA; pos/dur] (smooth L1, all real frames), "
                        "added to L_m with this weight. 0 = off.")
    p.add_argument("--lm_energy_floor_db", type=float, default=-60.0,
                   help="L_energy target floor, dB re the utterance's 95th percentile.")
    p.add_argument("--sil_metadata_csv", type=str, default=None,
                   help="metadata.csv with phonemes_single (word boundaries) for the "
                        "train split. Default: <audio_dir>/train/metadata.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump({"argv": sys.argv, "args": vars(args)},
              open(os.path.join(args.out_dir, "run_args.json"), "w"), indent=2)

    print(f"[train] losses: ctc={args.ctc_weight} align={args.align_weight} "
          f"contrastive={args.contrastive_weight}")
    print(f"[train] alignment target: {args.align_target}")
    print(f"[train] prior schedule: {args.prior_schedule}")
    print(f"[train] pretrained_model: {args.pretrained_model}")
    print(f"[train] schedule: L_FS 0 until step {args.align_start_steps}, full by step "
          f"{args.align_start_steps + args.align_warmup_steps}; L_m 0 until step "
          f"{args.contrastive_start_steps}, full by step "
          f"{args.contrastive_start_steps + args.contrastive_warmup_steps}")

    # 1) data
    dataset = build_or_load_dataset(args)
    train_ds = dataset["train"]
    if args.sil and args.sil_positions == "word":
        if not (args.sil_metadata_csv or args.audio_dir):
            raise SystemExit("--sil_positions word needs --sil_metadata_csv or --audio_dir")
        meta = args.sil_metadata_csv or os.path.join(args.audio_dir, "train", "metadata.csv")
        train_ds = add_word_slots(train_ds, meta, args.vocab_path)
    eval_ds = dataset.get(args.eval_split, None)
    if eval_ds is not None and args.max_eval_samples > 0 and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=0).select(range(args.max_eval_samples))
    if eval_ds is None:
        print(f"[train] no '{args.eval_split}' split found; PER evaluation disabled.")
    else:
        print(f"[train] PER evaluation on {len(eval_ds)} utterances every "
              f"{args.eval_steps} steps.")

    # 2) processor
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=args.vocab_path,
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=args.sampling_rate,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    # 3) config
    config = Wav2Vec2Config.from_pretrained(args.pretrained_model)
    config.vocab_size = vocab_size
    config.pad_token_id = pad_id
    config.ctc_loss_reduction = "mean"
    config.ctc_zero_infinity = True
    config.prior_scaling_factor = args.prior_scaling_factor
    config.layerdrop = args.layerdrop
    if args.layerdrop > 0:
        print(f"[train] WARNING: layerdrop={args.layerdrop} > 0.")

    # Pooling configuration.
    config.pool_mode = args.pool_mode
    config.prior_type = args.prior_type
    config.window_sigma_frames = args.window_sigma_frames
    config.convbank_kernels = [int(k) for k in args.convbank_kernels.split(",")]
    ckpt_sr = int(getattr(config, "input_sampling_rate", 16000) or 16000)
    if ckpt_sr != args.sampling_rate:
        raise SystemExit(f"--sampling_rate {args.sampling_rate} but the checkpoint was "
                         f"trained at {ckpt_sr} Hz")
    config.input_sampling_rate = args.sampling_rate
    fr_ms = config.inputs_to_logits_ratio / args.sampling_rate * 1000
    print(f"[train] encoder frame = {fr_ms:.0f} ms (conv_stride {config.conv_stride}); "
          f"sigma={args.window_sigma_frames} frames, convbank={config.convbank_kernels}, "
          f"prior_scaling_factor={args.prior_scaling_factor}")
    config.gt_span_source = args.gt_span_source
    config.align_hidden_size = (
        args.align_hidden_size
        if args.align_hidden_size is not None
        else config.hidden_size // 2
    )

    # Contrastive branch configuration.
    config.n_negatives = args.n_negatives
    config.contrastive_temperature = args.contrastive_temperature
    config.mask_prob_low = args.mask_prob_low
    config.mask_prob_high = args.mask_prob_high
    config.num_codevectors_ema = args.num_codevectors_ema
    config.ema_decay = args.ema_decay
    config.use_position_features = not args.no_position_features
    config.quantizer_assign = args.quantizer_assign
    print(f"[train] EMA quantizer assignment: {args.quantizer_assign}")
    config.lm_mask_same_code = args.lm_mask_same_code
    config.lm_energy_weight = args.lm_energy_weight
    config.lm_energy_floor_db = args.lm_energy_floor_db
    if args.lm_energy_weight > 0:
        print(f"[train] L_energy: weight {args.lm_energy_weight}, floor "
              f"{args.lm_energy_floor_db} dB re p95, from [YA; pos/dur]")
    if args.lm_mask_same_code:
        print("[train] L_m: negatives with the positive's code are masked")

    # Latent SIL configuration (recorded in config.json).
    config.use_sil = bool(args.sil)
    if args.sil:
        config.sil_positions = args.sil_positions
        config.sil_eval_positions = args.sil_eval_positions
        config.sil_logp = args.sil_logp
        config.sil_logp_learned = args.sil_logp_learned
        config.sil_min_frames = args.sil_min_frames
        config.lm_att_source = args.lm_att_source
        print(f"[train] latent SIL from step {args.sil_start_steps}: positions={args.sil_positions} "
              f"(eval {args.sil_eval_positions}), entry logp={args.sil_logp}"
              f"{' (learned)' if args.sil_logp_learned else ''}, min frames={args.sil_min_frames}, "
              f"init={args.sil_init}, L_m attention={args.lm_att_source}")

    # Gradient routing: both branches attached unless explicitly detached.
    config.detach_alignment_input = args.detach_align_from_encoder
    config.detach_contrastive_input = args.detach_contrastive_from_encoder
    if args.encoder_alpha is not None:
        if args.detach_align_from_encoder:
            print("[train] WARNING: --detach_align_from_encoder forces encoder_alpha "
                  "to 0 (L_FS never reaches the encoder).")
        config.detach_contrastive_input = True        # recorded in config.json
        config.joint_encoder_alpha = args.encoder_alpha
        config.joint_lm_encoder_alpha = args.lm_encoder_alpha
        config.encoder_top_layers = args.encoder_top_layers
        if args.encoder_top_layers > 0 and args.layerdrop > 0:
            raise SystemExit("--encoder_top_layers needs --layerdrop 0")
        where = (f"the last {args.encoder_top_layers} transformer layers"
                 if args.encoder_top_layers > 0 else "all transformer layers")
        stop = ("" if args.encoder_alpha_stop_steps is None else
                f"; back to 0 from step {args.encoder_alpha_stop_steps} over "
                f"{args.encoder_alpha_stop_ramp_steps} steps")
        lm_txt = (f"L_m -> {where} via its attention, grad x {args.lm_encoder_alpha}"
                  if args.lm_encoder_alpha else
                  "L_m stops at the alignment/contrastive modules")
        print(f"[train] JOINT routing: L_CTC -> all encoder layers; L_FS -> {where} via "
              f"queries only, grad x {args.encoder_alpha}; {lm_txt}. Alpha schedule: 0 until "
              f"step {args.encoder_alpha_start_steps}, full by step "
              f"{args.encoder_alpha_start_steps + args.encoder_alpha_ramp_steps}{stop}. "
              f"Keys and L_m targets detached.")
    else:
        print("[train] L_FS gradients "
              + ("stop at the alignment branch."
                 if config.detach_alignment_input
                 else "reach the wav2vec2 encoder (attached)."))
        print("[train] L_m gradients "
              + ("stop at the contrastive branch."
                 if config.detach_contrastive_input
                 else "reach the wav2vec2 encoder (attached)."))
    if args.encoder_alpha is None and not (config.detach_alignment_input
                                           and config.detach_contrastive_input):
        print(f"[train] NOTE: the encoder now receives three gradient paths at "
              f"backbone_lr={args.backbone_lr}. Watch ctc_loss for the first "
              f"~1000 steps; if it rises, lower backbone_lr rather than the "
              f"loss weights.")

    if not args.spec_augment:
        config.apply_spec_augment = False
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        print("[train] SpecAugment disabled (collator + config).")
    else:
        config.apply_spec_augment = True
        print("[train] SpecAugment enabled: mask_prob ~ U[0.05, 0.40], mask_length=2.")

    # 4) model
    if args.seed is not None:
        from transformers import set_seed
        set_seed(args.seed)
        print(f"[train] seed = {args.seed} (model init, data order, dropout)")
    model = Wav2Vec2ForCTCPoolContrastive.from_pretrained(
        args.pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_extractor()
    if args.sil:
        if args.encoder_alpha is not None and (args.lm_encoder_alpha or 0.0) != args.encoder_alpha:
            raise SystemExit("--sil needs --lm_encoder_alpha == --encoder_alpha")
        model.reset_sil_parameters()

    # 5) collators
    train_collator = DataCollatorCTCAlignment(
        processor=processor,
        feat_extract_output_lengths_fn=model._get_feat_extract_output_lengths,
        apply_spec_augment=args.spec_augment,
    )
    eval_collator = DataCollatorCTCAlignment(
        processor=processor,
        feat_extract_output_lengths_fn=model._get_feat_extract_output_lengths,
        apply_spec_augment=False,
    )

    # 6) training args
    ta_kwargs = dict(
        output_dir=args.out_dir,
        group_by_length=True,
        length_column_name="length",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=args.fp16,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        learning_rate=args.alignment_lr,
        weight_decay=args.weight_decay,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.out_dir, "runs"),
        remove_unused_columns=False,
        label_names=["labels"],
    )
    if args.seed is not None:
        ta_kwargs["seed"] = args.seed
        ta_kwargs["data_seed"] = args.seed
    if eval_ds is not None:
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
        ta_kwargs[eval_key] = "steps"
        ta_kwargs["eval_steps"] = args.eval_steps
        ta_kwargs["eval_accumulation_steps"] = 8
        ta_kwargs["metric_for_best_model"] = "per"
        ta_kwargs["greater_is_better"] = False
    training_args = TrainingArguments(**ta_kwargs)

    trainer = CTCAlignmentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=train_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=make_compute_metrics(pad_id) if eval_ds is not None else None,
        backbone_lr=args.backbone_lr,
        alignment_lr=args.alignment_lr,
        weight_decay=args.weight_decay,
        ctc_weight=args.ctc_weight,
        align_weight=args.align_weight,
        contrastive_weight=args.contrastive_weight,
        align_warmup_steps=args.align_warmup_steps,
        align_start_steps=args.align_start_steps,
        contrastive_start_steps=args.contrastive_start_steps,
        contrastive_warmup_steps=args.contrastive_warmup_steps,
        reseed_codebook_at_lm_start=not args.no_reseed_codebook,
        encoder_alpha=args.encoder_alpha,
        encoder_alpha_start_steps=args.encoder_alpha_start_steps,
        encoder_alpha_ramp_steps=args.encoder_alpha_ramp_steps,
        lm_encoder_alpha=args.lm_encoder_alpha,
        encoder_alpha_stop_steps=args.encoder_alpha_stop_steps,
        encoder_alpha_stop_ramp_steps=args.encoder_alpha_stop_ramp_steps,
        gt_targets_only=(args.align_target == "gt"),
        prior_constant=(args.prior_schedule == "constant"),
        pred_target_start_frac=args.pred_target_start_frac,
        pred_target_end_frac=args.pred_target_end_frac,
        prior_anneal_start_frac=args.prior_anneal_start_frac,
        prior_anneal_end_frac=args.prior_anneal_end_frac,
        viz_every_n_steps=args.viz_every_n_steps,
        viz_max_samples=args.viz_max_samples,
        tokenizer_for_viz=processor.tokenizer,
        processor_for_save=processor,
        probe_dataset=eval_ds if eval_ds is not None else train_ds,
        probe_collator=eval_collator,
        use_sil=args.sil,
        sil_start_steps=args.sil_start_steps,
        sil_init=args.sil_init,
    )

    if eval_ds is not None:
        _orig_get_eval_dataloader = trainer.get_eval_dataloader

        def _get_eval_dataloader(eval_dataset=None, _orig=_orig_get_eval_dataloader,
                                 _tr=trainer, _c=eval_collator):
            saved = _tr.data_collator
            _tr.data_collator = _c
            try:
                return _orig(eval_dataset)
            finally:
                _tr.data_collator = saved

        trainer.get_eval_dataloader = _get_eval_dataloader

    if args.eval_manifest:
        os.makedirs(os.path.join(args.out_dir, "heldout"), exist_ok=True)
        trainer.add_callback(EvalManifestCallback(
            manifest_path=args.eval_manifest,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            trainer=trainer,
            every_n_evals=args.eval_manifest_every_n_evals,
            csv_dir=os.path.join(args.out_dir, "heldout"),
        ))
        print(f"[train] manifest alignment eval every "
              f"{args.eval_manifest_every_n_evals} eval(s).")

    if args.optimizer_last_only:
        trainer.add_callback(OptimizerLastOnlyCallback())
        print("[train] optimizer.pt kept in the newest checkpoint only")
    if args.stop_after_steps:
        trainer.add_callback(StopAfterStepsCallback(args.stop_after_steps))
        print(f"[train] stopping at step {args.stop_after_steps}")

    # Auto-resume from the newest checkpoint in out_dir.
    last_ckpt = None
    if os.path.isdir(args.out_dir):
        last_ckpt = get_last_checkpoint(args.out_dir)
    if last_ckpt is not None:
        print(f"[train] resuming from {last_ckpt}")
    else:
        print("[train] starting from scratch (no checkpoint in out_dir)")

    # EMA codevectors from real frames. Only on a fresh run: a resumed
    # checkpoint already carries trained codevectors, and re-seeding them from
    # noise-free frames would discard that history.
    if last_ckpt is None and args.init_quantizer_batches > 0:
        model.to(training_args.device)
        initialize_quantizer(
            model, trainer.get_train_dataloader(), training_args.device,
            n_batches=args.init_quantizer_batches,
        )
    elif last_ckpt is None:
        print("[train] WARNING: quantizer init skipped; EMA codevectors start "
              "from noise and L_m will be uninformative early on.")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(os.path.join(args.out_dir, "final"))
    print("[train] done.")


if __name__ == "__main__":
    main()