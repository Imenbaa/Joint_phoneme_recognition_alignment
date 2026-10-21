#!/usr/bin/env python3
"""Score a checkpoint on an evaluation manifest: DTW on the cross-attention energy.

The model aligns its own CTC greedy decode (no reference phones, no prior),
chunk by chunk: no padding (the model sees exactly [start, end]), onset-frame
extraction with contiguous ends, no silence trimming. ctcfa_eval.py uses the
same conventions, so the two are directly comparable.

Manifest (JSON):
    {"files": {
        "<file_id>": {
            "wav": "/path/to/audio.wav",                      # any rate, mono or not
            "chunks": [{"start": 0.0, "end": 12.3}, ...],     # seconds, scored segments
            "ref_intervals": [{"phoneme": "a", "start": 0.12, "end": 0.20}, ...],
            "style": "read",                                  # optional grouping key
            "pause_intervals": [[s, e], ...],                 # optional, SIL scoring
            "excluded_intervals": [[s, e], ...]               # optional, SIL scoring
        }, ...}}

Frame timing is read from the model config (frame stride and receptive field),
so the same code scores 20 ms and 10 ms models.
"""
import numpy as np, torch, soundfile as sf, librosa, unicodedata
from itertools import groupby
from librosa.sequence import dtw

SR = 16000
FRAME_STRIDE_S = 0.02          # wav2vec2: 320 samples @ 16 kHz (default only)


def model_sr(model):
    """Sampling rate the model was trained on (config.input_sampling_rate,
    16 kHz when absent)."""
    return int(getattr(model.config, "input_sampling_rate", SR) or SR)


def frame_stride_s(model):
    """Seconds per encoder frame: conv hop (samples) / input rate. 0.02 for the
    standard model; 0.01 with last conv stride 1 or 32 kHz input.
    Frame i starts at i * stride (checked: 10 ms frame 2i == 20 ms frame i)."""
    return getattr(model.config, "inputs_to_logits_ratio", 320) / model_sr(model)


def frame_time_offset_s(model):
    """Fixed, architecture-only correction from frame index to boundary time.

    Frame i of the conv front-end sees samples [i*hop, i*hop + RF) (no padding),
    so its centre is at i*hop + RF/2. A label change at frame i is placed halfway
    between the centres of frames i-1 and i:
        boundary = i*hop + (RF - hop) / 2
    RF and hop come from config.conv_kernel / conv_stride. Nothing is tuned on data:
    +2.5 ms for the 20 ms model, +7.5 ms for 10 ms (last stride 1), +1.25 ms at 32 kHz.
    """
    rf, hop = 1, 1
    for k, s in zip(model.config.conv_kernel, model.config.conv_stride):
        rf += (k - 1) * hop
        hop *= s
    return (rf - hop) / 2 / model_sr(model)


def to_model_rate(seg, model):
    """16 kHz chunk -> the model's input rate. Audio is always loaded at 16 kHz
    first, so a 32 kHz model sees the same 8 kHz band limit as its (upsampled
    16 kHz) training data."""
    sr = model_sr(model)
    if sr == SR:
        return seg
    return librosa.resample(seg, orig_sr=SR, target_sr=sr).astype(np.float32)


# Hypothesis tokens rewritten before scoring (affricates of the training phone
# set that the reference inventory writes as one fricative).
HYP_FIX = {"ts": "s", "tʃ": "ʃ"}
SKIP_TOK = ("", " ", "[", "|", "[PAD]", "[UNK]", "[SIL]")


def load_audio(path):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = np.mean(a, axis=1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


def dtw_path(att):
    """[T, P] -> [T] phone index per frame.

    Steps [1,1] and [1,0]: no skips, every frame consumed once, every phone
    gets at least one frame.
    """
    T, P = att.shape
    if P == 0 or T < P:
        return None
    if P == 1:
        return np.zeros(T, dtype=np.int32)
    _, wp = dtw(C=-att, step_sizes_sigma=np.array([[1, 1], [1, 0]]))
    path = np.full(T, -1, dtype=np.int32)
    for fr, ph in wp:                      # librosa returns the path reversed
        if 0 <= fr < T and path[fr] < ph:
            path[fr] = ph
    last = 0
    for t in range(T):
        if path[t] < 0:
            path[t] = last
        else:
            last = path[t]
    return path


@torch.inference_mode()
def run_manifest(model, feature_extractor, tokenizer, manifest,
                 device="cuda", max_files=None):
    """-> {file_id: {ref_intervals, hyp_intervals, ref_seq, hyp_seq, style}}"""
    was_training = model.training
    model.eval()
    store = {}
    stride = frame_stride_s(model)
    offset = frame_time_offset_s(model)   # receptive-field centre rule
    try:
        for nm, meta in list(manifest["files"].items())[:max_files]:
            wav = torch.from_numpy(load_audio(meta["wav"])).float()
            hyp = []
            for ch in meta["chunks"]:
                s0, s1 = int(ch["start"] * SR), int(ch["end"] * SR)
                seg = wav[s0:s1]
                if seg.numel() < 400:      # under one conv receptive field
                    continue
                # the extractor only normalises; pass its own rate so a 16 kHz
                # extractor also serves a 32 kHz model
                x = feature_extractor(to_model_rate(seg.numpy(), model),
                                      sampling_rate=feature_extractor.sampling_rate,
                                      return_tensors="pt", return_attention_mask=True)
                o = model(input_values=x.input_values.to(device),
                          attention_mask=x.attention_mask.to(device),
                          labels=None,          # target = CTC greedy decode
                          prior_weight=0.0,     # no beta-binomial prior
                          ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
                T, P = int(o.input_lengths[0]), int(o.text_lengths[0])
                if P == 0:
                    continue
                # log_softmax then exp: softmax-then-log clips the losers to a
                # constant floor and destroys their ordering.
                att = np.exp(o.energy[0, :T, :P].float().log_softmax(-1).cpu().numpy())
                path = dtw_path(att)
                if path is None:
                    continue
                tokens = tokenizer.convert_ids_to_tokens(
                    o.predicted_phone_ids[0, :P].cpu().tolist())

                # run-length encode; keep each phone's ONSET frame
                spans, n = [], 0
                for idx, grp in groupby(path.tolist()):
                    spans.append((idx, n))
                    n += sum(1 for _ in grp)

                ca = []
                for pi, sframe in spans:
                    ph = tokens[pi]
                    if ph in SKIP_TOK:
                        continue
                    if len(ph) == 1 and unicodedata.combining(ph):
                        if ca:
                            ca[-1]["phoneme"] += ph
                        continue
                    ca.append({"phoneme": HYP_FIX.get(ph, ph),
                               "start": s0 / SR + sframe * stride + offset,
                               "end": None})
                # contiguous ends: end_i = start_{i+1}; last = chunk end
                for i in range(len(ca) - 1):
                    ca[i]["end"] = ca[i + 1]["start"]
                if ca:
                    ca[-1]["end"] = s0 / SR + n * stride + offset
                hyp.extend(ca)

            hyp.sort(key=lambda x: x["start"])
            store[nm] = {"ref_intervals": meta["ref_intervals"],
                         "hyp_intervals": hyp,
                         "ref_seq": [x["phoneme"] for x in meta["ref_intervals"]],
                         "hyp_seq": [x["phoneme"] for x in hyp],
                         "style": meta.get("style", "ALL")}
    finally:
        model.train(was_training)
    return store


def duration_fidelity(store, align_sequences, gate=0.5):
    """Slope of hypothesis duration on reference duration, over matched phones.
    1.0 = faithful; 0.0 = every phone gets the same length whatever it is.
    The gate discards pairs the edit-distance matched across seconds of audio.
    """
    R, H = [], []
    for d in store.values():
        for i, j in align_sequences(d["ref_seq"], d["hyp_seq"]):
            if i is None or j is None or d["ref_seq"][i] != d["hyp_seq"][j]:
                continue
            r, h = d["ref_intervals"][i], d["hyp_intervals"][j]
            if abs(h["start"] - r["start"]) > gate or abs(h["end"] - r["end"]) > gate:
                continue
            R.append((r["end"] - r["start"]) * 1000)
            H.append((h["end"] - h["start"]) * 1000)
    if len(R) < 100:
        return {}
    R, H = np.array(R), np.array(H)
    return {"dur_slope": float(np.polyfit(R, H, 1)[0]),
            "dur_corr": float(np.corrcoef(R, H)[0, 1])}