#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scoring of latent-SIL checkpoints on an evaluation manifest (nothing is fitted).

One model pass per chunk, as attention_eval.run_manifest (CTC greedy target,
prior 0, eval mode), then two decodes of the same energies:

  sil   Viterbi over the extended topology [SIL] p1 [SIL] p2 ... [SIL]
        (positions = config.sil_eval_positions, default "all": the greedy
        target has no word boundaries), learned key / entry cost, min frames
        as trained. A phone ends where the following SIL starts; SIL frames
        belong to no phone.
  dtw   the plain decode (attention_eval.dtw_path on the phone softmax, onsets,
        contiguous ends), for comparison on the same pass.

Boundary time of frame f in a chunk starting at cs: cs + f * HOP + offset, with
HOP and offset read from the model (attention_eval.frame_stride_s,
frame_time_offset_s): 20 ms / +2.5 ms, 10 ms / +7.5 ms.

Metrics: F1@10/20/50, MedianBE, P90, PER (metrics.compute_metrics), duration
slope/corr (all phones, and in-speech phones only), and, when the manifest has
pause_intervals, SIL against the reference pauses over each whole file (a pause
cut by a chunk edge counts once): frame precision/recall/F1 (frame centres,
excluded_intervals ignored) and pause-interval P/R/F1 at IoU >= 0.5.
"""
import itertools
import os
import tempfile
import unicodedata

import numpy as np
import torch

from .. import sil_fs
from . import attention_eval as E
from .metrics import compute_metrics, align_sequences

# Set from the model by _set_frame (defaults: 20 ms model). CENTRE = centre of
# frame 0's receptive field, used for frame-level SIL scoring.
HOP = 0.02
CENTRE = 0.0125


def _set_frame(model):
    global HOP, CENTRE
    m = getattr(model, "module", model)
    HOP = E.frame_stride_s(m)
    CENTRE = E.frame_time_offset_s(m) + HOP / 2      # (RF - hop) / 2 + hop / 2 = RF / 2
    return E.frame_time_offset_s(m)


PAUSE_MIN = 0.02      # a reference pause shorter than 20 ms is ignored
GATE = 0.5            # same gate as attention_eval.duration_fidelity


def _gap_has_pause(pauses, a, b):
    """True if the gap (a, b) between two consecutive reference phones holds
    >= PAUSE_MIN of reference pause (pauses: [(start, end)]; None = every gap
    >= PAUSE_MIN counts as a pause)."""
    if b - a < PAUSE_MIN:
        return False
    if pauses is None:
        return True
    out = []
    for s, e in pauses:
        s2, e2 = max(s, a), min(e, b)
        if e2 - s2 <= 1e-6:
            continue
        if out and s2 - out[-1][1] < 1e-6:
            out[-1] = (out[-1][0], e2)
        else:
            out.append((s2, e2))
    return any(e - s >= PAUSE_MIN for s, e in out)


def duration_fidelity_inspeech(store, pauses_by_file=None, gate=GATE):
    """attention_eval.duration_fidelity restricted to phones with no pause on
    either side."""
    R, Hd = [], []
    for nm, d in store.items():
        refs = d["ref_intervals"]
        pauses = None if pauses_by_file is None else pauses_by_file.get(nm, [])
        N = len(refs)
        for i, j in align_sequences(d["ref_seq"], d["hyp_seq"]):
            if i is None or j is None or d["ref_seq"][i] != d["hyp_seq"][j]:
                continue
            if i == 0 or i == N - 1:
                continue
            r, h = refs[i], d["hyp_intervals"][j]
            if abs(h["start"] - r["start"]) > gate or abs(h["end"] - r["end"]) > gate:
                continue
            if (_gap_has_pause(pauses, refs[i - 1]["end"], r["start"])
                    or _gap_has_pause(pauses, r["end"], refs[i + 1]["start"])):
                continue
            R.append((r["end"] - r["start"]) * 1000)
            Hd.append((h["end"] - h["start"]) * 1000)
    if len(R) < 100:
        return {}
    R, Hd = np.array(R), np.array(Hd)
    return {"dur_slope_sp": float(np.polyfit(R, Hd, 1)[0]), "dur_corr_sp": float(np.corrcoef(R, Hd)[0, 1])}


def _segments(states, tokens, cs, offset, sil_frames):
    """Per-frame states -> hypothesis phone intervals. sil_frames collects
    (start, end) of every SIL run (seconds)."""
    ca = []
    for s, grp in itertools.groupby(enumerate(states), key=lambda x: x[1]):
        grp = list(grp)
        f0, f1 = grp[0][0], grp[-1][0] + 1
        t0, t1 = cs + f0 * HOP + offset, cs + f1 * HOP + offset
        if s is None:
            sil_frames.append((t0, t1))
            continue
        ph = tokens[s]
        if ph in E.SKIP_TOK:
            continue
        if len(ph) == 1 and unicodedata.combining(ph):
            if ca:
                ca[-1]["phoneme"] += ph
                ca[-1]["end"] = t1
            continue
        ca.append({"phoneme": E.HYP_FIX.get(ph, ph), "start": t0, "end": t1})
    return ca


@torch.inference_mode()
def run_manifest_sil(model, feature_extractor, tokenizer, manifest, device="cuda",
                     max_files=None, positions=None, offset=None, with_dtw=True):
    """-> (store_sil, store_dtw or None, sil_runs {file: [(s, e)]}, chunks {file: [(s, e)]})
    offset None = the model's frame_time_offset_s."""
    model_offset = _set_frame(model)
    offset = model_offset if offset is None else offset
    m = getattr(model, "module", model)
    positions = positions or getattr(m, "sil_eval_positions", "all")
    k = int(getattr(m, "sil_min_frames", 1))
    sil_logp = float(torch.as_tensor(m.sil_logp).detach())   # a learned parameter in most checkpoints
    was_training = model.training
    model.eval()
    store, store_dtw, sil_runs, chunks = {}, {}, {}, {}
    try:
        for nm, meta in list(manifest["files"].items())[:max_files]:
            wav = torch.from_numpy(E.load_audio(meta["wav"])).float()
            hyp, hyp_dtw, runs = [], [], []
            chunks[nm] = [(float(c["start"]), float(c["end"])) for c in meta["chunks"]]
            for ch in meta["chunks"]:
                s0, s1 = int(ch["start"] * E.SR), int(ch["end"] * E.SR)
                seg = wav[s0:s1]
                if seg.numel() < 400:
                    continue
                x = feature_extractor(E.to_model_rate(seg.numpy(), m),
                                      sampling_rate=feature_extractor.sampling_rate,
                                      return_tensors="pt", return_attention_mask=True)
                o = model(input_values=x.input_values.to(device),
                          attention_mask=x.attention_mask.to(device), labels=None,
                          prior_weight=0.0, ctc_weight=0.0, align_weight=0.0,
                          contrastive_weight=0.0)
                T, P = int(o.input_lengths[0]), int(o.text_lengths[0])
                if P == 0:
                    continue
                cs = s0 / E.SR
                tokens = tokenizer.convert_ids_to_tokens(o.predicted_phone_ids[0, :P].cpu().tolist())
                en = o.energy[0, :T, :P].float()
                sil = o.sil_energy_raw[0, :T].float()
                logp_ext = torch.log_softmax(torch.cat([en, sil[:, None]], -1), -1).cpu().numpy()
                top, path = sil_fs.viterbi(logp_ext, P, sil_fs.slot_mask(P, positions), k, sil_logp)
                if path is not None:
                    st = [None if top["col"][s] < 0 else int(top["col"][s]) for s in path]
                    hyp.extend(_segments(st, tokens, cs, offset, runs))
                if with_dtw:
                    att = np.exp(en.log_softmax(-1).cpu().numpy())
                    p2 = E.dtw_path(att)
                    if p2 is not None:
                        ca = _segments([int(v) for v in p2], tokens, cs, offset, [])
                        for i in range(len(ca) - 1):           # contiguous ends, as attention_eval
                            ca[i]["end"] = ca[i + 1]["start"]
                        if ca:
                            ca[-1]["end"] = cs + T * HOP + offset
                        hyp_dtw.extend(ca)
            for st_, h in ((store, hyp), (store_dtw, hyp_dtw)):
                h = sorted(h, key=lambda z: z["start"])
                st_[nm] = {"ref_intervals": meta["ref_intervals"], "hyp_intervals": h,
                           "ref_seq": [z["phoneme"] for z in meta["ref_intervals"]],
                           "hyp_seq": [z["phoneme"] for z in h], "style": meta.get("style", "ALL")}
            sil_runs[nm] = runs
    finally:
        model.train(was_training)
    return store, (store_dtw if with_dtw else None), sil_runs, chunks


def _score(store, tag=""):
    tmp = tempfile.mkdtemp()
    df = compute_metrics(store, os.path.join(tmp, "m.csv"), None, f1_tolerances=(0.01, 0.02, 0.05))
    g = df[df.group == "GLOBAL"].iloc[0]

    def col(*keys):
        for c in df.columns:
            if all(k.lower() in str(c).lower() for k in keys):
                return float(g[c])
        return float("nan")
    return {f"F1@10{tag}": col("F1@10"), f"F1@20{tag}": col("F1@20"), f"F1@50{tag}": col("F1@50"),
            f"MedianBE{tag}": col("median", "BE"), f"P90{tag}": col("P90"), f"PER{tag}": col("PER")}


def _sil_timeline(phones, end):
    """Whole-file SIL runs: the time in [0, end] not covered by a hypothesis phone.
    SIL-Viterbi gives every decoded frame to a phone or to SIL, so this is the
    decoded SIL with a pause cut by a chunk edge joined back into one run, and
    audio left out between chunks counted as SIL."""
    runs, cur = [], 0.0
    for p in sorted(phones, key=lambda z: z["start"]):
        if p["start"] - cur >= HOP / 2:
            runs.append((cur, min(p["start"], end)))
        cur = max(cur, p["end"])
    if end - cur >= HOP / 2:
        runs.append((cur, end))
    return [r for r in runs if r[1] - r[0] >= HOP / 2]


def _frame_counts(a, b, P, X, R):
    """(tp, fp, fn) of SIL frames vs reference pause frames, frame centres in [a, b)."""
    c = np.arange(a, b - HOP / 2, HOP) + CENTRE
    c = c[c < b]
    keep = ~np.any((X[:, :1] <= c) & (c < X[:, 1:]), axis=0) if len(X) else np.ones(len(c), bool)
    ref = np.any((P[:, :1] <= c) & (c < P[:, 1:]), axis=0) if len(P) else np.zeros(len(c), bool)
    hyp = np.any((R[:, :1] <= c) & (c < R[:, 1:]), axis=0) if len(R) else np.zeros(len(c), bool)
    return int((hyp & ref & keep).sum()), int((hyp & ~ref & keep).sum()), int((~hyp & ref & keep).sum())


def _pause_matches(refs, hyps, iou):
    """Reference pauses matched one-to-one (greedy) by a SIL run at IoU >= iou."""
    used, m = set(), 0
    for r in refs:
        best, bj = 0.0, None
        for j, h in enumerate(hyps):
            if j in used:
                continue
            inter = max(0.0, min(r[1], h[1]) - max(r[0], h[0]))
            if inter <= 0:
                continue
            u = max(r[1], h[1]) - min(r[0], h[0])
            if inter / u > best:
                best, bj = inter / u, j
        if bj is not None and best >= iou:
            used.add(bj)
            m += 1
    return m


def sil_scores(manifest, sil_runs, chunks, store=None, iou=0.5, mode="file"):
    """SIL vs reference pauses (manifest pause_intervals).

    mode "file" (default, needs store = the SIL-Viterbi alignment store): one SIL
      timeline per file (_sil_timeline over [0, last chunk end]), frames over the
      whole file, every reference pause.
    mode "chunk" (the scores before Oct 2026, kept to compare with older runs):
      SIL runs per chunk, frames inside chunks only, reference pauses only if
      inside one chunk. Biased low by cuts: a pause crossing a chunk edge leaves
      the reference while the SIL on each side of the edge counts as a false
      positive, so the more chunks, the lower the pause precision."""
    if mode not in ("file", "chunk"):
        raise ValueError(f"mode {mode!r}: 'file' or 'chunk'")
    if mode == "file" and store is None:
        raise ValueError("mode 'file' needs the alignment store (hypothesis phones)")
    tp = fp = fn = 0
    m_tp = n_hyp = n_ref = 0
    for nm, runs in sil_runs.items():
        if not chunks[nm]:
            continue
        meta = manifest["files"][nm]
        P = np.asarray([p[:2] for p in meta.get("pause_intervals", [])], float).reshape(-1, 2)
        X = np.asarray([p[:2] for p in meta.get("excluded_intervals", [])], float).reshape(-1, 2)
        if mode == "file":
            end = chunks[nm][-1][1]
            hyps = _sil_timeline(store[nm]["hyp_intervals"], end)
            spans, refs = [(0.0, end)], list(P)
        else:
            hyps, spans = list(runs), chunks[nm]
            refs = [p for p in P if any(a <= p[0] and p[1] <= b for a, b in spans)]
        R = np.asarray(hyps, float).reshape(-1, 2)
        for a, b in spans:
            t, f_, n = _frame_counts(a, b, P, X, R)
            tp += t; fp += f_; fn += n
        n_ref += len(refs); n_hyp += len(hyps)
        m_tp += _pause_matches(refs, hyps, iou)
    f = lambda p, r: 2 * p * r / (p + r) if p + r > 0 else 0.0
    pf, rf = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    pi, ri = m_tp / max(n_hyp, 1), m_tp / max(n_ref, 1)
    return {"sil_frame_P": 100 * pf, "sil_frame_R": 100 * rf, "sil_frame_F1": 100 * f(pf, rf),
            "sil_pause_P": 100 * pi, "sil_pause_R": 100 * ri, "sil_pause_F1": 100 * f(pi, ri),
            "sil_n_hyp": n_hyp, "sil_n_ref": n_ref}


def evaluate(model, feature_extractor, tokenizer, manifest, device="cuda", max_files=None,
             positions=None, offset=None, with_dtw=True, sil_mode="file"):
    """All metrics as a flat dict (keys suffixed _dtw for the plain DTW decode).
    sil_mode: pause scoring, see sil_scores."""
    store, store_dtw, runs, chunks = run_manifest_sil(
        model, feature_extractor, tokenizer, manifest, device, max_files, positions, offset, with_dtw)
    has_pauses = any("pause_intervals" in m for m in manifest["files"].values())
    pauses = ({nm: [tuple(p[:2]) for p in m.get("pause_intervals", [])]
               for nm, m in manifest["files"].items()} if has_pauses else None)
    res = _score(store)
    res.update(E.duration_fidelity(store, align_sequences))
    res.update(duration_fidelity_inspeech(store, pauses))
    if has_pauses:
        res.update(sil_scores(manifest, runs, chunks, store, mode=sil_mode))
    if store_dtw is not None:
        res.update(_score(store_dtw, "_dtw"))
        res.update({f"{k}_dtw": v for k, v in E.duration_fidelity(store_dtw, align_sequences).items()})
    return res, store, store_dtw
