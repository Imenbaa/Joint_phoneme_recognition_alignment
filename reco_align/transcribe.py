"""
Phone transcription of any recording with a latent-SIL checkpoint: phones,
their boundaries, pauses, and the model's top-5 candidates for each phone,
written as a Praat TextGrid (and a CSV with one row per phone).

No transcript is needed: the model aligns its own CTC decode.

  1. The recording is cut into chunks of at most `max_chunk` seconds (10 s by
     default), back to back, so no audio is dropped. Silero VAD finds the
     non-speech; from each position the cut goes to the middle of the latest
     non-speech stretch (>= 0.1 s) within max_chunk, or, if there is none, to
     the lowest Silero speech probability in the second half of the window.
  2. One model pass per chunk: CTC greedy decode (the phones), then SIL-Viterbi
     over [SIL] p1 [SIL] p2 ... [SIL] on the aligner energies (the boundaries
     and pauses), exactly as evaluation/sil_eval.py scores it. A pause cut in
     two by a chunk edge is joined back into one.
  3. Candidates of each phone: the CTC posterior at the phone's peak frame (the
     frame of its greedy CTC run where the decoded symbol is most probable),
     over phone symbols only (blank and special symbols removed, renormalised
     to sum to 1). The probabilities are the model's own, not calibrated: they
     are often close to 1 even when the phone is wrong, but the ORDER of the
     candidates is informative.

TextGrid tiers (interval tiers, empty labels between phones):
  phones      the recognised phone (top-1)
  confidence  probability of the top-1 phone
  top2..top5  the next candidates, "symbol probability"
  pauses      "SIL" on every pause the model detected
"""
import itertools
import unicodedata

import numpy as np
import torch

from . import sil_fs
from .evaluation import attention_eval as E
from .evaluation import sil_eval

K = 5
# model symbols for nasal vowels and œ -> IPA (the vocabulary uses single characters)
MODEL2IPA = {"@": "ɑ̃", "1": "ɛ̃", "§": "ɔ̃", "9": "œ"}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
_VAD = None
VAD_SR = 16000
VAD_WIN = 512                                   # samples per Silero window at 16 kHz (32 ms)


def _silero():
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    return _VAD


def plan_chunks(speech, dur, max_chunk=10.0, min_gap=0.1, quietest=None):
    """[(start, end)] in seconds covering [0, dur], back to back, each at most
    max_chunk long, given the speech segments [(start, end)] of a VAD.

    From each position the cut goes to the middle of the latest non-speech
    stretch (>= min_gap) within max_chunk (clamped to the window). With none,
    quietest(lo, hi) gives the cut (the least speech-like time in [lo, hi],
    the second half of the window); quietest None -> cut at max_chunk."""
    if dur <= max_chunk:
        return [(0.0, dur)]
    gaps, prev = [], 0.0
    for s, e in sorted(speech):
        if s - prev >= min_gap:
            gaps.append((prev, s))
        prev = max(prev, e)
    if dur - prev >= min_gap:
        gaps.append((prev, dur))
    out, pos = [], 0.0
    while dur - pos > max_chunk:
        hi = pos + max_chunk
        cand = [min(max((gs + ge) / 2, gs, pos + 1e-3), ge, hi) for gs, ge in gaps if ge > pos + 1e-3 and gs < hi]
        cand = [c for c in cand if pos + 1e-3 < c <= hi]
        if cand:
            cut = max(cand)
        else:                                   # no pause within max_chunk: least speech-like time
            cut = hi if quietest is None else quietest(pos + max_chunk / 2, hi)
        out.append((pos, cut))
        pos = cut
    out.append((pos, dur))
    return out


def chunk_bounds(wav, sr, max_chunk=10.0, min_gap=0.1):
    """plan_chunks on the Silero VAD speech segments of wav (16 kHz mono float
    array); a cut with no pause goes to the lowest Silero speech probability."""
    from silero_vad import get_speech_timestamps
    assert sr == VAD_SR, "chunk_bounds expects 16 kHz audio"
    dur = len(wav) / sr
    if dur <= max_chunk:
        return [(0.0, dur)]
    model = _silero()
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    speech = [(t["start"] / sr, t["end"] / sr) for t in get_speech_timestamps(x, model, sampling_rate=sr)]
    probs = []

    def quietest(lo, hi):
        if not probs:                           # one extra Silero pass, only when needed
            with torch.no_grad():
                probs.append(model.audio_forward(x, sr)[0].numpy())
        p = probs[0]
        starts = np.arange(len(p)) * VAD_WIN / sr
        idx = np.where((starts >= lo) & (starts <= hi))[0]
        return float(starts[idx[np.argmin(p[idx])]]) if len(idx) else hi

    return plan_chunks(speech, dur, max_chunk, min_gap, quietest)


def join_pauses(pauses, phones, tol):
    """Merge SIL runs separated by at most tol with no phone between them
    (a pause cut by a chunk edge)."""
    out = []
    for s, e in sorted(pauses):
        if out and s - out[-1][1] <= tol and not any(out[-1][1] < p["end"] and p["start"] < s for p in phones):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------
def _phones_with_tokens(states, tokens, cs, hop, off):
    """SIL-Viterbi states -> phone intervals (as sil_eval._segments) with the
    token index of each phone, and the SIL runs."""
    phones, sils = [], []
    for s, grp in itertools.groupby(enumerate(states), key=lambda x: x[1]):
        grp = list(grp)
        t0, t1 = cs + grp[0][0] * hop + off, cs + (grp[-1][0] + 1) * hop + off
        if s is None:
            sils.append((t0, t1))
            continue
        ph = tokens[s]
        if ph in E.SKIP_TOK:
            continue
        if len(ph) == 1 and unicodedata.combining(ph):      # diacritic: joins the previous phone
            if phones:
                phones[-1]["phone"] += ph
                phones[-1]["end"] = t1
            continue
        phones.append({"phone": E.HYP_FIX.get(ph, ph), "start": t0, "end": t1, "tok": s})
    return phones, sils


def _candidates(post_f, sym_ids, sym_labels, k=K):
    p = {}
    for i, lab in zip(sym_ids, sym_labels):
        p[lab] = p.get(lab, 0.0) + float(post_f[i])
    z = sum(p.values()) or 1.0
    return sorted(((lab, v / z) for lab, v in p.items()), key=lambda x: -x[1])[:k]


@torch.inference_mode()
def transcribe_chunk(model, fe, tok, seg, cs, sym_ids, sym_labels, device, positions, k_min, sil_logp,
                     hop, off):
    m = getattr(model, "module", model)
    x = fe(E.to_model_rate(seg, m), sampling_rate=fe.sampling_rate, return_tensors="pt",
           return_attention_mask=True)
    o = model(input_values=x.input_values.to(device), attention_mask=x.attention_mask.to(device),
              labels=None, prior_weight=0.0, ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
    T, P = int(o.input_lengths[0]), int(o.text_lengths[0])
    if P == 0:                                  # nothing recognised in this chunk
        return [], []
    pred = o.predicted_phone_ids[0, :P].cpu().tolist()
    tokens = tok.convert_ids_to_tokens(pred)
    en = o.energy[0, :T, :P].float()
    sil = o.sil_energy_raw[0, :T].float()
    lp = torch.log_softmax(torch.cat([en, sil[:, None]], -1), -1).cpu().numpy()
    top, path = sil_fs.viterbi(lp, P, sil_fs.slot_mask(P, positions), k_min, sil_logp)
    if path is None:
        return [], []
    st = [None if top["col"][s] < 0 else int(top["col"][s]) for s in path]
    phones, sils = _phones_with_tokens(st, tokens, cs, hop, off)

    # peak frame of each decoded token in its greedy CTC run
    post = o.ctc_logits[0, :T].float().softmax(-1).cpu().numpy()
    blank = tok.pad_token_id
    runs = []
    for v, g in itertools.groupby(enumerate(post.argmax(1).tolist()), key=lambda x: x[1]):
        g = list(g)
        if v != blank:
            runs.append((v, g[0][0], g[-1][0] + 1))
    same = [r[0] for r in runs] == pred
    for p in phones:
        n, v = p["tok"], pred[p["tok"]]
        if same:
            _, f0, f1 = runs[n]
        else:                                  # fallback: the phone's own aligned frames
            f0 = max(0, int(round((p["start"] - cs - off) / hop)))
            f1 = max(f0 + 1, min(T, int(round((p["end"] - cs - off) / hop))))
        f = f0 + int(post[f0:f1, v].argmax())
        p["cands"] = _candidates(post[f], sym_ids, sym_labels)
    return phones, sils


# ---------------------------------------------------------------------------
# One recording
# ---------------------------------------------------------------------------
def load_model(checkpoint, vocab_path=None, device=None):
    import os
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
    from .model import Wav2Vec2ForCTCPoolContrastive
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = Wav2Vec2CTCTokenizer(vocab_path or os.path.join(checkpoint, "vocab.json"),
                               unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = Wav2Vec2ForCTCPoolContrastive.from_pretrained(checkpoint).to(device).eval()
    if not getattr(model, "use_sil", False):
        raise ValueError(f"{checkpoint} is not a latent-SIL checkpoint (no SIL state to decode pauses)")
    return model, fe, tok, device


def transcribe(path, model, fe, tok, device, max_chunk=10.0, symbol_map=None):
    """-> (phones [{phone, start, end, cands: [(symbol, prob)] * K}], pauses [(s, e)], duration)."""
    symbol_map = symbol_map or {}
    off = sil_eval._set_frame(model)
    hop = sil_eval.HOP
    m = getattr(model, "module", model)
    positions = getattr(m, "sil_eval_positions", "all")
    k_min = int(getattr(m, "sil_min_frames", 1))
    sil_logp = float(torch.as_tensor(m.sil_logp).detach())   # a learned parameter in most checkpoints
    vocab = tok.get_vocab()
    sym = [(i, E.HYP_FIX.get(s, s)) for s, i in vocab.items()
           if s not in E.SKIP_TOK and s not in (tok.pad_token, tok.unk_token, "<s>", "</s>")]
    sym_ids = [i for i, _ in sym]
    sym_labels = [symbol_map.get(l, l) for _, l in sym]

    wav = E.load_audio(path)
    dur = len(wav) / E.SR
    phones, pauses = [], []
    for c0, c1 in chunk_bounds(wav, E.SR, max_chunk):
        seg = wav[int(c0 * E.SR):int(c1 * E.SR)]
        if len(seg) < 400:                     # under one conv receptive field
            continue
        ph, si = transcribe_chunk(model, fe, tok, seg, c0, sym_ids, sym_labels, device,
                                  positions, k_min, sil_logp, hop, off)
        phones += ph
        pauses += si
    for p in phones:
        p["phone"] = symbol_map.get(p["phone"], p["phone"])
        p.pop("tok", None)
    phones.sort(key=lambda z: z["start"])
    # the frames of a chunk span about [cs + off, ce - off] (the conv front-end drops up to
    # one more frame at the end): the two SIL runs of a pause cut at a chunk edge are
    # 2 * off to 2 * off + hop apart (15-25 ms at 10 ms frames)
    pauses = join_pauses(pauses, phones, 2 * off + 2 * hop)
    return phones, pauses, dur


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def fmt_prob(v):
    return f"{v:.2f}" if v >= 0.095 else (f"{v:.3f}" if v >= 0.001 else "<0.001")


def _tier(items, dur):
    """[(start, end, label)] -> contiguous intervals over [0, dur], gaps labelled ""."""
    out, t = [], 0.0
    for s, e, lab in sorted(items):
        s, e = max(s, t), min(e, dur)
        if e - s < 1e-4:
            continue
        if s - t > 1e-4:
            out.append((t, s, ""))
        out.append((s, e, lab))
        t = e
    if dur - t > 1e-4 or not out:
        out.append((t, dur, ""))
    return out


def write_textgrid(path, phones, pauses, dur):
    def q(s):
        return '"' + s.replace('"', '""') + '"'
    tiers = [("phones", [(p["start"], p["end"], p["phone"]) for p in phones]),
             ("confidence", [(p["start"], p["end"], fmt_prob(p["cands"][0][1])) for p in phones if p["cands"]])]
    for r in range(2, K + 1):
        tiers.append((f"top{r}", [(p["start"], p["end"], f"{p['cands'][r - 1][0]} {fmt_prob(p['cands'][r - 1][1])}")
                                  for p in phones if len(p["cands"]) >= r]))
    tiers.append(("pauses", [(s, e, "SIL") for s, e in pauses]))
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0", f"xmax = {dur:.6f}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for i, (name, items) in enumerate(tiers, 1):
        iv = _tier(items, dur)
        lines += [f"    item [{i}]:", '        class = "IntervalTier"', f"        name = {q(name)}",
                  "        xmin = 0", f"        xmax = {dur:.6f}", f"        intervals: size = {len(iv)}"]
        for j, (s, e, lab) in enumerate(iv, 1):
            lines += [f"        intervals [{j}]:", f"            xmin = {s:.6f}", f"            xmax = {e:.6f}",
                      f"            text = {q(lab)}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path, phones):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "phone"] + [c for r in range(1, K + 1) for c in (f"top{r}", f"p{r}")])
        for p in phones:
            row = [f"{p['start']:.4f}", f"{p['end']:.4f}", p["phone"]]
            for r in range(K):
                lab, v = p["cands"][r] if r < len(p["cands"]) else ("", None)
                row += [lab, "" if v is None else f"{v:.6f}"]
            w.writerow(row)
