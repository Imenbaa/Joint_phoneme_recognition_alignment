"""Comparison of a transcription with a reference phone TextGrid.

    JointRecAlign transcribe recordings/ --reference textgrids/ [--ref_tier phones] [--ref_map map.json]

For <name>.wav the reference is <reference folder>/<name>.TextGrid. Its tier
(--ref_tier, or the first tier named phones / phone / phonemes / phonèmes /
phonemes, or else the first interval tier) is read as:
  pause labels (--pause_labels: "", SIL, sil, sp, spn, _, #, <p:>, ...)  -> reference pauses
  skip labels (--skip_labels, none by default: e.g. noise, overlap)      -> left out
  every other label, rewritten by --ref_map                              -> reference phones
and compared with the transcription as in the evaluation (evaluate.py):
  PER          substitutions + deletions + insertions over the reference phones
  F1@20/50 ms  boundary detection: every phone onset and offset, each reference boundary
               matched one-to-one to the nearest hypothesis boundary within 20 / 50 ms
  MedianBE     median |error| of the starts and ends of the phones matched with the same label
  pause P/R/F1 reference pauses matched one-to-one by a hypothesis pause (the time not
               covered by a recognised phone) with intersection over union >= 0.5
Corpus figures pool the counts of all recordings.
"""
import json
import os
import re

import numpy as np

PAUSE_LABELS = ("", "SIL", "sil", "sp", "SP", "spn", "_", "#", "<p:>", "<SIL>", "pause")
TIER_NAMES = ("phones", "phone", "phonemes", "phonèmes", "phoneme", "phonème", "phon", "phonetic")
_TOK = re.compile(r'"((?:[^"]|"")*)"|(\[\s*\d+\s*\])|(<exists>|<absent>)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


# ---------------------------------------------------------------------------
# TextGrid reading (long and short text formats, UTF-8 or UTF-16)
# ---------------------------------------------------------------------------
def _text(path):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue


def read_textgrid(path):
    """-> {tier name: [(start, end, label)]} for the interval tiers (point tiers are skipped)."""
    vals = []
    for m in _TOK.finditer(_text(path)):
        if m.group(1) is not None:
            vals.append(m.group(1).replace('""', '"'))
        elif m.group(3) is not None:
            vals.append(m.group(3))
        elif m.group(4) is not None:
            vals.append(float(m.group(4)))
    if len(vals) < 2 or vals[0] != "ooTextFile" or vals[1] != "TextGrid":
        raise ValueError("not a Praat TextGrid in text format")
    i = 4                                       # "ooTextFile" "TextGrid" xmin xmax
    if vals[i] != "<exists>":
        return {}
    n_tiers, i = int(vals[i + 1]), i + 2
    tiers = {}
    for _ in range(n_tiers):
        cls, name, n = vals[i], vals[i + 1], int(vals[i + 4])
        i += 5
        if cls == "IntervalTier":
            tiers.setdefault(name, [(float(vals[i + 3 * k]), float(vals[i + 3 * k + 1]), str(vals[i + 3 * k + 2]))
                                    for k in range(n)])
            i += 3 * n
        else:                                   # TextTier: time, mark
            i += 2 * n
    return tiers


def find_reference(audio_path, folder):
    """<folder>/<name>.TextGrid (any case of the extension), or None."""
    name = os.path.splitext(os.path.basename(audio_path))[0]
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if stem == name and ext.lower() == ".textgrid":
            return os.path.join(folder, f)
    return None


def pick_tier(tiers, name=None):
    if name:
        if name not in tiers:
            raise ValueError(f"no interval tier {name!r} (tiers: {', '.join(tiers) or 'none'})")
        return name
    low = {t.lower(): t for t in tiers}
    for n in TIER_NAMES:
        if n in low:
            return low[n]
    if not tiers:
        raise ValueError("no interval tier")
    return next(iter(tiers))


def load_reference(path, tier=None, ref_map=None, pause_labels=PAUSE_LABELS, skip_labels=()):
    """-> {"tier", "phones": [{"phone", "start", "end"}], "pauses": [(s, e)], "skipped": [(s, e)], "end"}"""
    tiers = read_textgrid(path)
    name = pick_tier(tiers, tier)
    ref_map = ref_map or {}
    pause_labels, skip_labels = set(pause_labels), set(skip_labels)
    phones, pauses, skipped = [], [], []
    for s, e, lab in tiers[name]:
        lab = lab.strip()
        if lab in pause_labels or not lab:             # an empty interval is never a phone
            if pauses and abs(pauses[-1][1] - s) < 1e-6:     # adjacent pause intervals: one pause
                pauses[-1] = (pauses[-1][0], e)
            else:
                pauses.append((s, e))
        elif lab in skip_labels:
            skipped.append((s, e))
        else:
            phones.append({"phone": ref_map.get(lab, lab), "start": s, "end": e})
    end = max([e for _, e, _ in tiers[name]] or [0.0])
    return {"tier": name, "phones": phones, "pauses": pauses, "skipped": skipped, "end": end}


def load_map(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
TOLS = (0.02, 0.05)


def compare(ref, phones, dur):
    """Counts of one recording (pooled later by totals) and the per-phone match status
    for the plots: ref_status / hyp_status in {"ok", "sub", "del", "ins"}, pairs [(ref i, hyp j)]."""
    from jiwer import process_words
    from .evaluation import metrics as M
    from .evaluation import sil_eval
    rseq, hseq = [p["phone"] for p in ref["phones"]], [p["phone"] for p in phones]
    c = {"n_ref": len(rseq), "n_hyp": len(hseq)}
    if rseq or hseq:
        o = process_words(" ".join(rseq) or "∅", " ".join(hseq) or "∅")
        c.update(S=o.substitutions, D=o.deletions, I=o.insertions)
    else:
        c.update(S=0, D=0, I=0)
    rb, hb = M.boundary_times(ref["phones"]), M.boundary_times(phones)
    c.update(b_ref=len(rb), b_hyp=len(hb))
    for tol in TOLS:
        tp, fp, fn = M.compute_f1_counts(rb, hb, tol)
        t = int(round(tol * 1000))
        c.update({f"tp{t}": tp, f"fp{t}": fp, f"fn{t}": fn})
    errs = M.per_phoneme_errors(ref["phones"], phones, rseq, hseq)
    c["errors_ms"] = [r["start_err"] for r in errs] + [r["end_err"] for r in errs]
    hyp_pauses = sil_eval._sil_timeline(phones, dur)
    c["pause_ref"], c["pause_hyp"] = len(ref["pauses"]), len(hyp_pauses)
    c["pause_tp"] = sil_eval._pause_matches(ref["pauses"], hyp_pauses, 0.5)

    ref_status, hyp_status, pairs = ["del"] * len(rseq), ["ins"] * len(hseq), []
    for i, j in M.align_sequences(rseq, hseq):
        if i is not None and j is not None:
            st = "ok" if rseq[i] == hseq[j] else "sub"
            ref_status[i] = hyp_status[j] = st
            pairs.append((i, j))
    return c, {"ref_status": ref_status, "hyp_status": hyp_status, "pairs": pairs}


def scores(counts):
    """Pooled counts (one dict or a list of them) -> the reported figures (percent, ms)."""
    if isinstance(counts, dict):
        counts = [counts]
    tot = lambda k: sum(c[k] for c in counts)
    pct = lambda a, b: 100.0 * a / b if b else None
    f1 = lambda p, r: (2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else
                       (0.0 if p is not None and r is not None else None))
    n_ref = tot("n_ref")
    out = {"n_ref": n_ref, "n_hyp": tot("n_hyp"), "S": tot("S"), "D": tot("D"), "I": tot("I"),
           "PER": pct(tot("S") + tot("D") + tot("I"), n_ref)}
    for t in (int(round(x * 1000)) for x in TOLS):
        tp, fp, fn = tot(f"tp{t}"), tot(f"fp{t}"), tot(f"fn{t}")
        out[f"F1@{t}"] = f1(pct(tp, tp + fp), pct(tp, tp + fn))
    errs = [e for c in counts for e in c["errors_ms"]]
    out["MedianBE"] = float(np.median(errs)) if errs else None
    out["P90"] = float(np.percentile(errs, 90)) if errs else None
    tp, nh, nr = tot("pause_tp"), tot("pause_hyp"), tot("pause_ref")
    out.update(pause_ref=nr, pause_hyp=nh, pause_P=pct(tp, nh), pause_R=pct(tp, nr))
    out["pause_F1"] = f1(out["pause_P"], out["pause_R"])
    return out


CSV_COLUMNS = ["recording", "n_ref", "n_hyp", "S", "D", "I", "PER", "F1@20", "F1@50", "MedianBE", "P90",
               "pause_ref", "pause_hyp", "pause_P", "pause_R", "pause_F1"]


def write_csv(path, rows):
    """rows: [(name, scores dict)]; the last row should be the corpus total."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for name, s in rows:
            w.writerow([name] + ["" if s.get(k) is None else (round(s[k], 2) if isinstance(s[k], float) else s[k])
                                 for k in CSV_COLUMNS[1:]])
