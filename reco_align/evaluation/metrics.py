"""
Phone-alignment metrics.

Input: an alignment store {file_id: {ref_intervals, hyp_intervals, ref_seq, hyp_seq, style}},
times in SECONDS; all reported errors are in milliseconds.

  - Boundary F1 at several tolerances: the boundary set is every phone onset
    together with the final offset, deduplicated; each REFERENCE boundary is
    matched to the NEAREST unused hypothesis boundary within the tolerance
    (one-to-one); TP/FP/FN are pooled over all utterances before P, R and F1.
  - Boundary errors on phones matched by edit distance (same label only):
    mean / median / P90, rates of errors > 50 ms and > 100 ms, signed mean
    shift of starts and ends, duration error.
  - PER of the hypothesis phone sequence.

Sequences are aligned on a shared character encoding, so multi-character
phone tokens (IPA strings) are compared correctly. Group statistics are pooled
over every phone of the group (micro-average).
"""

import numpy as np
import pandas as pd
import Levenshtein
from jiwer import process_words


# ----------------------------------------------------------------------
# Sequence alignment (Levenshtein), robust to multi-char phoneme tokens
# ----------------------------------------------------------------------
def _editops(ref, hyp):
    """editops on a shared char-encoding so IPA tokens of any length work."""
    vocab = {}

    def enc(seq):
        out = []
        for t in seq:
            if t not in vocab:
                vocab[t] = chr(0xE000 + len(vocab))  # private-use code points
            out.append(vocab[t])
        return "".join(out)

    return Levenshtein.editops(enc(ref), enc(hyp))


def align_sequences(ref, hyp):
    """Return list of (ref_idx, hyp_idx); None on either side = del/ins."""
    alignment = []
    ops = _editops(ref, hyp)
    ref_idx = hyp_idx = op_idx = 0

    while ref_idx < len(ref) or hyp_idx < len(hyp):
        if op_idx < len(ops):
            op_type, src_pos, dest_pos = ops[op_idx]
            if op_type == "delete" and src_pos == ref_idx:
                alignment.append((ref_idx, None)); ref_idx += 1; op_idx += 1; continue
            elif op_type == "insert" and dest_pos == hyp_idx:
                alignment.append((None, hyp_idx)); hyp_idx += 1; op_idx += 1; continue
            elif op_type == "replace" and src_pos == ref_idx and dest_pos == hyp_idx:
                alignment.append((ref_idx, hyp_idx)); ref_idx += 1; hyp_idx += 1; op_idx += 1; continue
        if ref_idx < len(ref) and hyp_idx < len(hyp):
            alignment.append((ref_idx, hyp_idx)); ref_idx += 1; hyp_idx += 1
        elif ref_idx < len(ref):
            alignment.append((ref_idx, None)); ref_idx += 1
        elif hyp_idx < len(hyp):
            alignment.append((None, hyp_idx)); hyp_idx += 1
    return alignment


# ----------------------------------------------------------------------
# Per-phoneme errors (only phoneme-matched pairs contribute boundary error)
# ----------------------------------------------------------------------
def per_phoneme_errors(ref_alignments, hyp_alignments, ref_seq, hyp_seq):
    """One record per correctly-matched phoneme. Times in seconds in -> ms out."""
    alignment = align_sequences(ref_seq, hyp_seq)
    records = []
    for ref_idx, hyp_idx in alignment:
        if ref_idx is None or hyp_idx is None:
            continue
        if ref_seq[ref_idx] != hyp_seq[hyp_idx]:
            continue
        rs, re = ref_alignments[ref_idx]["start"], ref_alignments[ref_idx]["end"]
        hs, he = hyp_alignments[hyp_idx]["start"], hyp_alignments[hyp_idx]["end"]
        records.append({
            "phoneme":      ref_seq[ref_idx],
            # signed (keep direction): + = hypothesis is LATE
            "signed_start": (hs - rs) * 1000.0,
            "signed_end":   (he - re) * 1000.0,
            # absolute magnitudes
            "start_err":    abs(hs - rs) * 1000.0,
            "end_err":      abs(he - re) * 1000.0,
            "dur_err":      abs((he - hs) - (re - rs)) * 1000.0,
            "mid_err":      abs((hs + he) / 2 - (rs + re) / 2) * 1000.0,
        })
    return records


# ----------------------------------------------------------------------
# Boundary-detection F1 (time-only, label-agnostic).
#
# Boundary set: every phone onset together with the final offset.
#   Taking all starts AND all ends then deduplicating is equivalent to
#   "starts + last end" under a contiguous segmentation (phone n's end IS
#   phone n+1's start), but stays correct when a pause leaves a gap between
#   consecutive intervals, where "starts + last end" would silently drop the
#   offset preceding the gap.
#
# Matching: each REFERENCE boundary takes the NEAREST unused hypothesis
#   boundary within the tolerance, one-to-one.
#   FP = hypothesis boundaries left unmatched, FN = reference boundaries with
#   no hypothesis boundary within tolerance. A boundary that exists but is
#   outside tolerance therefore costs one FP and one FN.
#
# tolerance is in SECONDS.
# ----------------------------------------------------------------------
def boundary_times(intervals):
    b = [iv["start"] for iv in intervals] + [iv["end"] for iv in intervals]
    return sorted(set(b))


def compute_f1_counts(ref_bounds, pred_bounds, tolerance):
    ref_bounds  = np.sort(np.asarray(ref_bounds,  dtype=float))
    pred_bounds = np.sort(np.asarray(pred_bounds, dtype=float))
    used = np.zeros(len(pred_bounds), dtype=bool)
    TP = 0
    for rb in ref_bounds:
        if not len(pred_bounds):
            break
        d = np.abs(pred_bounds - rb)
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= tolerance:
            used[j] = True
            TP += 1
    FP = len(pred_bounds) - TP
    FN = len(ref_bounds) - TP
    return TP, FP, FN


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0.0


# ----------------------------------------------------------------------
# Pooled boundary statistics over a set of per-phoneme records (ms).
# ----------------------------------------------------------------------
def boundary_stats(df):
    if len(df) == 0:
        return {}
    s, e, d = df["start_err"], df["end_err"], df["dur_err"]
    both = pd.concat([s, e])  # AAS / tail pool start AND end boundaries
    return {
        "K":                      int(len(df)),
        # --- central ---
        "AAS (ms)":               both.mean(),               # mean |error| over all boundaries
        "MedianBE (ms)":          both.median(),             # median |error| over all boundaries
        "Median_start (ms)":      s.median(),
        "Mean_start (ms)":        s.mean(),
        "Median_end (ms)":        e.median(),
        "Mean_end (ms)":          e.mean(),
        # --- tail: pooled over start+end ---
        "P90 (ms)":               both.quantile(0.90),
        "%>50ms":                 (both > 50).mean() * 100,  # gross-error rate
        "%>100ms":                (both > 100).mean() * 100,
        "%within20ms":            (both <= 20).mean() * 100,
        # --- direction of bias ---
        "SignedMean_start (ms)":  df["signed_start"].mean(), # + = hyp late
        "SignedMean_end (ms)":    df["signed_end"].mean(),
        # --- duration ---
        "Mean_dur (ms)":          d.mean(),
        "Median_dur (ms)":        d.median(),
        "%dur>50ms":              (d > 50).mean() * 100,
    }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def compute_metrics(alignment_store, csv_path, per_phoneme_csv=None,
                    f1_tolerances=(0.02, 0.05)):
    """
    alignment_store: {file_id: {ref_intervals, hyp_intervals, ref_seq, hyp_seq, style}}
        *_intervals: list of {"start": sec, "end": sec}
        *_seq:       list of phoneme tokens
    Writes a file/style/global metrics CSV, and optionally a per-phoneme CSV.
    All boundary stats are pooled (micro-averaged) within each group.
    """
    all_records = []   # long-form: one row per matched phoneme, tagged with file+style
    f1_rows = []       # per-file F1 counts + PER, pooled later

    for f, data in alignment_store.items():
        ref_intervals = data["ref_intervals"]
        hyp_intervals = data["hyp_intervals"]
        ref_seq, hyp_seq = data["ref_seq"], data["hyp_seq"]
        style = data.get("style", "ALL")

        recs = per_phoneme_errors(ref_intervals, hyp_intervals, ref_seq, hyp_seq)
        for r in recs:
            r["file"] = f; r["style"] = style
        all_records.extend(recs)

        # F1 over all boundary times (label-agnostic boundary detection)
        ref_bounds  = boundary_times(ref_intervals)
        pred_bounds = boundary_times(hyp_intervals)
        row = {"file": f, "style": style, "N_ref": len(ref_seq), "N_hyp": len(hyp_seq),
               "B_ref": len(ref_bounds), "B_hyp": len(pred_bounds)}
        for tol in f1_tolerances:
            tp, fp, fn = compute_f1_counts(ref_bounds, pred_bounds, tol)
            tag = f"{int(round(tol * 1000))}"
            row[f"TP_{tag}"], row[f"FP_{tag}"], row[f"FN_{tag}"] = tp, fp, fn

        out = process_words(" ".join(ref_seq), " ".join(hyp_seq))
        row["S"], row["D"], row["I"] = out.substitutions, out.deletions, out.insertions
        f1_rows.append(row)

    records_df = pd.DataFrame(all_records)
    f1_df = pd.DataFrame(f1_rows)
    tags = [f"{int(round(t * 1000))}" for t in f1_tolerances]

    def assemble(rec_subset, f1_subset, label, style_val):
        out = {"group": label, "style": style_val}
        out.update(boundary_stats(rec_subset))
        out["N_ref"] = int(f1_subset["N_ref"].sum())
        out["N_hyp"] = int(f1_subset["N_hyp"].sum())
        out["B_ref"] = int(f1_subset["B_ref"].sum())
        out["B_hyp"] = int(f1_subset["B_hyp"].sum())
        S, D, I = f1_subset["S"].sum(), f1_subset["D"].sum(), f1_subset["I"].sum()
        out["PER (%)"] = (S + D + I) / out["N_ref"] * 100 if out["N_ref"] else np.nan
        out["Deletions"], out["Insertions"] = int(D), int(I)
        for tag in tags:
            tp = f1_subset[f"TP_{tag}"].sum()
            fp = f1_subset[f"FP_{tag}"].sum()
            fn = f1_subset[f"FN_{tag}"].sum()
            out[f"P@{tag}ms (%)"]  = tp / (tp + fp) * 100 if (tp + fp) > 0 else np.nan
            out[f"R@{tag}ms (%)"]  = tp / (tp + fn) * 100 if (tp + fn) > 0 else np.nan
            out[f"F1@{tag}ms (%)"] = _f1(tp, fp, fn)
        return out

    rows = []
    # per file
    for f in f1_df["file"]:
        rsub = records_df[records_df["file"] == f] if len(records_df) else records_df
        fsub = f1_df[f1_df["file"] == f]
        rows.append(assemble(rsub, fsub, f, fsub["style"].iloc[0]))
    # per style
    for s, fsub in f1_df.groupby("style"):
        rsub = records_df[records_df["style"] == s] if len(records_df) else records_df
        rows.append(assemble(rsub, fsub, f"STYLE_{s}", s))
    # global
    rows.append(assemble(records_df, f1_df, "GLOBAL", "ALL"))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # optional per-phoneme table (pooled over all files)
    if per_phoneme_csv is not None and len(records_df):
        ph_rows = []
        for ph, sub in records_df.groupby("phoneme"):
            r = {"phoneme": ph}; r.update(boundary_stats(sub)); ph_rows.append(r)
        pd.DataFrame(ph_rows).sort_values("AAS (ms)", ascending=False)\
            .to_csv(per_phoneme_csv, index=False)

    return df