#!/usr/bin/env python3
"""Score checkpoints as phone aligners on an evaluation manifest.

The model aligns its own CTC greedy decode; boundaries are compared with the
manifest's reference intervals (format: reco_align/evaluation/attention_eval.py).

    python evaluate.py --manifest test_manifest.json --out_dir eval/ \\
        runs/aligner_10ms_sil/checkpoint-43000 [more checkpoints ...]

--method
  attention     DTW on the cross-attention energy; for a latent-SIL model, the
                SIL Viterbi decode (and the DTW decode of the same pass, keys
                suffixed _dtw), plus SIL scores when the manifest has
                pause_intervals.
  ctcfa         CTC forced alignment of the same decoded phones on the CTC head,
                one interval per non-blank run.
  ctcfa_contig  as ctcfa, each phone ending where the next one starts.

Outputs, per checkpoint, in --out_dir:
    <tag>.csv       per-file / per-style / GLOBAL metrics
    <tag>.pkl       alignment store (reference + hypothesis intervals)
    summary.jsonl   one line per checkpoint and method
with <tag> = <run>_<checkpoint>_<method>.
"""
import argparse
import json
import math
import os
import pickle

import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

from reco_align.model import Wav2Vec2ForCTCPoolContrastive
from reco_align.evaluation import sil_eval
from reco_align.evaluation.attention_eval import run_manifest, duration_fidelity
from reco_align.evaluation.ctcfa_eval import run_manifest_ctcfa
from reco_align.evaluation.metrics import compute_metrics, align_sequences


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", choices=("attention", "ctcfa", "ctcfa_contig"), default="attention")
    ap.add_argument("--vocab_path", default=None,
                    help="phone vocabulary (default: vocab.json in the checkpoint)")
    ap.add_argument("--sil_positions", choices=("all", "edges"), default=None,
                    help="SIL slots of the Viterbi decode (default: as trained)")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--per_phoneme", action="store_true", help="also write a per-phoneme CSV")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    manifest = json.load(open(a.manifest))

    for ck in a.checkpoints:
        ck = ck.rstrip("/")
        vocab = a.vocab_path or os.path.join(ck, "vocab.json")
        tok = Wav2Vec2CTCTokenizer(vocab, unk_token="[UNK]", pad_token="[PAD]",
                                   word_delimiter_token="")
        run = os.path.basename(os.path.dirname(os.path.abspath(ck)))
        tag = f"{run}_{os.path.basename(ck)}_{a.method}"
        model = Wav2Vec2ForCTCPoolContrastive.from_pretrained(ck).to(device).eval()
        print("=" * 72)
        print(f"{ck}  [{a.method}]  vocab: tokenizer {len(tok)} | model {model.config.vocab_size}")

        extra = {}
        with torch.no_grad():
            if a.method == "attention" and model.use_sil:
                extra, store, _ = sil_eval.evaluate(model, fe, tok, manifest, device=device,
                                                    max_files=a.max_files, positions=a.sil_positions)
            elif a.method == "attention":
                store = run_manifest(model, fe, tok, manifest, device=device, max_files=a.max_files)
            else:
                store = run_manifest_ctcfa(model, fe, tok, manifest, device=device,
                                           max_files=a.max_files,
                                           end_mode="contiguous" if a.method == "ctcfa_contig" else "span")
        with open(os.path.join(a.out_dir, tag + ".pkl"), "wb") as f:
            pickle.dump(store, f)
        per_ph = os.path.join(a.out_dir, tag + "_per_phoneme.csv") if a.per_phoneme else None
        df = compute_metrics(store, os.path.join(a.out_dir, tag + ".csv"), per_ph,
                             f1_tolerances=(0.01, 0.02, 0.05))
        g = df[df.group == "GLOBAL"].iloc[0]
        dur = duration_fidelity(store, align_sequences)
        for c in df.columns:
            if c not in ("group", "style"):
                print(f"  {c:28s} {g[c]}")

        row = {"checkpoint": ck, "method": a.method,
               "F1@10": float(g["F1@10ms (%)"]), "F1@20": float(g["F1@20ms (%)"]),
               "F1@50": float(g["F1@50ms (%)"]), "MedianBE": float(g["MedianBE (ms)"]),
               "P90": float(g["P90 (ms)"]), "PER": float(g["PER (%)"]), **dur}
        # SIL scores and the DTW decode of the same pass (latent-SIL models only)
        row.update({k: v for k, v in extra.items()
                    if k.startswith("sil_") or k.endswith("_dtw") or k.endswith("_sp")})
        row = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
               for k, v in row.items()}
        for k, v in row.items():
            if k not in ("checkpoint", "method") and k not in g.index:
                print(f"  {k:28s} {v}")
        print("  SUMMARY", json.dumps(row))
        with open(os.path.join(a.out_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
