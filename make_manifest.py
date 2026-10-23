#!/usr/bin/env python3
"""Build an evaluation manifest (see evaluate.py) from audio files and Praat
TextGrids with a phone tier.

For every <audio_dir>/**/<id>.wav with a TextGrid <textgrid_dir>/<id><suffix>:
  chunks              WhisperX VAD chunks of at most --max_chunk_dur seconds
                      (jointrecalign/evaluation/vad_chunking.py)
  ref_intervals       the phone intervals of --tier, labels mapped with --phone_map;
                      pause and skipped labels are left out
  pause_intervals     intervals whose label is in --pause_labels (used by the SIL scores)
  excluded_intervals  intervals whose label is in --skip_labels (non-phone labels
                      that are not pauses: noise, overlaps, unknown words, ...)
  style               optional grouping key, from --style rules on the file id

Runs in the environment of requirements-manifest.txt (whisperx), not the
training one:

    python make_manifest.py --audio_dir wav/ --textgrid_dir textgrids/ --tier phones \\
        --phone_map phone_map.json --out manifest.json

--phone_map is a JSON object {"tier label": "vocabulary phone"} that rewrites the
reference labels into the model's phone set (vocab.json); unmapped labels are
kept as they are.
"""
import argparse
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf
import tgt
import torch

from jointrecalign.evaluation.vad_chunking import vad_chunk_with_timestamps

SR = 16000
PAUSE_LABELS = ["", "SIL", "sil", "spn", "SP", "<SIL>", "_"]


def load_audio(path):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = np.mean(a, axis=1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


def read_textgrid(path, tier, phone_map, pause_labels, skip_labels):
    """-> (ref_intervals, pause_intervals, excluded_intervals)"""
    t = tgt.io.read_textgrid(path).get_tier_by_name(tier)
    ref, pauses, excluded = [], [], []
    for iv in t.intervals:
        lab = iv.text.strip()
        s, e = round(iv.start_time, 6), round(iv.end_time, 6)
        if lab in pause_labels:
            pauses.append([s, e])
            continue
        if lab in skip_labels:
            excluded.append([s, e, lab])
            continue
        ph = phone_map.get(lab, lab)
        if ph in pause_labels or ph in skip_labels:
            continue
        ref.append({"phoneme": ph, "start": s, "end": e})
    return ref, pauses, excluded


def style_of(file_id, rules):
    for pattern, style in rules:
        if pattern in file_id:
            return style
    return "ALL"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio_dir", required=True, help="searched recursively for *.wav")
    p.add_argument("--textgrid_dir", required=True)
    p.add_argument("--textgrid_suffix", default=".TextGrid",
                   help="TextGrid file name = <audio id> + this suffix "
                        "(write --textgrid_suffix=-x.TextGrid if it starts with '-')")
    p.add_argument("--tier", required=True, help="phone tier name")
    p.add_argument("--out", required=True)
    p.add_argument("--phone_map", default=None, help="JSON {tier label: vocabulary phone}")
    p.add_argument("--pause_labels", nargs="*", default=PAUSE_LABELS,
                   help="labels of pause intervals (default: %(default)s)")
    p.add_argument("--skip_labels", nargs="*", default=[],
                   help="other non-phone labels to leave out of the reference")
    p.add_argument("--style", default="", metavar="SUBSTRING=STYLE,...",
                   help="style of a file whose id contains SUBSTRING (first match), "
                        "e.g. --style=_read=read,_spont=spontaneous")
    p.add_argument("--max_chunk_dur", type=float, default=10.0)
    p.add_argument("--skip", nargs="*", default=[], help="file ids to leave out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    phone_map = json.load(open(a.phone_map)) if a.phone_map else {}
    pause_labels, skip_labels = set(a.pause_labels), set(a.skip_labels)
    rules = [r.split("=", 1) for r in a.style.split(",") if r]

    man = {"config": vars(a), "files": {}}
    wavs = sorted(glob.glob(os.path.join(a.audio_dir, "**", "*.wav"), recursive=True))
    for k, wp in enumerate(wavs):
        nm = os.path.splitext(os.path.basename(wp))[0]
        if nm in a.skip:
            continue
        tg = os.path.join(a.textgrid_dir, nm + a.textgrid_suffix)
        if not os.path.exists(tg):
            print(f"[skip] no TextGrid: {nm}")
            continue
        audio = load_audio(wp)
        ref, pauses, excluded = read_textgrid(tg, a.tier, phone_map, pause_labels, skip_labels)
        chunks = vad_chunk_with_timestamps(
            torch.from_numpy(audio).float(), sampling_rate=SR,
            max_chunk_duration=a.max_chunk_dur, device=a.device)
        man["files"][nm] = {
            "wav": os.path.abspath(wp), "duration": len(audio) / SR,
            "style": style_of(nm, rules),
            "ref_intervals": ref,
            "chunks": [{"start": c["start"], "end": c["end"]} for c in chunks],
            "pause_intervals": pauses,
            "excluded_intervals": excluded,
        }
        print(f"[{k + 1}/{len(wavs)}] {nm}  ref={len(ref)}  pauses={len(pauses)}  chunks={len(chunks)}")

    json.dump(man, open(a.out, "w"), ensure_ascii=False)
    print(f"\n[saved] {a.out}  {len(man['files'])} files, "
          f"{sum(len(v['chunks']) for v in man['files'].values())} chunks")


if __name__ == "__main__":
    main()
