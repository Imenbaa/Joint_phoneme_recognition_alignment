"""HTML report of a transcription: one self-contained page per recording, plus an
index of the recordings of an output folder. Opened by double-clicking; no
server, no internet.

A recording's page shows, on one time axis that scrolls and zooms:
  waveform, recognised phones and pauses (hover: the top-5 of a phone; click: play it),
  attention   the aligner's log attention over the recognised phones and SIL,
              with the SIL-Viterbi path (where the boundaries come from),
  posteriors  the CTC log posterior of every vocabulary symbol per frame, and P(blank),
  top 5       the five most probable symbols of each frame, by rank (from the unclipped posteriors).

The arrays are stored as 8-bit values (log probabilities clipped at -10); the
attention is kept in a band of +-BAND phones around the path, so a page stays
small (about 1 MB per minute of audio).
"""
import base64
import html
import json
import os
import re

import numpy as np

from .evaluation import attention_eval as E

FLOOR = -10.0
BAND = 10
# vocabulary rows of the posterior plots: blank, then the phones by class
ORDER = ["a", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "9", "ə",      # oral vowels
         "@", "1", "§",                                              # nasal vowels
         "j", "ɥ", "w",                                              # glides
         "p", "t", "k", "b", "d", "ɡ", "g",                          # stops
         "f", "s", "ʃ", "v", "z", "ʒ", "ts", "tʃ",                   # fricatives, affricates
         "m", "n", "ɲ", "ŋ",                                         # nasals
         "l", "ʁ"]                                                   # liquids
CLASS_START = ["@", "j", "p", "f", "m", "l"]


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _q(logp):
    """log probabilities -> uint8, 0 = FLOOR (or less), 255 = log 1."""
    return np.round((np.clip(logp, FLOOR, 0.0) - FLOOR) * (255.0 / -FLOOR)).astype(np.uint8)


def vocab_rows(tok, symbol_map):
    """[(vocab id, label)]: blank first, then the phones in ORDER, then any other phone."""
    vocab = tok.get_vocab()
    special = set(E.SKIP_TOK) | {tok.unk_token, tok.pad_token, "<s>", "</s>"}
    rows = [(vocab[tok.pad_token], "blank")]
    for s in ORDER + sorted(vocab):
        if s in vocab and s not in special and all(i != vocab[s] for i, _ in rows):
            rows.append((vocab[s], symbol_map.get(s, s)))
    breaks = [r for r, (i, _) in enumerate(rows)
              if any(vocab.get(s) == i for s in CLASS_START)]
    return rows, [1] + breaks


def _waveform(wav, dur, hop):
    """min and max of the (peak-normalised) signal per frame, as int8 pairs."""
    n = int(np.ceil(dur / hop))
    step = max(1, int(round(hop * E.SR)))
    x = np.zeros(n * step, np.float32)
    x[:min(len(wav), n * step)] = wav[:n * step]
    x = x.reshape(n, step)
    peak = float(np.abs(wav).max()) or 1.0
    mm = np.stack([x.min(1), x.max(1)], 1) / peak
    return np.round(mm * 127).astype(np.int8)


def page_data(name, audio_rel, wav, phones, pauses, dur, trace, tok, symbol_map):
    rows, breaks = vocab_rows(tok, symbol_map)
    ids = [i for i, _ in rows]
    hop = trace[0]["hop"] if trace else 0.01
    chunks, tokens = [], []
    for t in trace:
        lp_rows = t["ctc_logp"][:, ids]                  # (T, rows)
        # the top 5 of each frame from the unclipped posteriors: the 8-bit copy ties every symbol
        # under exp(FLOOR), which is all of them but blank in most frames
        top5 = np.argsort(-lp_rows, axis=1, kind="stable")[:, :5].astype(np.uint8)
        c = {"t0": t["start"] + t["offset"], "T": t["frames"], "post": _b64(_q(lp_rows.T)), "top5": _b64(top5)}
        if t["path"] is not None:
            toks = [symbol_map.get(E.HYP_FIX.get(s, s), E.HYP_FIX.get(s, s)) for s in t["tokens"]]
            P, T = len(toks), t["frames"]
            path = np.array(t["path"], np.int16)
            center = path.copy()                       # SIL frames: the band stays on the last phone
            last = 0
            for f in range(T):
                if center[f] < 0:
                    center[f] = last
                else:
                    last = center[f]
            band = np.zeros((2 * BAND + 1, T), np.uint8)
            lp = t["att_logp"]
            for k in range(-BAND, BAND + 1):
                r = center.astype(np.int64) + k
                ok = (r >= 0) & (r < P)
                band[k + BAND, ok] = _q(lp[np.arange(T)[ok], r[ok]])
            c.update(tok0=len(tokens), P=P, center=_b64(center.astype("<i2")), path=_b64(path.astype("<i2")),
                     band=_b64(band), sil=_b64(_q(lp[:, P])))
            tokens += toks
        chunks.append(c)
    return {
        "name": name, "audio": audio_rel, "duration": dur, "hop": hop, "floor": FLOOR, "band": BAND,
        "rows": [l for _, l in rows], "breaks": breaks, "tokens": tokens, "chunks": chunks,
        "wave": _b64(_waveform(wav, dur, hop)),
        "phones": [[round(p["start"], 4), round(p["end"], 4), p["phone"],
                    [[s, round(v, 4)] for s, v in p["cands"]]] for p in phones],
        "pauses": [[round(s, 4), round(e, 4)] for s, e in pauses],
    }


def _template(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_assets", name),
              encoding="utf-8") as f:
        return f.read()


def _audio_ref(audio_path, out_dir):
    try:
        return os.path.relpath(os.path.abspath(audio_path), os.path.abspath(out_dir)).replace(os.sep, "/")
    except ValueError:                                  # another drive (Windows)
        return "file:///" + os.path.abspath(audio_path).replace(os.sep, "/").lstrip("/")


def write_page(path, audio_path, phones, pauses, dur, trace, tok, symbol_map):
    name = os.path.splitext(os.path.basename(path))[0]
    wav = E.load_audio(audio_path)
    data = page_data(name, _audio_ref(audio_path, os.path.dirname(path)), wav, phones, pauses, dur,
                     trace, tok, symbol_map)
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (_template("page.html").replace("__TITLE__", html.escape(name))
            .replace("/*__DATA__*/null", js))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def summary(name, phones, pauses, dur, low=0.5):
    conf = [p["cands"][0][1] for p in phones if p["cands"]]
    return {"name": name, "duration": round(dur, 2), "phones": len(phones), "pauses": len(pauses),
            "pause_time": round(sum(e - s for s, e in pauses), 2),
            "mean_conf": round(float(np.mean(conf)), 3) if conf else None,
            "low_conf": int(sum(c < low for c in conf))}


def _previous_rows(path):
    """Rows of an existing index.html whose page is still there (earlier runs into the same folder)."""
    try:
        m = re.search(r"const D = (\{.*?\});\n", open(path, encoding="utf-8").read())
        rows = json.loads(m.group(1))["rows"]
    except Exception:
        return []
    d = os.path.dirname(path)
    return [r for r in rows if os.path.exists(os.path.join(d, r["name"] + ".html"))]


def write_index(path, rows, failed=()):
    """rows: summary() dicts of the recordings with a page; failed: [(file, reason)]. Recordings
    of earlier runs into the same folder stay listed while their page exists."""
    names = {r["name"] for r in rows}
    rows = [r for r in _previous_rows(path) if r["name"] not in names] + list(rows)
    data = {"rows": sorted(rows, key=lambda r: r["name"]),
            "failed": [[os.path.basename(f), why] for f, why in failed]}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_template("index.html").replace("/*__DATA__*/null", js))
