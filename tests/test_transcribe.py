#!/usr/bin/env python3
"""CPU tests of reco_align.transcribe (no data, no pretrained weights):

    python tests/test_transcribe.py

1. plan_chunks: chunks cover the recording back to back, each <= max_chunk;
   cuts in the middle of the latest pause, a long pause cut inside the window,
   no pause -> the quietest point of the second half of the window.
2. join_pauses: a pause split by a chunk edge is joined, not across a phone or
   a longer gap.
3. Candidates: summed per label, renormalised, sorted, top 5.
4. TextGrid / CSV writers: every tier covers [0, duration] without gaps or
   overlaps, labels and tier names as documented.
5. End to end on a tiny random latent-SIL model and 25 s of synthetic audio:
   load_model, transcribe (several chunks), writers, and the command line.
   Needs silero-vad (skipped without it).
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config

from reco_align import transcribe as TR

SR = 16000
torch.manual_seed(0)
rng = np.random.default_rng(0)


def covers(chunks, dur, max_chunk):
    assert abs(chunks[0][0]) < 1e-9 and abs(chunks[-1][1] - dur) < 1e-9, chunks
    for (a, b), (c, _) in zip(chunks[:-1], chunks[1:]):
        assert abs(b - c) < 1e-9, chunks                        # back to back
    assert all(0 < b - a <= max_chunk + 1e-9 for a, b in chunks), chunks


def test_plan_chunks():
    assert TR.plan_chunks([(0.5, 8.0)], 8.0) == [(0.0, 8.0)]   # short: one chunk
    # pauses 6.0-6.4 and 9.0-9.2: cut at the latest one, in its middle
    speech = [(0.2, 6.0), (6.4, 9.0), (9.2, 15.5), (16.0, 24.0)]
    ch = TR.plan_chunks(speech, 25.0, max_chunk=10.0)
    covers(ch, 25.0, 10.0)
    assert abs(ch[0][1] - 9.1) < 1e-9, ch
    assert abs(ch[1][1] - 15.75) < 1e-9, ch                     # next pause 15.5-16.0
    # a pause shorter than min_gap is not a cut place
    ch = TR.plan_chunks([(0.0, 6.0), (6.05, 30.0)], 30.0, 10.0, min_gap=0.1,
                        quietest=lambda lo, hi: lo + 1.0)
    covers(ch, 30.0, 10.0)
    assert abs(ch[0][1] - 6.0) < 1e-9, ch                       # quietest(5, 10) -> 6.0
    # a long pause covering the window end: cut inside the window, not after it
    ch = TR.plan_chunks([(0.0, 4.0), (40.0, 45.0)], 45.0, 10.0)
    covers(ch, 45.0, 10.0)
    assert abs(ch[0][1] - 10.0) < 1e-9, ch
    # no pause and no quietest: cut at max_chunk
    ch = TR.plan_chunks([(0.0, 25.0)], 25.0, 10.0)
    assert ch == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)], ch
    # random speech: always a valid cover
    for _ in range(200):
        dur = float(rng.uniform(1, 120))
        b = np.sort(rng.uniform(0, dur, 2 * int(rng.integers(0, 30))))
        speech = list(zip(b[::2], b[1::2]))
        m = float(rng.uniform(2, 30))
        covers(TR.plan_chunks(speech, dur, m, quietest=lambda lo, hi: (lo + hi) / 2), dur, m)
    print("plan_chunks ok")


def test_join_pauses():
    ph = [{"start": 1.0, "end": 1.2}, {"start": 1.6, "end": 1.7}]
    tol = 0.035
    assert TR.join_pauses([(1.2, 1.38), (1.40, 1.6)], ph, tol) == [(1.2, 1.6)]
    assert TR.join_pauses([(1.2, 1.3), (1.38, 1.6)], ph, tol) == [(1.2, 1.3), (1.38, 1.6)]   # gap 80 ms
    ph2 = ph + [{"start": 1.38, "end": 1.40}]                                              # phone between
    assert TR.join_pauses([(1.2, 1.38), (1.40, 1.6)], ph2, tol) == [(1.2, 1.38), (1.40, 1.6)]
    assert TR.join_pauses([(3.0, 3.5), (0.5, 0.9)], [], tol) == [(0.5, 0.9), (3.0, 3.5)]  # sorted
    print("join_pauses ok")


def test_candidates():
    post = np.array([0.5, 0.1, 0.2, 0.05, 0.1, 0.03, 0.02])   # index 0 = blank, not a symbol
    ids, labels = [1, 2, 3, 4, 5, 6], ["a", "e", "a", "i", "o", "u"]
    c = TR._candidates(post, ids, labels)
    assert [l for l, _ in c] == ["e", "a", "i", "o", "u"], c  # a = 0.10 + 0.05 summed, below e = 0.20
    assert abs(sum(p for _, p in c) - 1.0) < 1e-9
    assert all(c[i][1] >= c[i + 1][1] for i in range(len(c) - 1))
    assert len(TR._candidates(np.r_[0.0, np.ones(10)], list(range(1, 11)), list("abcdefghij"))) == TR.K
    print("candidates ok")


def read_textgrid(path):
    tg = open(path, encoding="utf-8").read()
    xmax = float(re.search(r"^xmax = ([\d.]+)", tg, re.M).group(1))
    tiers = {}
    for block in tg.split("item [")[2:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        tiers[name] = [(float(a), float(b), t.replace('""', '"')) for a, b, t in
                       re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "((?:[^"]|"")*)"', block)]
    return xmax, tiers


def check_outputs(tg_path, csv_path, phones, pauses, dur):
    xmax, tiers = read_textgrid(tg_path)
    assert abs(xmax - dur) < 1e-5
    assert list(tiers) == ["phones", "confidence", "top2", "top3", "top4", "top5", "pauses"], list(tiers)
    for name, iv in tiers.items():
        assert abs(iv[0][0]) < 1e-6 and abs(iv[-1][1] - dur) < 1e-5, name
        assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(iv[:-1], iv[1:])), name   # no gap / overlap
        assert all(b > a for a, b, _ in iv), name
    assert [t for _, _, t in tiers["phones"] if t] == [p["phone"] for p in phones]
    assert sum(t == "SIL" for _, _, t in tiers["pauses"]) == len(pauses)
    assert set(t for _, _, t in tiers["pauses"]) <= {"", "SIL"}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert len(rows) == len(phones)
    assert list(rows[0])[:5] == ["start", "end", "phone", "top1", "p1"] if rows else True
    for r, p in zip(rows, phones):
        assert r["phone"] == p["phone"] and abs(float(r["start"]) - p["start"]) < 1e-4


def test_writers():
    dur = 3.0
    phones = [{"phone": "b", "start": 0.30, "end": 0.38, "cands": [("b", 0.9), ("p", 0.05), ("d", 0.03),
                                                                    ("m", 0.01), ("v", 0.01)]},
              {"phone": "ɔ̃", "start": 0.38, "end": 0.55, "cands": [("ɔ̃", 0.6), ("o", 0.3), ("ɑ̃", 0.1)]},
              {"phone": 'a"', "start": 1.20, "end": 1.31, "cands": [('a"', 1.0)]}]
    pauses = [(0.0, 0.30), (0.55, 1.20), (1.31, 3.0)]
    with tempfile.TemporaryDirectory() as d:
        TR.write_textgrid(f"{d}/x.TextGrid", phones, pauses, dur)
        TR.write_csv(f"{d}/x.csv", phones)
        check_outputs(f"{d}/x.TextGrid", f"{d}/x.csv", phones, pauses, dur)
        _, tiers = read_textgrid(f"{d}/x.TextGrid")
        assert [t for _, _, t in tiers["top2"] if t] == ["p 0.050", "o 0.30"], tiers["top2"]
        assert [t for _, _, t in tiers["top4"] if t] == ["m 0.010"], tiers["top4"]
    assert TR.fmt_prob(0.5) == "0.50" and TR.fmt_prob(0.0004) == "<0.001"
    print("writers ok")


def tiny_model_dir(d):
    """A tiny random latent-SIL model (as tests/test_model.py) saved with a vocabulary."""
    from reco_align.model import Wav2Vec2ForCTCPoolContrastive
    cfg = Wav2Vec2Config(
        vocab_size=12, pad_token_id=0, hidden_size=32, num_hidden_layers=3,
        num_attention_heads=2, intermediate_size=64, conv_dim=(16,) * 7,
        conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 1),   # 10 ms frames
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        codevector_dim=16, proj_codevector_dim=16, num_codevectors_per_group=8,
        do_stable_layer_norm=True, feat_extract_norm="layer",
        ctc_loss_reduction="mean", ctc_zero_infinity=True, layerdrop=0.0,
        apply_spec_augment=False, mask_time_prob=0.0, mask_feature_prob=0.0)
    for k, v in dict(prior_scaling_factor=0.5, pool_mode="window", prior_type="beta_binomial",
                     window_sigma_frames=10.0, gt_span_source="forced_align",
                     convbank_kernels=[1, 5, 9], input_sampling_rate=SR, align_hidden_size=16,
                     n_negatives=10, contrastive_temperature=0.1, mask_prob_low=0.1,
                     mask_prob_high=0.5, num_codevectors_ema=32, ema_decay=0.99,
                     use_position_features=False, quantizer_assign="nearest",
                     detach_alignment_input=False, detach_contrastive_input=True,
                     joint_encoder_alpha=0.1, joint_lm_encoder_alpha=0.1, encoder_top_layers=2,
                     use_sil=True, sil_positions="word", sil_eval_positions="all", sil_logp=-2.0,
                     sil_logp_learned=True, sil_min_frames=4, lm_att_source="fs_posterior",
                     lm_mask_same_code=True, lm_energy_weight=1.0).items():
        setattr(cfg, k, v)
    m = Wav2Vec2ForCTCPoolContrastive(cfg)
    with torch.no_grad():                       # random weights decode mostly blank: favour phones
        m.lm_head.bias[0] = -3.0
        m.lm_head.bias[1:] = torch.randn(11)
    m.save_pretrained(d)
    vocab = {"[PAD]": 0, "[UNK]": 1, **{s: i for i, s in enumerate("abdeiklmo@", start=2)}}
    json.dump(vocab, open(os.path.join(d, "vocab.json"), "w"))


def test_end_to_end():
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        print("end to end SKIPPED (pip install silero-vad)")
        return
    with tempfile.TemporaryDirectory() as d:
        tiny_model_dir(f"{d}/model")
        # 25 s: noise bursts ("speech") with pauses, at 22.05 kHz stereo (resampled + downmixed)
        sr = 22050
        t = np.arange(int(25 * sr)) / sr
        x = 0.002 * rng.standard_normal(len(t))
        for s in np.arange(0.5, 24.0, 2.5):
            on = (t >= s) & (t < s + 1.8)
            x[on] += 0.3 * np.sin(2 * np.pi * 180 * t[on]) * rng.uniform(0.5, 1, on.sum())
        sf.write(f"{d}/rec.wav", np.stack([x, x], 1).astype(np.float32), sr)
        model, fe, tok, device = TR.load_model(f"{d}/model", device="cpu")
        phones, pauses, dur = TR.transcribe(f"{d}/rec.wav", model, fe, tok, device, 10.0,
                                            dict(TR.MODEL2IPA))
        assert abs(dur - 25.0) < 1e-3, dur
        assert phones, "no phone decoded"
        assert all(0 <= p["start"] < p["end"] <= dur + 0.05 for p in phones)
        assert all(a["start"] <= b["start"] for a, b in zip(phones[:-1], phones[1:]))
        for p in phones:                        # top 5 of a distribution over the 10 phone symbols
            v = [x for _, x in p["cands"]]
            assert len(v) == TR.K and sum(v) <= 1 + 1e-6 and v == sorted(v, reverse=True), p
        assert "@" not in {p["phone"] for p in phones}             # IPA: @ -> ɑ̃
        assert all(a[1] <= b[0] for a, b in zip(pauses[:-1], pauses[1:]))
        TR.write_textgrid(f"{d}/rec.TextGrid", phones, pauses, dur)
        TR.write_csv(f"{d}/rec.csv", phones)
        check_outputs(f"{d}/rec.TextGrid", f"{d}/rec.csv", phones, pauses, dur)
        # command line, as installed (`reco-align-transcribe`) and from the repository root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "transcribe.py"), "--model", f"{d}/model",
                            "--device", "cpu", "--symbols", "model", "-o", f"{d}/out", d],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        assert os.path.exists(f"{d}/out/rec.TextGrid") and os.path.exists(f"{d}/out/rec.csv")
        r = subprocess.run([sys.executable, os.path.join(root, "transcribe.py"), "--model", f"{d}/model",
                            "-o", f"{d}/out", f"{d}/model/vocab.json"], capture_output=True, text=True, cwd=root)
        assert r.returncode != 0 and "unsupported audio format" in r.stderr, r.stderr[-500:]
    print(f"end to end ok ({len(phones)} phones, {len(pauses)} pauses)")


if __name__ == "__main__":
    test_plan_chunks()
    test_join_pauses()
    test_candidates()
    test_writers()
    test_end_to_end()
    print("ALL OK")
