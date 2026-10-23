#!/usr/bin/env python3
"""CPU tests of jointrecalign.transcribe (no data, no pretrained weights):

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
   load_model, transcribe (several chunks), writers, and the command line
   (validate, default results folder, a damaged file among good ones, summary,
   log, HTML report, comparison with a reference TextGrid).
   Needs silero-vad (skipped without it).
6. Model download on first use (from a local zip), then reuse of the cache.
7. Praat plugin installation.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config

from jointrecalign import transcribe as TR

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
    from jointrecalign.model import Wav2Vec2ForCTCPoolContrastive
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
        # command line, as installed (`JointRecAlign`) and from the repository root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "transcribe.py"), "--model", f"{d}/model",
                            "--device", "cpu", "--symbols", "model", "-o", f"{d}/out", d],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        assert os.path.exists(f"{d}/out/rec.TextGrid") and os.path.exists(f"{d}/out/rec.csv")
        r = subprocess.run([sys.executable, os.path.join(root, "transcribe.py"), "--model", f"{d}/model",
                            "-o", f"{d}/out", f"{d}/model/vocab.json"], capture_output=True, text=True, cwd=root)
        assert r.returncode == 1 and "not a supported audio file" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
        # `JointRecAlign transcribe` (python -m jointrecalign) without -o: results in JointRecAlign/ next to
        # the recordings; a damaged file is reported by the check and the others are still transcribed;
        # --report: one HTML page per recording and index.html
        open(f"{d}/damaged.wav", "wb").write(b"not audio")
        r = subprocess.run([sys.executable, "-m", "jointrecalign", "transcribe", "--model", f"{d}/model",
                            "--device", "cpu", "--report", d], capture_output=True, text=True, cwd=root)
        out = r.stdout
        assert r.returncode == 1, out + r.stderr
        assert "damaged.wav: cannot read the audio" in out and "1 NOT transcribed" in out, out
        assert "rec.wav: 2 channels" in out, out                              # warning from the check
        res = f"{d}/JointRecAlign"
        for f in ("rec.TextGrid", "rec.csv", "rec.html", "index.html", "JointRecAlign.log"):
            assert os.path.exists(f"{res}/{f}"), f
        assert not os.path.exists(f"{res}/damaged.TextGrid")
        assert "1 NOT transcribed" in open(f"{res}/JointRecAlign.log", encoding="utf-8").read()
        page = open(f"{res}/rec.html", encoding="utf-8").read()
        data = json.loads(re.search(r"const D = (\{.*?\});\n", page).group(1))
        assert data["audio"] == "../rec.wav" and abs(data["duration"] - dur) < 1e-3, (data["audio"], data["duration"])
        assert len(data["phones"]) == len(phones) and data["rows"][0] == "blank" and data["tokens"], data["rows"]
        assert sum(c["T"] for c in data["chunks"]) > 0.9 * dur / data["hop"]
        idx = json.loads(re.search(r"const D = (\{.*?\});\n", open(f"{res}/index.html", encoding="utf-8").read()).group(1))
        assert [x["name"] for x in idx["rows"]] == ["rec"] and idx["failed"][0][0] == "damaged.wav", idx
        # --reference: a reference TextGrid for rec (the transcription's own phones, the first one
        # changed), comparison.csv, the reference in the page and the index; a reference that the
        # results would overwrite is refused
        os.makedirs(f"{d}/ref")
        ref_ph = [dict(p) for p in phones]
        ref_ph[0]["phone"] = "zz"
        TR.write_textgrid(f"{d}/ref/rec.TextGrid", ref_ph, pauses, dur)
        r = subprocess.run([sys.executable, "-m", "jointrecalign", "transcribe", "--model", f"{d}/model",
                            "--device", "cpu", "--report", "--reference", f"{d}/ref", "-o", f"{d}/cmp",
                            f"{d}/rec.wav"], capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "compared with the reference: 1 of 1" in r.stdout and "'zz' (1)" in r.stdout, r.stdout
        rows = list(csv.DictReader(open(f"{d}/cmp/comparison.csv", encoding="utf-8")))
        assert [x["recording"] for x in rows] == ["rec", "TOTAL"] and rows[0]["S"] == "1", rows
        assert int(rows[0]["n_ref"]) == len(phones) and rows[0]["D"] == "0" and rows[0]["I"] == "0", rows
        page = json.loads(re.search(r"const D = (\{.*?\});\n", open(f"{d}/cmp/rec.html", encoding="utf-8").read()).group(1))
        assert page["ref"]["phones"][0][2:] == ["zz", "sub"] and page["ref"]["hyp_status"][0] == "sub", page["ref"]["phones"][:2]
        assert page["ref"]["scores"]["F1@20"] == 100.0, page["ref"]["scores"]
        idx = json.loads(re.search(r"const D = (\{.*?\});\n", open(f"{d}/cmp/index.html", encoding="utf-8").read()).group(1))
        assert idx["total"]["F1@20"] == 100.0 and idx["rows"][0]["PER"] is not None, idx
        r = subprocess.run([sys.executable, "-m", "jointrecalign", "transcribe", "--model", f"{d}/model",
                            "--device", "cpu", "--reference", f"{d}/ref", "-o", f"{d}/ref", f"{d}/rec.wav"],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 1 and "would overwrite the reference" in r.stdout, r.stdout + r.stderr
        assert open(f"{d}/ref/rec.TextGrid", encoding="utf-8").read().count('"zz"') == 1    # left as it was
        # validate alone: no model, same check
        r = subprocess.run([sys.executable, "-m", "jointrecalign", "validate", d], capture_output=True, text=True, cwd=root)
        assert r.returncode == 1 and "1 can be transcribed, 1 cannot" in r.stdout, r.stdout + r.stderr
        # a read-only recordings folder (a shared corpus) without -o: refused by the check, before the model
        os.makedirs(f"{d}/ro")
        shutil.copy(f"{d}/rec.wav", f"{d}/ro/rec.wav")
        os.chmod(f"{d}/ro", 0o555)
        try:
            r = subprocess.run([sys.executable, "-m", "jointrecalign", "transcribe", "--model", f"{d}/model",
                                "--device", "cpu", f"{d}/ro"], capture_output=True, text=True, cwd=root)
        finally:
            os.chmod(f"{d}/ro", 0o755)
        assert r.returncode == 1 and "no permission to write the results" in r.stdout, r.stdout + r.stderr
        assert "Loading the model" not in r.stdout and not os.path.exists(f"{d}/ro/JointRecAlign"), r.stdout
        # --copy_audio: the page plays a copy in <results>/audio/, so the results folder can be moved
        r = subprocess.run([sys.executable, "-m", "jointrecalign", "transcribe", "--model", f"{d}/model",
                            "--device", "cpu", "--report", "--copy_audio", "-o", f"{d}/moved", f"{d}/rec.wav"],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stdout + r.stderr
        assert open(f"{d}/moved/audio/rec.wav", "rb").read() == open(f"{d}/rec.wav", "rb").read()
        page = json.loads(re.search(r"const D = (\{.*?\});\n", open(f"{d}/moved/rec.html", encoding="utf-8").read()).group(1))
        assert page["audio"] == "audio/rec.wav", page["audio"]
    print(f"end to end ok ({len(phones)} phones, {len(pauses)} pauses)")


def test_download():
    """First use: the model zip is fetched, unzipped into the cache, then reused."""
    import zipfile
    from jointrecalign import download as DL
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/src/{DL.MODEL_NAME}")
        open(f"{d}/src/{DL.MODEL_NAME}/model.safetensors", "wb").write(os.urandom(3_000_000))
        with zipfile.ZipFile(f"{d}/m.zip", "w") as z:
            z.write(f"{d}/src/{DL.MODEL_NAME}/model.safetensors", f"{DL.MODEL_NAME}/model.safetensors")
        env = dict(os.environ)
        os.environ.update(JOINTRECALIGN_HOME=f"{d}/cache", JOINTRECALIGN_MODEL_URL=f"file://{d}/m.zip")
        try:
            lines = []
            folder = DL.default_model(lines.append)
            assert folder == f"{d}/cache/{DL.MODEL_NAME}", folder
            assert open(f"{folder}/model.safetensors", "rb").read() == \
                open(f"{d}/src/{DL.MODEL_NAME}/model.safetensors", "rb").read()
            assert sorted(os.listdir(f"{d}/cache")) == [DL.MODEL_NAME], os.listdir(f"{d}/cache")
            assert any("100 %" in x for x in lines), lines
            lines = []
            assert DL.default_model(lines.append) == folder and not lines    # second use: no download
            os.environ.update(JOINTRECALIGN_HOME=f"{d}/cache2", JOINTRECALIGN_MODEL_URL=f"file://{d}/none.zip")
            try:
                DL.default_model(lines.append)
                raise AssertionError("no error on a failed download")
            except RuntimeError as e:
                assert "could not download" in str(e), e
        finally:
            os.environ.clear()
            os.environ.update(env)
    print("download ok")


def test_praat_plugin():
    """The plugin folder gets setup.praat and transcribe.praat calling this Python."""
    from jointrecalign import praat
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ)
        os.environ["PRAAT_PREFS_DIR"] = d
        try:
            praat.install_plugin()
        finally:
            os.environ.clear()
            os.environ.update(env)
        dest = f"{d}/{praat.PLUGIN}"
        assert sorted(os.listdir(dest)) == ["setup.praat", "transcribe.praat"], os.listdir(dest)
        script = open(f"{dest}/transcribe.praat", encoding="utf-8").read()
        assert "@COMMAND@" not in script and f"{sys.executable} -m jointrecalign" in script.replace('""', "")
        assert '"transcribe.praat"' in open(f"{dest}/setup.praat").read()
    print("praat plugin ok")


if __name__ == "__main__":
    test_plan_chunks()
    test_join_pauses()
    test_candidates()
    test_writers()
    test_end_to_end()
    test_download()
    test_praat_plugin()
    print("ALL OK")
