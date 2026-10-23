#!/usr/bin/env python3
"""Tests of jointrecalign.reference (no model): python tests/test_reference.py

1. TextGrid reading: long and short text formats, UTF-8 and UTF-16, quotes in
   labels, point tiers skipped, tier choice.
2. Reference labels: pauses (adjacent ones merged, empty intervals always
   pauses), skipped labels, --ref_map.
3. Comparison on a hand-made case: PER counts, boundary F1 counts, matched
   boundary errors, pauses, per-phone status; pooled scores.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jointrecalign import reference as RF   # noqa: E402

LONG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 3
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 2
            text = "bonjour ""toi"""
    item [2]:
        class = "TextTier"
        name = "events"
        xmin = 0
        xmax = 2
        points: size = 1
        points [1]:
            number = 0.5
            mark = "click"
    item [3]:
        class = "IntervalTier"
        name = "Phones"
        xmin = 0
        xmax = 2
        intervals: size = 8
        intervals [1]:
            xmin = 0
            xmax = 0.3
            text = ""
        intervals [2]:
            xmin = 0.3
            xmax = 0.4
            text = "b"
        intervals [3]:
            xmin = 0.4
            xmax = 0.55
            text = "o~"
        intervals [4]:
            xmin = 0.55
            xmax = 0.7
            text = "#"
        intervals [5]:
            xmin = 0.7
            xmax = 0.9
            text = "sil"
        intervals [6]:
            xmin = 0.9
            xmax = 1.0
            text = "ʒ"
        intervals [7]:
            xmin = 1.0
            xmax = 1.2
            text = "[noise]"
        intervals [8]:
            xmin = 1.2
            xmax = 2
            text = "u"
'''

SHORT = '''File type = "ooTextFile"
Object class = "TextGrid"

0
1
<exists>
1
"IntervalTier"
"x"
0
1
2
0
0.5
"a"
0.5
1
""
'''


def test_read():
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/long.TextGrid", "w", encoding="utf-8").write(LONG)
        open(f"{d}/short.TextGrid", "w", encoding="utf-16").write(SHORT)
        t = RF.read_textgrid(f"{d}/long.TextGrid")
        assert list(t) == ["words", "Phones"], list(t)                    # the point tier is skipped
        assert t["words"] == [(0.0, 2.0, 'bonjour "toi"')], t["words"]
        assert len(t["Phones"]) == 8 and t["Phones"][2] == (0.4, 0.55, "o~")
        s = RF.read_textgrid(f"{d}/short.TextGrid")
        assert s == {"x": [(0.0, 0.5, "a"), (0.5, 1.0, "")]}, s
        assert RF.pick_tier(t) == "Phones" and RF.pick_tier(s) == "x" and RF.pick_tier(t, "words") == "words"
        try:
            RF.pick_tier(t, "phon")
            raise AssertionError("no error for a missing tier")
        except ValueError as e:
            assert "words, Phones" in str(e), e
        os.makedirs(f"{d}/ref")
        open(f"{d}/ref/rec.textgrid", "w").write(SHORT)
        assert RF.find_reference("/x/rec.wav", f"{d}/ref") == f"{d}/ref/rec.textgrid"
        assert RF.find_reference("/x/other.wav", f"{d}/ref") is None
    print("TextGrid reading ok")


def test_load():
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/r.TextGrid", "w", encoding="utf-8").write(LONG)
        r = RF.load_reference(f"{d}/r.TextGrid", ref_map={"o~": "ɔ̃"}, skip_labels=["[noise]"])
        assert r["tier"] == "Phones" and r["end"] == 2.0
        assert [p["phone"] for p in r["phones"]] == ["b", "ɔ̃", "ʒ", "u"], r["phones"]
        assert r["pauses"] == [(0.0, 0.3), (0.55, 0.9)], r["pauses"]      # "#" + "sil" merged
        assert r["skipped"] == [(1.0, 1.2)]
        r = RF.load_reference(f"{d}/r.TextGrid", pause_labels=["sil"])
        assert r["pauses"] == [(0.0, 0.3), (0.7, 0.9)] and "#" in [p["phone"] for p in r["phones"]]
    print("reference labels ok")


def test_compare():
    ph = lambda *xs: [{"phone": p, "start": s, "end": e} for p, s, e in xs]
    ref = {"phones": ph(("b", 0.30, 0.40), ("ɔ̃", 0.40, 0.55), ("ʒ", 0.90, 1.00), ("u", 1.00, 1.20)),
           "pauses": [(0.0, 0.3), (0.55, 0.9)]}
    # b right, ɔ̃ -> o (substitution), ʒ 30 ms late, u deleted, an inserted l
    hyp = ph(("b", 0.31, 0.40), ("o", 0.40, 0.56), ("ʒ", 0.93, 1.03), ("l", 1.03, 1.20))
    c, m = RF.compare(ref, hyp, 1.2)
    assert (c["n_ref"], c["n_hyp"]) == (4, 4)
    assert c["S"] + c["D"] + c["I"] == 2 and c["S"] == 2, c                  # ɔ̃->o, u->l
    # boundaries ref {0.30 0.40 0.55 0.90 1.00 1.20}, hyp {0.31 0.40 0.56 0.93 1.03 1.20}
    assert (c["tp20"], c["fp20"], c["fn20"]) == (4, 2, 2), c
    assert (c["tp50"], c["fp50"], c["fn50"]) == (6, 0, 0), c
    assert sorted(round(e) for e in c["errors_ms"]) == [0, 10, 30, 30], c["errors_ms"]   # b and ʒ
    assert m["ref_status"] == ["ok", "sub", "ok", "sub"] and m["hyp_status"] == ["ok", "sub", "ok", "sub"], m
    s = RF.scores(c)
    assert abs(s["PER"] - 50.0) < 1e-9 and abs(s["F1@20"] - 100 * 4 / 6) < 1e-9 and s["F1@50"] == 100.0, s
    assert abs(s["MedianBE"] - 20.0) < 1e-6, s
    assert s["pause_ref"] == 2 and s["pause_hyp"] == 2 and s["pause_F1"] == 100.0, s   # [0, .31] and [.56, .93]
    tot = RF.scores([c, c])
    assert tot["n_ref"] == 8 and abs(tot["PER"] - 50.0) < 1e-9
    # a deletion; an insertion
    c, m = RF.compare(ref, ph(("b", 0.3, 0.4), ("ʒ", 0.9, 1.0), ("u", 1.0, 1.2)), 1.2)
    assert m["ref_status"] == ["ok", "del", "ok", "ok"] and m["hyp_status"] == ["ok", "ok", "ok"], m
    assert (c["S"], c["D"], c["I"]) == (0, 1, 0), c
    c, m = RF.compare(ref, ph(("b", .3, .4), ("ɔ̃", .4, .55), ("ʒ", .9, 1.), ("u", 1., 1.1), ("l", 1.1, 1.2)), 1.2)
    assert m["ref_status"] == ["ok"] * 4 and m["hyp_status"] == ["ok"] * 4 + ["ins"], m
    assert (c["S"], c["D"], c["I"]) == (0, 0, 1), c
    with tempfile.TemporaryDirectory() as d:
        RF.write_csv(f"{d}/c.csv", [("a", s), ("TOTAL", tot)])
        lines = open(f"{d}/c.csv", encoding="utf-8").read().splitlines()
        assert lines[0].startswith("recording,n_ref") and lines[2].startswith("TOTAL,8"), lines
    print("comparison ok")


if __name__ == "__main__":
    test_read()
    test_load()
    test_compare()
    print("ALL OK")
