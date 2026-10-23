"""JointRecAlign: phones, boundaries, pauses and the top-5 candidates of each phone
of French recordings, as Praat TextGrids. No transcript is needed.

    JointRecAlign validate recordings/              check the audio files, without the model
    JointRecAlign transcribe recordings/ [-o out/]  check, then transcribe
    JointRecAlign transcribe recordings/ --report   also an HTML page per recording (plots)
    JointRecAlign transcribe recordings/ --reference textgrids/
                                                    compare with reference phone TextGrids
    JointRecAlign download                          download the model now (else: on first use)
    JointRecAlign install-praat-plugin              a "JointRecAlign: transcribe..." button in Praat

Results go to a folder JointRecAlign/ next to the recordings, unless -o gives
another folder. For each recording:
    <name>.TextGrid   tiers: phones, confidence, top2, top3, top4, top5, pauses
    <name>.csv        one row per phone: start, end, phone, top1..top5 and their probabilities
    <name>.html       with --report: waveform, phones, attention, posteriors, top 5 (index.html lists them)
and JointRecAlign.log, the summary of the run; with --reference, comparison.csv
(PER, boundary F1, boundary errors, pause scores per recording and for the corpus;
see jointrecalign/reference.py). The exit code is 1 when a recording could not be
transcribed.

Audio: WAV, FLAC, MP3, OGG or AIFF, any sampling rate, mono or stereo.

The probabilities are the model's own (not calibrated): the top-1 is often given
about 0.99 even when it is wrong, but the order of the candidates is informative.
See jointrecalign/transcribe.py for the method.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")
OUT_NAME = "JointRecAlign"
COMMANDS = ("validate", "transcribe", "download", "install-praat-plugin")


def find_audio(inputs):
    """Audio files among files and folders -> (files, problems)."""
    files, problems = [], []
    for x in inputs:
        if os.path.isdir(x):
            found = sorted(os.path.join(x, f) for f in os.listdir(x)
                           if f.lower().endswith(AUDIO) and not f.startswith("."))
            if not found:
                problems.append(f"no audio file in folder {x}")
            files += found
        elif os.path.isfile(x):
            if x.lower().endswith(AUDIO):
                files.append(x)
            else:
                problems.append(f"not a supported audio file: {x} (use {', '.join(AUDIO)})")
        else:
            problems.append(f"not found: {x}")
    return files, problems


def out_dir_for(path, out_dir=None):
    """Where the results of a recording go: out_dir, or JointRecAlign/ next to the recording."""
    return out_dir or os.path.join(os.path.dirname(os.path.abspath(path)), OUT_NAME)


def check_file(path):
    """-> [(level, message)] for one recording; level "error" (cannot be transcribed) or "warning"."""
    import numpy as np
    import soundfile as sf
    try:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return [("error", f"cannot read the audio ({str(e).splitlines()[0] if str(e) else type(e).__name__})")]
    out, dur = [], len(a) / sr
    if dur < 0.5:
        return [("error", f"too short ({dur:.2f} s)")]
    if float(np.abs(a).max()) < 1e-4:
        return [("error", "silent")]
    if sr < 16000:
        out.append(("warning", f"sampled at {sr} Hz, below the model's 16 kHz: expect lower accuracy"))
    if a.shape[1] > 1:
        out.append(("warning", f"{a.shape[1]} channels: they are averaged into one"))
    clip = float((np.abs(a) >= 0.999).mean())
    if clip > 1e-3:
        out.append(("warning", f"clipped ({100 * clip:.1f} % of the samples at full scale)"))
    if float(np.sqrt((a ** 2).mean())) < 1e-3:
        out.append(("warning", "very quiet recording"))
    return out


def check_reference(path, out_dir, ref):
    """-> [(level, message)] about the reference TextGrid of a recording (ref: the --reference options)."""
    import soundfile as sf
    from . import reference as RF
    name = os.path.splitext(os.path.basename(path))[0]
    rp = RF.find_reference(path, ref["folder"])
    if rp is None:
        return [("warning", f"no reference {name}.TextGrid in {ref['folder']}: not compared")]
    if os.path.abspath(rp) == os.path.abspath(os.path.join(out_dir_for(path, out_dir), name + ".TextGrid")):
        return [("error", f"the results would overwrite the reference {rp}: choose another -o")]
    try:
        r = RF.load_reference(rp, ref["tier"], ref["map"], ref["pause_labels"], ref["skip_labels"])
    except Exception as e:
        return [("warning", f"reference {rp} not usable ({e}): not compared")]
    out = []
    if not r["phones"]:
        out.append(("warning", f"reference tier {r['tier']!r} of {rp} has no phone"))
    try:
        dur = sf.info(path).duration
        if abs(r["end"] - dur) > 1.0:
            out.append(("warning", f"reference {rp} ends at {r['end']:.1f} s, the audio at {dur:.1f} s: "
                                   "is it the right TextGrid?"))
    except Exception:
        pass
    return out


def validate(files, out_dir=None, log=print, ref=None):
    """Check every file (and its reference TextGrid, with ref); -> {path: [(level, message)]}
    for the files with a problem."""
    issues, seen = {}, {}
    for path in files:
        found = check_file(path)
        if ref and not any(lv == "error" for lv, _ in found):
            found += check_reference(path, out_dir, ref)
        key = (out_dir_for(path, out_dir), os.path.splitext(os.path.basename(path))[0].lower())
        if key in seen:
            found.append(("error", f"same name as {seen[key]}: its results would be overwritten "
                                   "(rename one of them, or give each folder its own -o)"))
        else:
            seen[key] = path
        if found:
            issues[path] = found
    n_err = sum(any(lv == "error" for lv, _ in v) for v in issues.values())
    n_warn = sum(1 for v in issues.values() for lv, _ in v if lv == "warning")
    for path, found in issues.items():
        for level, msg in found:
            log(f"  {level.upper():7s} {path}: {msg}")
    log(f"Checked {len(files)} recordings: {len(files) - n_err} can be transcribed"
        + (f", {n_err} cannot" if n_err else "") + (f", {n_warn} warnings" if n_warn else ""))
    return issues


def symbol_map_for(symbols="ipa", phone_map=None):
    from . import transcribe as TR
    symbol_map = dict(TR.MODEL2IPA) if symbols == "ipa" else {}
    if phone_map:
        user = json.load(open(phone_map, encoding="utf-8"))
        symbol_map = {k: user.get(v, v) for k, v in symbol_map.items()}   # IPA first, then the user's labels
        symbol_map.update({k: v for k, v in user.items() if k not in symbol_map})
    return symbol_map


class Log:
    """Prints, and keeps the lines for the JointRecAlign.log files."""
    def __init__(self):
        self.lines = []

    def __call__(self, line):
        print(line, flush=True)
        self.lines.append(line)


def run(files, out_dir=None, model=None, symbols="ipa", phone_map=None, max_chunk=10.0, device=None,
        log=print, vocab_path=None, report=False, rejected=(), ref=None):
    """Transcribe the files; one failing file does not stop the others. rejected: [(file, reason)]
    refused before (listed in the report index); ref: the --reference options.
    -> (written TextGrids, failures [(file, reason)], comparison counts [(file, counts)],
        reference labels outside the model's phones {label: count})."""
    warnings.filterwarnings("ignore")
    # a harmless torch notice on macOS ("Redirects are currently not supported")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers.utils import logging as hf_logging
    from . import transcribe as TR
    from .download import default_model
    hf_logging.disable_progress_bar()
    symbol_map = symbol_map_for(symbols, phone_map)
    model = model or default_model(log)
    log("Loading the model")
    m, fe, tok, dev = TR.load_model(model, vocab_path, device)
    log(f"Running on {'the GPU' if dev == 'cuda' else 'the CPU'}")
    done, failed, pages, compared, unknown = [], [], {}, [], {}
    if ref:
        from . import reference as RF
        from .report import vocab_rows
        known = {l for _, l in vocab_rows(tok, symbol_map)[0][1:]} | {symbol_map.get(v, v) for v in TR.E.HYP_FIX.values()}
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        t = time.time()
        d = out_dir_for(path, out_dir)
        try:
            trace = [] if report else None
            phones, pauses, dur = TR.transcribe(path, m, fe, tok, dev, max_chunk, symbol_map, trace)
            os.makedirs(d, exist_ok=True)
            tg = os.path.join(d, name + ".TextGrid")
            TR.write_textgrid(tg, phones, pauses, dur)
            TR.write_csv(os.path.join(d, name + ".csv"), phones)
            cmp_ = None
            rp = RF.find_reference(path, ref["folder"]) if ref else None
            if rp:
                try:
                    r = RF.load_reference(rp, ref["tier"], ref["map"], ref["pause_labels"], ref["skip_labels"])
                    counts, match = RF.compare(r, phones, dur)
                    compared.append((path, counts))
                    for p in r["phones"]:
                        if p["phone"] not in known:
                            unknown[p["phone"]] = unknown.get(p["phone"], 0) + 1
                    cmp_ = (r, match, RF.scores(counts))
                except Exception:
                    cmp_ = None                       # reported by the check
            if report:
                from . import report as R
                R.write_page(os.path.join(d, name + ".html"), path, phones, pauses, dur, trace, tok, symbol_map,
                             cmp_)
                pages.setdefault(d, []).append(R.summary(name, phones, pauses, dur, cmp_ and cmp_[2]))
        except Exception as e:
            failed.append((path, str(e).splitlines()[0] if str(e) else type(e).__name__))
            log(f"[{i}/{len(files)}] {name}: FAILED ({failed[-1][1]})")
            continue
        done.append(tg)
        msg = (f"[{i}/{len(files)}] {name}: {dur:.1f} s of audio, {len(phones)} phones, {len(pauses)} pauses "
               f"({time.time() - t:.1f} s)")
        if cmp_:
            s = cmp_[2]
            msg += f"; vs reference: PER {_pc(s['PER'])}, F1@20ms {_pc(s['F1@20'])}, pause F1 {_pc(s['pause_F1'])}"
        log(msg)
    if report:
        from . import report as R
        for d in sorted({out_dir_for(p, out_dir) for p in files} | {out_dir_for(p, out_dir) for p, _ in rejected}):
            fails = [(p, w) for p, w in list(rejected) + failed if out_dir_for(p, out_dir) == d]
            mine = [c for p, c in compared if out_dir_for(p, out_dir) == d]
            if d in pages or fails:
                os.makedirs(d, exist_ok=True)
                R.write_index(os.path.join(d, "index.html"), pages.get(d, []), fails,
                              RF.scores(mine) if mine else None)
    if compared:
        from . import reference as RF
        for d in sorted({out_dir_for(p, out_dir) for p, _ in compared}):
            mine = [(p, c) for p, c in compared if out_dir_for(p, out_dir) == d]
            rows = [(os.path.splitext(os.path.basename(p))[0], RF.scores(c)) for p, c in mine]
            RF.write_csv(os.path.join(d, "comparison.csv"), rows + [("TOTAL", RF.scores([c for _, c in mine]))])
    return done, failed, compared, unknown


def _pc(v, unit="%"):
    return "-" if v is None else (f"{v:.1f} {unit}" if unit else f"{v:.1f}")


def transcribe_command(a, log):
    files, problems = find_audio(a.inputs)
    for p in problems:
        log(f"  ERROR   {p}")
    if not files:
        log("No audio file to transcribe.")
        return 1
    ref = ref_options(a)
    log(f"Checking {len(files)} recordings" + (f" and their reference TextGrids in {a.reference}" if ref else ""))
    issues = validate(files, a.out_dir, log, ref)
    bad = {p for p, v in issues.items() if any(lv == "error" for lv, _ in v)}
    todo = [p for p in files if p not in bad]
    rejected = [(p, "; ".join(m for lv, m in issues[p] if lv == "error")) for p in files if p in bad]
    done, failed, compared, unknown = [], [], [], {}
    if todo:
        t = time.time()
        done, failed, compared, unknown = run(todo, a.out_dir, a.model, a.symbols, a.phone_map, a.max_chunk,
                                              a.device, log, a.vocab_path, a.report, rejected, ref)
        log(f"Time: {time.time() - t:.0f} s")
    failed = rejected + failed
    log("")
    log("Summary")
    log(f"  {len(files)} recordings: {len(done)} transcribed" + (f", {len(failed)} NOT transcribed" if failed else ""))
    for p, why in failed:
        log(f"  not transcribed: {p}: {why}")
    for p, v in issues.items():
        for lv, msg in v:
            if lv == "warning":
                log(f"  warning: {p}: {msg}")
    if ref:
        from . import reference as RF
        s = RF.scores([c for _, c in compared]) if compared else None
        log(f"  compared with the reference: {len(compared)} of {len(done)} transcribed recordings")
        if s:
            log(f"    PER {_pc(s['PER'])} ({s['S']} substitutions, {s['D']} deletions, {s['I']} insertions, "
                f"{s['n_ref']} reference phones)")
            log(f"    boundaries: F1 {_pc(s['F1@20'])} within 20 ms, {_pc(s['F1@50'])} within 50 ms; "
                f"median error {_pc(s['MedianBE'], 'ms')}")
            log(f"    pauses: P {_pc(s['pause_P'])}, R {_pc(s['pause_R'])}, F1 {_pc(s['pause_F1'])} "
                f"({s['pause_ref']} reference pauses)")
        if unknown:
            top = ", ".join(f"{k!r} ({n})" for k, n in sorted(unknown.items(), key=lambda x: -x[1])[:12])
            log(f"  warning: reference labels that are not phones of the model: {top}. They count as errors; "
                "map them with --ref_map, or list non-phone labels in --pause_labels / --skip_labels")
    made = [d for d in sorted({out_dir_for(p, a.out_dir) for p in files}) if os.path.isdir(d)]
    for d in made:
        log(f"  results: {d}" + (f"  (report: {os.path.join(d, 'index.html')})" if a.report else "")
            + ("  (comparison.csv)" if os.path.exists(os.path.join(d, "comparison.csv")) and compared else ""))
    for d in made:
        with open(os.path.join(d, "JointRecAlign.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(log.lines) + "\n")
    return 1 if failed or problems else 0


def ref_options(a):
    """The --reference options as a dict, or None."""
    if not getattr(a, "reference", None):
        return None
    from . import reference as RF
    if not os.path.isdir(a.reference):
        raise ValueError(f"reference folder not found: {a.reference}")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    return {"folder": a.reference, "tier": a.ref_tier, "map": RF.load_map(a.ref_map),
            "pause_labels": split(a.pause_labels) if a.pause_labels is not None else RF.PAUSE_LABELS,
            "skip_labels": split(a.skip_labels) if a.skip_labels else ()}


def parser():
    ap = argparse.ArgumentParser(prog="JointRecAlign", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", metavar="command")
    for name, help_ in (("validate", "check the audio files (readable, duration, sampling rate, clipping...)"),
                        ("transcribe", "check, then transcribe the recordings")):
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("inputs", nargs="+", help="audio files or folders of audio files")
        p.add_argument("-o", "--out_dir", default=None,
                       help=f"folder for the results (default: a folder {OUT_NAME}/ next to each recording)")
        g = p.add_argument_group("comparison with a reference (see jointrecalign/reference.py)")
        g.add_argument("--reference", default=None, metavar="FOLDER",
                       help="folder of the reference phone TextGrids, <name>.TextGrid for <name>.wav")
        g.add_argument("--ref_tier", default=None,
                       help="tier of the reference phones (default: phones / phonemes / ..., else the first)")
        g.add_argument("--ref_map", default=None, metavar="JSON",
                       help='JSON {"reference label": "model phone"}, e.g. {"ø": "ə"} or {"a~": "ɑ̃"}')
        g.add_argument("--pause_labels", default=None,
                       help="comma-separated labels that mean a pause, besides empty intervals "
                            "(default: SIL, sil, sp, spn, _, #, <p:>, ...)")
        g.add_argument("--skip_labels", default=None,
                       help="comma-separated labels left out of the comparison (noise, overlap...)")
        if name == "validate":
            continue
        p.add_argument("--report", action="store_true",
                       help="also write an HTML page per recording with the plots, and index.html")
        p.add_argument("--symbols", choices=("ipa", "model"), default="ipa",
                       help="ipa: nasal vowels and œ in IPA (ɑ̃ ɛ̃ ɔ̃ œ); model: the model's one-character codes")
        p.add_argument("--phone_map", default=None,
                       help='JSON {"model phone": "your label"} applied last, e.g. {"ə": "ø"}')
        p.add_argument("--model", default=None,
                       help="model folder (default: the released model, downloaded the first time)")
        p.add_argument("--max_chunk", type=float, default=10.0,
                       help="longest piece the model sees at once (s); cuts are placed in pauses found by Silero VAD")
        p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
        p.add_argument("--vocab_path", default=None, help="phone vocabulary (default: vocab.json in the model folder)")
    sub.add_parser("download", help="download the model now (otherwise it is downloaded on first use)")
    sub.add_parser("install-praat-plugin", help='add a "JointRecAlign: transcribe..." button for Sounds in Praat')
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy = {"--download": "download", "--install-praat-plugin": "install-praat-plugin"}
    if argv and argv[0] in legacy:
        argv[0] = legacy[argv[0]]
    elif argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["transcribe"] + argv              # `JointRecAlign recordings/` = transcribe
    ap = parser()
    a = ap.parse_args(argv)
    if not a.command:
        ap.print_help()
        return
    log = Log()
    try:
        if a.command == "install-praat-plugin":
            from .praat import install_plugin
            install_plugin()
            return
        if a.command == "download":
            from .download import default_model
            print(f"Model ready: {default_model()}")
            return
        if a.command == "validate":
            files, problems = find_audio(a.inputs)
            for p in problems:
                log(f"  ERROR   {p}")
            issues = validate(files, a.out_dir, log, ref_options(a)) if files else {}
            code = 1 if problems or not files or any(lv == "error" for v in issues.values() for lv, _ in v) else 0
        else:
            code = transcribe_command(a, log)
    except KeyboardInterrupt:
        sys.exit("stopped")
    except (RuntimeError, ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    if code:
        sys.exit(code)


def main_transcribe(argv=None):
    """`JointRecAlign-transcribe ...` = `JointRecAlign transcribe ...` (the command's first name)."""
    main(["transcribe"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
