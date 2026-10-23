"""Phone transcription and alignment of recordings, with the top-5 candidates of
each phone, as Praat TextGrids. No transcript is needed.

    JointRecAlign recording1.wav recording2.wav [folder/ ...]
    JointRecAlign                       (no argument: opens a window)

The model is downloaded the first time (1.2 GB, once). Results go to a folder
JointRecAlign/ next to the recordings, unless -o gives another folder.
For each recording:
    <name>.TextGrid   tiers: phones, confidence, top2, top3, top4, top5, pauses
    <name>.csv        one row per phone: start, end, phone, top1..top5 and their probabilities

Audio: WAV, FLAC, MP3, OGG or AIFF, any sampling rate, mono or stereo.

The probabilities are the model's own (not calibrated): the top-1 is often given
about 0.99 even when it is wrong, but the order of the candidates is informative.
See jointrecalign/transcribe.py for the method.
"""
import argparse
import json
import os
import sys
import time
import warnings

AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")
OUT_NAME = "JointRecAlign"


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


def symbol_map_for(symbols="ipa", phone_map=None):
    from . import transcribe as TR
    symbol_map = dict(TR.MODEL2IPA) if symbols == "ipa" else {}
    if phone_map:
        user = json.load(open(phone_map, encoding="utf-8"))
        symbol_map = {k: user.get(v, v) for k, v in symbol_map.items()}   # IPA first, then the user's labels
        symbol_map.update({k: v for k, v in user.items() if k not in symbol_map})
    return symbol_map


def run(files, out_dir=None, model=None, symbols="ipa", phone_map=None, max_chunk=10.0, device=None,
        log=print, vocab_path=None):
    """Transcribe the files; one failing file does not stop the others. -> (written TextGrids, failures)."""
    warnings.filterwarnings("ignore")
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
    done, failed = [], []
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        t = time.time()
        try:
            phones, pauses, dur = TR.transcribe(path, m, fe, tok, dev, max_chunk, symbol_map)
            d = out_dir_for(path, out_dir)
            os.makedirs(d, exist_ok=True)
            tg = os.path.join(d, name + ".TextGrid")
            TR.write_textgrid(tg, phones, pauses, dur)
            TR.write_csv(os.path.join(d, name + ".csv"), phones)
        except Exception as e:
            failed.append((path, str(e) or type(e).__name__))
            log(f"[{i}/{len(files)}] {name}: FAILED ({failed[-1][1]})")
            continue
        done.append(tg)
        log(f"[{i}/{len(files)}] {name}: {dur:.1f} s of audio, {len(phones)} phones, {len(pauses)} pauses "
            f"({time.time() - t:.1f} s) -> {tg}")
    log(f"Done: {len(done)} of {len(files)} recordings transcribed" + (f", {len(failed)} failed" if failed else ""))
    return done, failed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="JointRecAlign", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="audio files or folders of audio files")
    ap.add_argument("-o", "--out_dir", default=None,
                    help=f"folder for the results (default: a folder {OUT_NAME}/ next to each recording)")
    ap.add_argument("--model", default=None,
                    help="model folder (default: the released model, downloaded the first time)")
    ap.add_argument("--symbols", choices=("ipa", "model"), default="ipa",
                    help="ipa: write nasal vowels and œ in IPA (ɑ̃ ɛ̃ ɔ̃ œ); model: the model's one-character codes")
    ap.add_argument("--phone_map", default=None,
                    help='JSON {"model phone": "your label"} applied last, e.g. {"ə": "ø"} for a corpus that writes the schwa ø')
    ap.add_argument("--max_chunk", type=float, default=10.0,
                    help="longest piece the model sees at once (s); cuts are placed in pauses found by Silero VAD")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--vocab_path", default=None, help="phone vocabulary (default: vocab.json in the model folder)")
    ap.add_argument("--download", action="store_true", help="only download the model, then stop")
    ap.add_argument("--install-praat-plugin", action="store_true",
                    help="add a JointRecAlign command to Praat's menus, then stop")
    a = ap.parse_args(argv)

    try:
        if a.install_praat_plugin:
            from .praat import install_plugin
            install_plugin()
            return
        if a.download:
            from .download import default_model
            default_model()
            return
        if not a.inputs:
            ap.error("give audio files or folders (or run JointRecAlign with nothing after it to open the window)")
        files, problems = find_audio(a.inputs)
        for p in problems:
            print(p, file=sys.stderr)
        if not files:
            sys.exit("no audio file to transcribe")
        _, failed = run(files, a.out_dir, a.model, a.symbols, a.phone_map, a.max_chunk, a.device,
                        vocab_path=a.vocab_path)
    except KeyboardInterrupt:
        sys.exit("stopped")
    except (RuntimeError, ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    if failed or problems:
        sys.exit(1)


def app(argv=None):
    """`JointRecAlign`: the window when there is no argument, the command line otherwise."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        from .gui import main as gui
        return gui()
    return main(argv)


if __name__ == "__main__":
    main()
