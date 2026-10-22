"""Phone transcription and alignment of recordings, with the top-5 candidates of
each phone, as Praat TextGrids. No transcript is needed.

    reco-align-transcribe --model runs/aligner_10ms_sil/checkpoint-26000 \\
        -o results/ recording1.wav recording2.wav [folder/ ...]

(or `python transcribe.py ...` from the repository root.)

For each recording, in the output folder:
    <name>.TextGrid   tiers: phones, confidence, top2, top3, top4, top5, pauses
    <name>.csv        one row per phone: start, end, phone, top1..top5 and their probabilities

Audio: WAV, FLAC, MP3, OGG or AIFF, any sampling rate, mono or stereo.

The probabilities are the model's own (not calibrated): the top-1 is often given
about 0.99 even when it is wrong, but the order of the candidates is informative.
See reco_align/transcribe.py for the method.
"""
import argparse
import json
import os
import sys
import time

from . import transcribe as TR

AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio files or folders of audio files")
    ap.add_argument("--model", required=True, help="latent-SIL checkpoint folder")
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--vocab_path", default=None, help="phone vocabulary (default: vocab.json in the checkpoint)")
    ap.add_argument("--symbols", choices=("ipa", "model"), default="ipa",
                    help="ipa: write nasal vowels and œ in IPA (ɑ̃ ɛ̃ ɔ̃ œ); model: the model's one-character codes")
    ap.add_argument("--phone_map", default=None,
                    help='JSON {"model phone": "your label"} applied last, e.g. {"ə": "ø"} for a corpus that writes the schwa ø')
    ap.add_argument("--max_chunk", type=float, default=10.0,
                    help="longest piece the model sees at once (s); cuts are placed in pauses found by Silero VAD")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    a = ap.parse_args(argv)

    files = []
    for x in a.inputs:
        if os.path.isdir(x):
            files += sorted(os.path.join(x, f) for f in os.listdir(x) if f.lower().endswith(AUDIO))
        elif os.path.isfile(x):
            if not x.lower().endswith(AUDIO):
                sys.exit(f"unsupported audio format: {x} (use {', '.join(AUDIO)})")
            files.append(x)
        else:
            sys.exit(f"not found: {x}")
    if not files:
        sys.exit("no audio file found")

    symbol_map = dict(TR.MODEL2IPA) if a.symbols == "ipa" else {}
    if a.phone_map:
        user = json.load(open(a.phone_map, encoding="utf-8"))
        symbol_map = {k: user.get(v, v) for k, v in symbol_map.items()}   # IPA first, then the user's labels
        symbol_map.update({k: v for k, v in user.items() if k not in symbol_map})
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"loading {a.model}")
    model, fe, tok, device = TR.load_model(a.model, a.vocab_path, a.device)
    print(f"device: {device}")
    for path in files:
        t = time.time()
        phones, pauses, dur = TR.transcribe(path, model, fe, tok, device, a.max_chunk, symbol_map)
        name = os.path.splitext(os.path.basename(path))[0]
        TR.write_textgrid(os.path.join(a.out_dir, name + ".TextGrid"), phones, pauses, dur)
        TR.write_csv(os.path.join(a.out_dir, name + ".csv"), phones)
        print(f"{name}: {dur:.1f} s audio, {len(phones)} phones, {len(pauses)} pauses, "
              f"{time.time() - t:.1f} s -> {a.out_dir}/{name}.TextGrid")


if __name__ == "__main__":
    main()
