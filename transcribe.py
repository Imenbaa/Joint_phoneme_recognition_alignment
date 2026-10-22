#!/usr/bin/env python3
"""Phone transcription and alignment of recordings as Praat TextGrids (no transcript needed).
Same as the `reco-align-transcribe` command; see reco_align/transcribe_cli.py or --help.

    python transcribe.py --model runs/aligner_10ms_sil/checkpoint-26000 -o results/ recordings/
"""
from reco_align.transcribe_cli import main

if __name__ == "__main__":
    main()
