#!/usr/bin/env python3
"""Phone transcription and alignment of recordings as Praat TextGrids (no transcript needed).
Same as the `JointRecAlign` command; see jointrecalign/transcribe_cli.py or --help.

    python transcribe.py recordings/
"""
from jointrecalign.transcribe_cli import main

if __name__ == "__main__":
    main()
