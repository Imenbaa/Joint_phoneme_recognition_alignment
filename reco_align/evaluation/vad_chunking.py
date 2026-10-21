"""
WhisperX-style VAD chunking of long recordings, for evaluation manifests.

Speech segments come from the WhisperX voice-activity model (pyannote
segmentation) with hysteresis binarization (onset 0.5 / offset 0.363), then
are cut and merged into chunks of at most `max_chunk_duration` seconds,
keeping the pauses inside a chunk:

  - segment bounds use the frame START / END of the VAD frames (not their
    centres, which would cut ~31 ms into speech on each side);
  - a segment longer than the maximum is split at its quietest frame in the
    second half of the window;
  - the first chunk starts at 0.0 and the last ends at the audio duration, so
    nothing before the first / after the last speech is dropped.

Needs `whisperx` (requirements-manifest.txt); it is imported only when the
VAD runs, so the rest of the package does not depend on it.
"""

import numpy as np
import torch

# WhisperX defaults
ONSET = 0.5
OFFSET = 0.363

_VAD_PIPELINE = None


def get_vad_pipeline(device="cuda", model_fp=None):
    """Build the segmentation pipeline once and reuse it."""
    global _VAD_PIPELINE
    if _VAD_PIPELINE is None:
        from whisperx.vads.pyannote import load_vad_model
        kw = {"device": device}
        if model_fp is not None:
            kw["model_fp"] = model_fp
        _VAD_PIPELINE = load_vad_model(**kw)
    return _VAD_PIPELINE


def load_whisperx_vad(wav, sampling_rate=16000, device="cuda", model_fp=None):
    """Return (scores, starts, ends): per-frame VAD score and its true bounds.

    `sliding_window` gives start/step/duration. Frame i spans
        [start + i*step, start + i*step + duration]
    """
    pipeline = get_vad_pipeline(device=device, model_fp=model_fp)

    if isinstance(wav, torch.Tensor):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)                    # (T,) -> (channel, T)
        wav = {"waveform": wav, "sample_rate": sampling_rate}

    vad_scores = pipeline(wav)
    scores = np.asarray(vad_scores.data[:, 0], dtype=float)
    sw = vad_scores.sliding_window
    idx = np.arange(len(scores))
    starts = sw.start + idx * sw.step
    ends = starts + sw.duration
    return scores, starts, ends


def _binarize(scores, starts, ends, onset=ONSET, offset=OFFSET):
    """Hysteresis binarization -> list of (start_s, end_s) speech segments.

    A segment opens at the START of the frame that crossed `onset` and closes
    at the END of the last frame above `offset`.
    """
    segments = []
    is_active = scores[0] > onset
    seg_start = starts[0] if is_active else None
    last_active_end = ends[0] if is_active else None

    for i in range(1, len(scores)):
        sc = scores[i]
        if is_active:
            if sc < offset:
                segments.append((seg_start, last_active_end))
                is_active = False
            else:
                last_active_end = ends[i]
        else:
            if sc > onset:
                seg_start = starts[i]
                last_active_end = ends[i]
                is_active = True
    if is_active:
        segments.append((seg_start, last_active_end))
    return segments


def _split_long_by_score(seg_start, seg_end, starts, scores, chunk_size):
    """Split a > chunk_size segment at the quietest frame in the second half."""
    pieces, cur = [], seg_start
    while seg_end - cur > chunk_size:
        lo, hi = cur + chunk_size * 0.5, cur + chunk_size
        idx = np.where((starts >= lo) & (starts <= hi))[0]
        cut = starts[idx[np.argmin(scores[idx])]] if len(idx) else cur + chunk_size
        pieces.append((cur, cut))
        cur = cut
    pieces.append((cur, seg_end))
    return pieces


def merge_chunks(segments, starts, scores, chunk_size=20.0, duration=None):
    """Cut & merge -> list of {"start","end"} chunks, each <= chunk_size.

    The first chunk is clamped to 0.0 and the last to `duration` (when given),
    so the head and tail of the file are never dropped.
    """
    starts = np.asarray(starts, dtype=float)
    scores = np.asarray(scores, dtype=float)

    split = []
    for s, e in segments:
        if e - s > chunk_size:
            split.extend(_split_long_by_score(s, e, starts, scores, chunk_size))
        else:
            split.append((s, e))
    if not split:
        return [] if duration is None else [{"start": 0.0, "end": float(duration)}]

    merged = []
    curr_start, curr_end = split[0]
    for s, e in split[1:]:
        if e - curr_start > chunk_size and curr_end - curr_start > 0:
            merged.append({"start": curr_start, "end": curr_end})
            curr_start = s
        curr_end = e
    merged.append({"start": curr_start, "end": curr_end})

    merged[0]["start"] = 0.0
    if duration is not None:
        merged[-1]["end"] = float(duration)

    for c in merged:
        c["start"] = float(c["start"])
        c["end"] = float(c["end"])
    return merged


def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=8.0,
    onset=ONSET,
    offset=OFFSET,
    device="cuda",
    model_fp=None,
):
    """Return [{"start","end"}] in seconds, original file timeline.

    wav : torch.Tensor (1D, 16 kHz) or a file path.
    """
    duration = None
    if isinstance(wav, torch.Tensor):
        duration = wav.shape[-1] / float(sampling_rate)

    scores, starts, ends = load_whisperx_vad(
        wav, sampling_rate=sampling_rate, device=device, model_fp=model_fp)

    segments = _binarize(scores, starts, ends, onset, offset)

    return merge_chunks(segments, starts, scores,
                        chunk_size=max_chunk_duration, duration=duration)
