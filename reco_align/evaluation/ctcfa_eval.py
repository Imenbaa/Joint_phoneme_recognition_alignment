"""Score a checkpoint with CTC forced alignment on its CTC head (no attention).

Same manifest, chunks and output format as attention_eval.run_manifest, so the
two are directly comparable; only the boundary extraction differs:

  attention (run_manifest): DTW on the cross-attention energy, onset frames,
                            contiguous ends.
  ctcfa (this file):        greedy CTC decode, torchaudio forced_align of that
                            sequence on the CTC log posteriors, one interval per
                            non-blank run with its own start AND end frame
                            (blank frames between phones stay unassigned).

The decoded phone sequence is the same CTC greedy decode the attention path
aligns, so PER matches the attention eval; only the boundaries change.

end_mode="contiguous" instead ends each phone where the next one starts and
the last phone of a chunk at the chunk end, exactly as run_manifest does for
the DTW path, so blank frames go to the preceding phone.

Frame timing follows attention_eval: frame stride and receptive-field offset
are read from the model config, so frame f of a chunk starting at cs is at
cs + f * stride + offset, for 20 ms and 10 ms models alike.
"""
import unicodedata

import numpy as np
import torch
import torchaudio

from .attention_eval import (SR, HYP_FIX, SKIP_TOK, load_audio, frame_stride_s,
                             frame_time_offset_s, to_model_rate)


def ctc_logits(o):
    for name in ("logits", "ctc_logits"):
        v = getattr(o, name, None)
        if v is not None:
            return v
    raise AttributeError("no CTC logits in the model output")


@torch.inference_mode()
def run_manifest_ctcfa(model, feature_extractor, tokenizer, manifest,
                       device="cuda", max_files=None, end_mode="span"):
    """-> {file_id: {ref_intervals, hyp_intervals, ref_seq, hyp_seq, style}}"""
    was_training = model.training
    model.eval()
    blank = tokenizer.pad_token_id
    stride = frame_stride_s(model)
    offset = frame_time_offset_s(model)
    store, n_fail = {}, 0
    try:
        for nm, meta in list(manifest["files"].items())[:max_files]:
            wav = torch.from_numpy(load_audio(meta["wav"])).float()
            hyp = []
            for ch in meta["chunks"]:
                s0, s1 = int(ch["start"] * SR), int(ch["end"] * SR)
                seg = wav[s0:s1]
                if seg.numel() < 400:      # under one conv receptive field
                    continue
                x = feature_extractor(to_model_rate(seg.numpy(), model),
                                      sampling_rate=feature_extractor.sampling_rate,
                                      return_tensors="pt", return_attention_mask=True)
                o = model(input_values=x.input_values.to(device),
                          attention_mask=x.attention_mask.to(device),
                          labels=None, prior_weight=0.0,
                          ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
                T = int(o.input_lengths[0])
                logp = ctc_logits(o)[0, :T].float().log_softmax(-1).cpu()

                # greedy collapse -> forced-alignment target
                ids, prev = [], None
                for t in logp.argmax(-1).tolist():
                    if t != prev and t != blank:
                        ids.append(t)
                    prev = t
                if not ids:
                    continue
                try:
                    aligned, _ = torchaudio.functional.forced_align(
                        logp.unsqueeze(0).contiguous(), torch.tensor([ids], dtype=torch.int32),
                        torch.tensor([T], dtype=torch.int32), torch.tensor([len(ids)], dtype=torch.int32),
                        blank=blank)
                except Exception as e:     # e.g. more targets than frames
                    n_fail += 1
                    print(f"  [forced_align skipped] {nm} {ch['start']:.2f}-{ch['end']:.2f}: {e}")
                    continue
                aligned = aligned[0].tolist()

                # runs with start AND end frame; blank frames stay unassigned
                spans, run_id, run_start = [], None, 0
                for fi, tid in enumerate(aligned):
                    if tid != run_id:
                        if run_id is not None and run_id != blank:
                            spans.append((run_id, run_start, fi))
                        run_id, run_start = tid, fi
                if run_id is not None and run_id != blank:
                    spans.append((run_id, run_start, len(aligned)))

                off = s0 / SR + offset
                ca = []
                for tid, sf, ef in spans:
                    ph = tokenizer.convert_ids_to_tokens(tid)
                    if ph in SKIP_TOK:
                        continue
                    if len(ph) == 1 and unicodedata.combining(ph):
                        if ca:
                            ca[-1]["phoneme"] += ph
                            ca[-1]["end"] = off + ef * stride
                        continue
                    ca.append({"phoneme": HYP_FIX.get(ph, ph),
                               "start": off + sf * stride,
                               "end": off + ef * stride})
                if end_mode == "contiguous":
                    for i in range(len(ca) - 1):
                        ca[i]["end"] = ca[i + 1]["start"]
                    if ca:
                        ca[-1]["end"] = off + T * stride
                hyp.extend(ca)

            hyp.sort(key=lambda x: x["start"])
            store[nm] = {"ref_intervals": meta["ref_intervals"],
                         "hyp_intervals": hyp,
                         "ref_seq": [x["phoneme"] for x in meta["ref_intervals"]],
                         "hyp_seq": [x["phoneme"] for x in hyp],
                         "style": meta.get("style", "ALL")}
    finally:
        model.train(was_training)
    if n_fail:
        print(f"  [ctcfa] {n_fail} chunks skipped by forced_align")
    return store
