#!/usr/bin/env python3
"""CPU checks of the full model on a tiny random wav2vec2 (no download, no data):

    python tests/test_model.py

1. SIL off: finite L_CTC / L_FS / L_m, gradients reach the aligner.
2. SIL on (fs_posterior attention for L_m, word slots, learned entry cost,
   same-code negative masking, L_energy, top-k joint routing):
   finite losses; gradients reach sil_key / sil_logp / queries / ConvBank /
   phone_proj, also from L_m alone; L_energy waits for SIL.
3. L_energy target: frame energies line up with the encoder frames on a
   silence / tone / silence signal; padded frames get 0.
4. Evaluation decodes (SIL Viterbi and DTW) run on the eval-mode outputs.
"""
import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

from jointrecalign import sil_fs
from jointrecalign.evaluation.attention_eval import dtw_path
from jointrecalign.model import Wav2Vec2ForCTCPoolContrastive, frame_log_energy

SR = 16000
torch.manual_seed(0)
rng = np.random.default_rng(0)


def tiny_config(use_sil):
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
                     joint_encoder_alpha=0.1, joint_lm_encoder_alpha=0.1,
                     encoder_top_layers=2).items():
        setattr(cfg, k, v)
    if use_sil:
        for k, v in dict(use_sil=True, sil_positions="word", sil_eval_positions="all",
                         sil_logp=-2.0, sil_logp_learned=True, sil_min_frames=4,
                         lm_att_source="fs_posterior", lm_mask_same_code=True,
                         lm_energy_weight=1.0).items():
            setattr(cfg, k, v)
    return cfg


def batch():
    """Three utterances of 2-3 s: noise bursts ("speech") separated by pauses,
    random phone labels with word boundaries."""
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    wavs, labs, slots = [], [], []
    for n in (2, 3, 3):
        parts = []
        for _ in range(n):
            parts += [1e-4 * rng.standard_normal(int(0.3 * SR)), 0.3 * rng.standard_normal(int(0.6 * SR))]
        wavs.append(np.concatenate(parts).astype(np.float32))
        words = [rng.integers(2, 12, size=rng.integers(2, 5)).tolist() for _ in range(n)]
        labs.append(sum(words, []))
        slots.append([0] + np.cumsum([len(w) for w in words]).tolist())
    x = fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True, return_attention_mask=True)
    lab = torch.full((len(labs), max(map(len, labs))), -100)
    for i, l in enumerate(labs):
        lab[i, :len(l)] = torch.tensor(l)
    ss = torch.full((len(slots), max(map(len, slots))), -1)
    for i, s in enumerate(slots):
        ss[i, :len(s)] = torch.tensor(s)
    return fe, x, lab, ss


def init_codebook(m, x):
    """As initialize_quantizer in training: codevectors = real encoder frames."""
    with torch.no_grad():
        Xe = m.wav2vec2(x.input_values, attention_mask=x.attention_mask).last_hidden_state
        Le = m._get_feat_extract_output_lengths(x.attention_mask.sum(-1))
        fr = torch.cat([Xe[i, :int(l)] for i, l in enumerate(Le)]).float()
        q = m.ema_quantizer
        pick = fr[torch.randperm(fr.size(0))[:q.num_vars]]
        q.codevectors.copy_(pick); q.ema_embed.copy_(pick); q.cluster_size.fill_(1.0)


def grads(m, names):
    return {k: (0.0 if p.grad is None else float(p.grad.norm())) for k, p in names.items()}


fe, x, lab, ss = batch()
kw = dict(input_values=x.input_values, attention_mask=x.attention_mask, labels=lab, ctc_weight=1.0,
          align_weight=1.0, contrastive_weight=1.0, prior_weight=0.6, encoder_alpha=0.1,
          lm_encoder_alpha=0.1)

# ---- 1. SIL off
m = Wav2Vec2ForCTCPoolContrastive(tiny_config(use_sil=False))
m.freeze_feature_extractor()
init_codebook(m, x)
m.train()
o = m(**kw)
for k in ("loss", "ctc_loss", "align_loss", "contrastive_loss"):
    assert torch.isfinite(getattr(o, k)), k
o.loss.backward()
g = grads(m, {"attention.q": m.attention.q.weight, "phone_proj": m.phone_proj[0].weight,
              "recon_head": m.reconstruction_head[0].weight,
              "top_layer": m.wav2vec2.encoder.layers[-1].feed_forward.output_dense.weight,
              "first_layer_attn": m.wav2vec2.encoder.layers[0].attention.q_proj.weight})
print("[SIL off] losses", {k: round(getattr(o, k).item(), 3)
                           for k in ("loss", "ctc_loss", "align_loss", "contrastive_loss")})
assert all(v > 0 for v in g.values()), g
print("1. SIL off: ok")

# ---- 2. SIL on
m = Wav2Vec2ForCTCPoolContrastive(tiny_config(use_sil=True))
m.freeze_feature_extractor()
m.reset_sil_parameters()
init_codebook(m, x)
m.train()
n, thr = m.init_sil_key_lowenergy(x.input_values, x.attention_mask)
print(f"lowenergy init: {n} frames <= {thr:.1f} dB, |sil_key| = {m.sil_key.norm():.3f}")
names = {"sil_key": m.sil_key, "sil_logp": m.sil_logp, "attention.q": m.attention.q.weight,
         "cnn.out_linear": m.cnn.out_linear.weight, "phone_proj": m.phone_proj[0].weight,
         "energy_head": m.energy_head[0].weight}
for aw, cw, tag in ((1.0, 1.0, "L_FS + L_m"), (0.0, 1.0, "L_m only"), (1.0, 0.0, "L_FS only")):
    torch.manual_seed(1)
    m.zero_grad()
    o = m(**{**kw, "align_weight": aw, "contrastive_weight": cw}, sil_active=True, sil_slots=ss)
    o.loss.backward()
    g = grads(m, names)
    print(f"[{tag}] loss {o.loss.item():.3f} fs {o.align_loss.item():.3f} lm {o.contrastive_loss.item():.3f} "
          f"energy {o.lm_energy_loss.item():.3f} | sil frame_frac {o.sil_frame_frac.item():.3f}")
    assert torch.isfinite(o.loss)
    if tag != "L_FS only":
        assert g["sil_key"] > 0 and g["attention.q"] > 0 and g["phone_proj"] > 0, (tag, g)
        assert g["energy_head"] > 0, (tag, g)
    if tag == "L_m only":
        assert g["sil_logp"] > 0, "L_m does not reach the entry cost through gamma"
o = m(**kw, sil_active=False, sil_slots=ss)
assert o.lm_energy_loss is None, "L_energy must wait for SIL"
print("2. SIL on: ok")

# ---- 3. L_energy target alignment
t = np.arange(SR) / SR
sig = np.concatenate([np.zeros(SR), 0.5 * np.sin(2 * np.pi * 220 * t), np.zeros(SR)])
sig = (sig + 1e-4 * rng.standard_normal(sig.size)).astype(np.float32)
m.eval()
xs = fe([sig], sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
with torch.no_grad():
    o = m(input_values=xs.input_values, attention_mask=xs.attention_mask, labels=None,
          prior_weight=0.0, ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
T = int(o.input_lengths[0])
hw = dict(hop=m._frame_hop, win=m._frame_win)
assert (m._frame_hop, m._frame_win) == (160, 400), hw
e = frame_log_energy(xs.input_values, xs.attention_mask, T, o.input_lengths, **hw)[0] * 20
fs_, wd = m._frame_hop / SR, m._frame_win / SR
sil_f = [f for f in range(T) if (f * fs_ + wd <= 1.0) or (f * fs_ >= 2.0)]
ton_f = [f for f in range(T) if 1.0 <= f * fs_ and f * fs_ + wd <= 2.0]
assert e[sil_f].max() < -50 and e[ton_f].min() > -3, (e[sil_f].max(), e[ton_f].min())
xb = fe([sig, sig[:2 * SR]], sampling_rate=SR, return_tensors="pt", return_attention_mask=True, padding=True)
with torch.no_grad():
    ob = m(input_values=xb.input_values, attention_mask=xb.attention_mask, labels=None,
           prior_weight=0.0, ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
eb = frame_log_energy(xb.input_values, xb.attention_mask, int(ob.input_lengths.max()), ob.input_lengths, **hw)
assert eb[1, int(ob.input_lengths[1]):].abs().max() == 0
print("3. L_energy target: ok")

# ---- 4. evaluation decodes (eval mode, CTC greedy target)
with torch.no_grad():
    o = m(input_values=x.input_values[:1], attention_mask=x.attention_mask[:1], labels=None,
          prior_weight=0.0, ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
T, P = int(o.input_lengths[0]), int(o.text_lengths[0])
if P == 0:      # untrained CTC head may decode nothing: align the reference phones instead
    with torch.no_grad():
        o = m(input_values=x.input_values[:1], attention_mask=x.attention_mask[:1], labels=lab[:1],
              prior_weight=0.0, ctc_weight=0.0, align_weight=0.0, contrastive_weight=0.0)
    T, P = int(o.input_lengths[0]), int(o.text_lengths[0])
lp = torch.log_softmax(torch.cat([o.energy[0, :T, :P], o.sil_energy_raw[0, :T, None]], -1), -1).numpy()
top, path = sil_fs.viterbi(lp, P, sil_fs.slot_mask(P, "all"), m.sil_min_frames, float(m.sil_logp))
assert path is not None and len(path) == T
d = dtw_path(np.exp(o.energy[0, :T, :P].float().log_softmax(-1).numpy()))
assert d is not None and len(d) == T and (np.diff(d) >= 0).all()
print(f"4. eval decodes: ok (T={T} P={P}, SIL frames {int((top['col'][path] < 0).sum())})")
print("model checks passed")
