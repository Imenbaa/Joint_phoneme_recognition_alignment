#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wav2Vec2ForCTCPoolContrastive: the pooled-key aligner (pool_alignment.py) plus a
masked reconstruction loss L_m.

Mask the encoder frames, reconstruct them from the masked frames plus the
alignment-weighted phone embeddings, and score the reconstruction against
EMA-quantized targets with an InfoNCE loss.

    X_masked, mask = mask_features(X)                    # p ~ U[0.1, 0.5]
    YA  = att @ phone_hidden                             # (B, T, d)
    p   = pos_proj([pos ; dur])                          # (B, T, 64)
    H   = reconstruction_head([X_masked ; YA ; p])       # (B, T, D)
    Q,_ = ema_quantizer(X.detach())                      # (B, T, D)
    L_m = InfoNCE(H, Q, mask)                            # masked frames only

Total loss = ctc_weight * L_CTC + align_weight * L_FS + contrastive_weight * L_m
(+ contrastive_weight * lm_energy_weight * L_energy when config.lm_energy_weight > 0).

Position features (config.use_position_features)
------------------------------------------------
Without them, YA is constant inside a phone, so the head can only output that
phone's average frame and nothing in L_m depends on phone duration. pos (relative
position of the frame inside its phone) and dur (soft duration of the phone) are
derived from the attention itself, so they are differentiable. They are projected
to 64 dimensions before concatenation. When they are disabled, zeros are fed
instead, so the architecture and checkpoints do not change.

L_energy (config.lm_energy_weight)
----------------------------------
A small head predicts each frame's waveform log-energy from the alignment content
only ([YA ; pos/dur]). A loud frame assigned to SIL is then predicted quiet, and
the error reaches the alignment, the SIL key and the SIL entry cost.

Gradient routing
----------------
forward(..., encoder_alpha=None): legacy routing set by the config flags
  detach_alignment_input / detach_contrastive_input.
forward(..., encoder_alpha=a, lm_encoder_alpha=b), config.encoder_top_layers=k:
  L_CTC -> every encoder layer
  L_FS  -> queries -> top-k transformer layers, gradient x a (keys detached)
  L_m   -> its own attention -> queries -> top-k layers, gradient x b
  L_m targets (EMA codes of the encoder output) and the masked frames stay
  detached: L_m can only reach the encoder through the attention.
  k = 0 means all transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pool_alignment import Wav2Vec2ForCTCPoolAlignment, GradScale, SilAlignmentOutput
from .sil_fs import unit_occupancy


POS_FEAT_DIM = 64

# Encoder frame t of the 20 ms wav2vec2 conv stack sees samples [320 t, 320 t + 400).
# The model passes its own hop / window (160 / 400 at 10 ms).
FRAME_HOP, FRAME_WIN = 320, 400


@torch.no_grad()
def frame_log_energy(input_values, attention_mask, T, input_lengths, floor_db=-60.0,
                     hop=FRAME_HOP, win=FRAME_WIN):
    """Per-encoder-frame log-energy target, from the waveform (never from features).

    dB of the mean square over the frame's 400-sample window, relative to the
    utterance's 95th percentile over its real frames, clipped at floor_db, /20.
    -> (B, T) float32; 0 = loud speech, floor_db/20 = silence. Padded frames 0."""
    x = input_values.float()
    if attention_mask is not None:
        x = x * attention_mask.to(x.dtype)
    need = (T - 1) * hop + win
    if x.size(1) < need:
        x = F.pad(x, (0, need - x.size(1)))
    fr = x.unfold(1, win, hop)[:, :T]                              # (B, T, win)
    db = 10.0 * torch.log10(fr.pow(2).mean(-1) + 1e-10)
    valid = (torch.arange(T, device=x.device)[None, :]
             < input_lengths.to(x.device)[:, None])
    out = torch.zeros_like(db)
    for b in range(db.size(0)):
        v = db[b, valid[b]]
        if v.numel() == 0:
            continue
        ref = torch.quantile(v, 0.95)
        out[b, valid[b]] = (v - ref).clamp(min=floor_db) / 20.0
    return out


class GumbelQuantizerEMA(nn.Module):
    """EMA vector quantizer of the encoder frames (L_m targets).

    assign="gumbel"  : hard Gumbel-softmax over weight_proj(X). weight_proj
                       is trained by L_m through the straight-through
                       estimator, so L_m can move its own targets.
    assign="nearest" : code with the smallest L2 distance to X, as in
                       EMA VQ-VAE. No gradient reaches the assignment;
                       weight_proj is unused.
    """

    ASSIGN_MODES = ("gumbel", "nearest")

    def __init__(self, input_dim: int, num_vars: int = 320,
                 temp: float = 2.0, decay: float = 0.99,
                 assign: str = "gumbel"):
        super().__init__()
        if assign not in self.ASSIGN_MODES:
            raise ValueError(f"assign must be one of {self.ASSIGN_MODES}, got {assign!r}")
        self.num_vars = num_vars
        self.temp = temp
        self.decay = decay
        self.assign = assign

        self.weight_proj = nn.Linear(input_dim, num_vars)

        self.register_buffer("codevectors", torch.randn(num_vars, input_dim))
        self.register_buffer("cluster_size", torch.ones(num_vars))
        self.register_buffer("ema_embed", torch.randn(num_vars, input_dim))
        nn.init.uniform_(self.codevectors, -1.0, 1.0)

    def forward(self, X: torch.Tensor, frame_mask: torch.Tensor = None):
        """frame_mask: (B, T) bool, True for real frames. Padded frames are
        still quantized but do not enter the EMA statistics."""
        B, T, D = X.shape
        if self.assign == "nearest":
            with torch.no_grad():
                Xf = X.reshape(-1, D).float()
                C = self.codevectors.float()
                dist = (Xf.pow(2).sum(1, keepdim=True)
                        - 2.0 * Xf @ C.T
                        + C.pow(2).sum(1)[None, :])
                indices = dist.argmin(dim=-1).reshape(B, T)
                probs = F.one_hot(indices, self.num_vars).to(X.dtype)
            logits = None
        else:
            logits = self.weight_proj(X)

        if self.training:
            if logits is not None:
                probs = F.gumbel_softmax(logits, tau=self.temp, hard=True)

            with torch.no_grad():
                flat_probs = probs.reshape(-1, self.num_vars)
                if frame_mask is not None:
                    flat_probs = flat_probs * frame_mask.reshape(-1, 1).to(flat_probs.dtype)
                flat_X = X.reshape(-1, D)
                counts = flat_probs.sum(0)
                embed_sum = flat_probs.T @ flat_X

                self.cluster_size = (self.decay * self.cluster_size
                                     + (1 - self.decay) * counts)
                self.ema_embed = (self.decay * self.ema_embed
                                  + (1 - self.decay) * embed_sum)
                self.codevectors = (self.ema_embed
                                    / self.cluster_size.unsqueeze(1).clamp(min=1e-5))
        elif logits is not None:
            indices = logits.argmax(dim=-1)
            probs = F.one_hot(indices, self.num_vars).float()

        quantized = torch.matmul(probs, self.codevectors)
        return quantized, probs


class Wav2Vec2ForCTCPoolContrastive(Wav2Vec2ForCTCPoolAlignment):

    def __init__(self, config):
        super().__init__(config)

        D = config.hidden_size
        d = self.align_hidden_size

        self.n_negatives = int(getattr(config, "n_negatives", 50))
        self.contrastive_temperature = float(
            getattr(config, "contrastive_temperature", 0.1)
        )
        self.mask_prob_low = float(getattr(config, "mask_prob_low", 0.1))
        self.mask_prob_high = float(getattr(config, "mask_prob_high", 0.5))
        self.detach_contrastive_input = bool(
            getattr(config, "detach_contrastive_input", False)
        )
        self.use_position_features = bool(
            getattr(config, "use_position_features", True)
        )
        # dur is a frame count: express it in 20 ms units so the feature has
        # the same meaning at 10 ms (1.0 for the standard 20 ms model).
        self._dur_scale = (getattr(config, "inputs_to_logits_ratio", 320)
                           / float(getattr(config, "input_sampling_rate", 16000) or 16000)
                           / 0.02)
        # Conv receptive field and hop of one encoder frame, in input samples
        # (400 / 320 for the 20 ms model, 400 / 160 for 10 ms).
        rf, hop = 1, 1
        for k, st in zip(config.conv_kernel, config.conv_stride):
            rf += (k - 1) * hop
            hop *= st
        self._frame_win, self._frame_hop = int(rf), int(hop)
        # Mask negatives whose code equals the positive's (wav2vec2
        # neg_is_pos). Off by default = previous behaviour.
        self.lm_mask_same_code = bool(getattr(config, "lm_mask_same_code", False))

        self.ema_quantizer = GumbelQuantizerEMA(
            input_dim=D,
            num_vars=int(getattr(config, "num_codevectors_ema", 320)),
            decay=float(getattr(config, "ema_decay", 0.99)),
            assign=getattr(config, "quantizer_assign", "gumbel"),
        )

        # pos and dur are two scalars; project them up so they are not
        # drowned out by the 1536 dimensions of X_masked and YA.
        self.pos_proj = nn.Sequential(
            nn.Linear(2, POS_FEAT_DIM),
            nn.GELU(),
        )

        # Two layers: (identity, position) -> acoustics is not a linear map.
        # Input width is fixed at D + d + POS_FEAT_DIM in both modes, so the
        # ablation does not change the architecture.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(D + d + POS_FEAT_DIM, 1024),
            nn.GELU(),
            nn.Linear(1024, D),
        )

        # L_energy. A head that predicts each frame's log-energy
        # (from the waveform) from the alignment content only: YA and pos/dur,
        # never the frame's own features. A loud frame given to SIL is then
        # predicted quiet and the error reaches gamma, sil_key and the entry
        # cost. Built only when the weight is > 0, so runs without it keep
        # the reference parameters and RNG stream.
        self.lm_energy_weight = float(getattr(config, "lm_energy_weight", 0.0))
        self.lm_energy_floor_db = float(getattr(config, "lm_energy_floor_db", -60.0))
        heads = [self.pos_proj, self.reconstruction_head]
        if self.lm_energy_weight > 0:
            self.energy_head = nn.Sequential(
                nn.Linear(d + POS_FEAT_DIM, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
            heads.append(self.energy_head)

        # nn.Sequential has no .weight / .bias — initialise the Linear
        # submodules explicitly.
        for module in heads:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Parameter groups — without this the new modules land in no optimizer
    # group and never train.
    # ------------------------------------------------------------------
    def get_alignment_parameters(self):
        # super() already holds sil_key / sil_logp when config.use_sil is set.
        return (
            super().get_alignment_parameters()
            + list(self.reconstruction_head.parameters())
            + list(self.pos_proj.parameters())
            + list(self.ema_quantizer.parameters())
            + (list(self.energy_head.parameters()) if self.lm_energy_weight > 0 else [])
        )

    # ------------------------------------------------------------------
    # Masking and InfoNCE
    # ------------------------------------------------------------------
    def mask_features(self, X, p_low=None, p_high=None, frame_mask=None):
        """frame_mask: (B, T) bool, True for real frames. Padded frames are
        never selected, so they are never reconstruction targets."""
        p_low = self.mask_prob_low if p_low is None else p_low
        p_high = self.mask_prob_high if p_high is None else p_high
        B, T, _ = X.shape
        mask_prob = torch.FloatTensor(1).uniform_(p_low, p_high).item()
        mask = torch.rand(B, T, device=X.device) < mask_prob
        if frame_mask is not None:
            mask = mask & frame_mask
        X_masked = X.clone()
        X_masked[mask] = 0.0
        return X_masked, mask

    def contrastive_loss(self, H, X_quantized, mask, temperature=None,
                         frame_mask=None, codes=None):
        """frame_mask: (B, T) bool, True for real frames. Negatives are drawn
        from real frames only. codes: (B, T) target code index; when given,
        negatives with the positive's code are masked out (their vector IS the
        target, so they can never be beaten)."""
        B, T, D = H.shape
        temp = temperature if temperature is not None else self.contrastive_temperature
        H_flat = H.reshape(B * T, D)
        Q_flat = X_quantized.reshape(B * T, D)
        mask_flat = mask.reshape(B * T)

        if mask_flat.sum() == 0:
            return H_flat.sum() * 0.0

        H_norm = F.normalize(H_flat + 1e-8, dim=-1)
        Q_norm = F.normalize(Q_flat + 1e-8, dim=-1)

        masked_H = H_norm[mask_flat]
        masked_Q = Q_norm[mask_flat]
        M = masked_H.shape[0]

        if frame_mask is not None:
            pool = frame_mask.reshape(B * T).nonzero(as_tuple=True)[0]
        else:
            pool = torch.arange(B * T, device=H.device)
        n_neg = min(self.n_negatives, pool.numel() - 1)
        neg_idx = pool[torch.randint(0, pool.numel(), (M, n_neg), device=H.device)]
        Q_neg = Q_norm[neg_idx]

        sim_pos = (masked_H * masked_Q).sum(dim=-1) / temp
        sim_neg = torch.bmm(Q_neg, masked_H.unsqueeze(-1)).squeeze(-1) / temp
        if codes is not None:
            c_flat = codes.reshape(B * T)
            same = c_flat[neg_idx] == c_flat[mask_flat].unsqueeze(1)
            sim_neg = sim_neg.masked_fill(same, float("-inf"))

        all_sims = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        loss = -sim_pos + torch.logsumexp(all_sims, dim=1)
        return loss.mean()

    # ------------------------------------------------------------------
    # Within-phone position and soft duration, both derived from att.
    # ------------------------------------------------------------------
    def _position_features(self, att, out_dtype):
        """att: (B, T, N) attention over phones. Returns (B, T, POS_FEAT_DIM).

        total[n] = sum_t att[t, n]         -> soft duration of phone n, in frames
        rel[t,n] = (mass before t) / total -> where t sits inside phone n, 0..1
        pos[t]   = sum_n att[t,n] * rel[t,n]
        dur[t]   = sum_n att[t,n] * total[n], log-compressed
        """
        a = att.float()
        total = a.sum(dim=1, keepdim=True).clamp(min=1e-6)      # (B, 1, N)
        cum = a.cumsum(dim=1) - 0.5 * a                         # midpoint rule
        rel = cum / total                                       # (B, T, N)
        pos = (a * rel).sum(-1, keepdim=True)                   # (B, T, 1)
        dur = (a * total.expand_as(a)).sum(-1, keepdim=True)    # (B, T, 1)
        dur = dur * self._dur_scale                             # 20 ms units
        dur = torch.log1p(dur) / 4.0                            # ~[0, 1]
        return self.pos_proj(torch.cat([pos, dur], dim=-1).to(out_dtype))

    # ------------------------------------------------------------------
    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ctc_weight=1.0,
        align_weight=1.0,
        contrastive_weight=1.0,
        pred_target_prob=0.0,
        prior_weight=1.0,
        skip_alignment=False,
        encoder_alpha=None,
        lm_encoder_alpha=None,
        sil_active=False,
        sil_slots=None,
    ):
        # The parent handles CTC, pooling, energy, prior and L_FS. It stashes
        # frame_hidden_raw and phone_hidden for us on _last_frame_hidden /
        # _last_phone_hidden.
        out = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ctc_weight=ctc_weight,
            align_weight=align_weight,
            contrastive_weight=0.0,          # parent must not add it
            pred_target_prob=pred_target_prob,
            prior_weight=prior_weight,
            skip_alignment=skip_alignment,
            encoder_alpha=encoder_alpha,
            lm_encoder_alpha=lm_encoder_alpha,
            sil_active=sil_active,
            sil_slots=sil_slots,
        )

        if skip_alignment or not self.training:
            return out

        X = self._last_frame_hidden
        phone_hidden = self._last_phone_hidden
        if X is None or phone_hidden is None:
            return out

        pm = out.predicted_phone_attention_mask

        if encoder_alpha is None:
            # Legacy routing (encoder_alpha None).
            if self.detach_contrastive_input:
                X = X.detach()
            energy_lm = out.energy
        else:
            # Joint routing. Targets and masked frames detached.
            X = X.detach()
            a = self._last_encoder_alpha or 0.0
            la = self._last_lm_alpha or 0.0
            if self.use_sil and sil_active and la != a:
                raise ValueError("latent SIL needs lm_encoder_alpha == encoder_alpha "
                                 "(L_m shares the forward-sum attention)")
            if la == a:
                # alpha = 0: out.energy is already detached.
                # alpha > 0: L_m shares the L_FS attention and its gradient
                # into the (top-k) encoder is scaled by the same alpha.
                energy_lm = out.energy
            else:
                # Different alpha for L_m: recompute its attention with its
                # own gradient scale. Same dropout masks as the L_FS pass
                # (replayed RNG), outer RNG untouched, so masking and
                # negatives are drawn exactly as in the shared case. The
                # prior term is re-added as a constant.
                X_top = self._last_top_hidden
                if la > 0.0:
                    lm_in = GradScale.apply(X_top, la)
                else:
                    lm_in = X_top.detach() if X_top is not None else X
                cpu_state, cuda_state = self._cnn_rng_state
                devs = [X.device] if X.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devs):
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, X.device)
                    frame_lm = self.cnn(lm_in)
                _, e_raw = self.attention(frame_lm, phone_hidden, pm)
                energy_lm = e_raw + (out.energy - out.energy_raw).detach()

        # Real (non-padded) encoder frames. The phone softmax gives every row,
        # padded ones included, one unit of mass; zero the padded rows so they
        # add nothing to YA or to the soft durations in _position_features.
        valid = (torch.arange(X.size(1), device=X.device)[None, :]
                 < out.input_lengths.to(X.device)[:, None])        # (B, T)
        sil = self._sil_state if (self.use_sil and sil_active) else None
        sil_occ = None                                             # (B, T) SIL mass
        if sil is None:
            att = torch.softmax(
                energy_lm + (1 - pm).unsqueeze(1).float() * -10000.0, dim=-1
            )                                                      # (B, T, P)
            att = att * valid.unsqueeze(-1).to(att.dtype)
            YA = torch.bmm(att.to(phone_hidden.dtype), phone_hidden)   # (B, T, d)
        else:
            Bsz, P = pm.shape
            vidx = sil["vidx"]
            key_sil = self.sil_key.to(phone_hidden.dtype)
            if self.lm_att_source == "fs_posterior":
                # Occupancy of every phone and every SIL slot from the
                # forward-backward over the extended sequence (differentiable:
                # L_m reaches the energies, sil_key and the entry cost through it).
                Uv = unit_occupancy(sil["gamma"], sil["top"], sil["n_units"])   # (Bv, T, 2P+1)
                U = Uv.new_zeros(Bsz, Uv.size(1), sil["n_units"]).index_copy(0, vidx, Uv)
                att = U * valid.unsqueeze(-1).to(U.dtype)          # units: phones | SIL slots
                K_units = torch.cat(
                    [phone_hidden, key_sil.view(1, 1, -1).expand(Bsz, P + 1, -1)], 1)
                YA = torch.bmm(att.to(phone_hidden.dtype), K_units)
                sil_occ = att[..., P:].sum(-1)
            else:
                # Per-frame softmax over [phones ; SIL] (the L_FS emissions).
                Pe = sil["logp_ext"].exp()                         # (Bv, T, P+1)
                Pf = Pe.new_zeros(Bsz, Pe.size(1), P + 1).index_copy(0, vidx, Pe)
                Pf = Pf * valid.unsqueeze(-1).to(Pf.dtype)
                att = Pf[..., :P]                                  # phone columns -> pos/dur
                sil_occ = Pf[..., P]
                YA = (torch.bmm(att.to(phone_hidden.dtype), phone_hidden)
                      + sil_occ.unsqueeze(-1).to(phone_hidden.dtype) * key_sil)

        X_quantized, q_probs = self.ema_quantizer(X.detach(), frame_mask=valid)
        # Codebook usage (perplexity of the average code distribution over
        # valid frames): watch it for a codebook collapse when L_m reaches the
        # encoder.
        with torch.no_grad():
            avg = q_probs[valid].float().mean(0)
            code_ppl = torch.exp(-(avg * torch.log(avg + 1e-7)).sum())
        X_masked, mask = self.mask_features(X, frame_mask=valid)

        if self.use_position_features:
            p = self._position_features(att, YA.dtype)
        else:
            # Zeros keep the head's input width identical, so the ablation
            # needs no architecture change and checkpoints stay compatible.
            p = torch.zeros(
                X_masked.size(0), X_masked.size(1), POS_FEAT_DIM,
                device=X_masked.device, dtype=YA.dtype,
            )

        H = self.reconstruction_head(torch.cat([X_masked, YA, p], dim=-1))

        codes = q_probs.argmax(-1) if self.lm_mask_same_code else None
        closs = self.contrastive_loss(H, X_quantized, mask, frame_mask=valid, codes=codes)

        sil_code = {}
        if sil_occ is not None:
            with torch.no_grad():
                w = sil_occ.detach().float()
                cd = (w.unsqueeze(-1) * q_probs.float()).sum((0, 1)) / w.sum().clamp(min=1e-6)
                sil_code["sil_code_top_share"] = cd.max()
                sil_code["sil_code_ppl"] = torch.exp(-(cd * torch.log(cd + 1e-7)).sum())

        # L_energy: every real frame, predicted from [YA ; pos/dur] only.
        # With use_sil it starts with SIL: before that, pause frames must sit
        # on phones, and the loss bends the phone attention to make them quiet
        # (this degraded the boundaries in early tests).
        eloss = None
        energy_on = self.lm_energy_weight > 0 and (sil_occ is not None or not self.use_sil)
        if energy_on or sil_occ is not None:
            e_tgt = frame_log_energy(input_values, attention_mask, X.size(1),
                                     out.input_lengths, self.lm_energy_floor_db,
                                     hop=self._frame_hop, win=self._frame_win)
            if sil_occ is not None:
                with torch.no_grad():
                    w = sil_occ.detach().float() * valid.float()
                    sil_code["sil_energy_db"] = 20.0 * (w * e_tgt).sum() / w.sum().clamp(min=1e-6)
            if energy_on:
                e_pred = self.energy_head(torch.cat([YA, p], dim=-1)).squeeze(-1).float()
                eloss = F.smooth_l1_loss(e_pred[valid], e_tgt[valid], beta=0.1)

        loss = out.loss
        lm = contrastive_weight * closs
        if eloss is not None:
            lm = lm + contrastive_weight * self.lm_energy_weight * eloss
        if loss is not None:
            loss = loss + lm
        else:
            loss = lm

        return SilAlignmentOutput(
            sil_energy_raw=getattr(out, "sil_energy_raw", None),
            sil_frame_frac=getattr(out, "sil_frame_frac", None),
            sil_used_frac=getattr(out, "sil_used_frac", None),
            sil_mean_dur=getattr(out, "sil_mean_dur", None),
            sil_logp_value=getattr(out, "sil_logp_value", None),
            **sil_code,
            loss=loss,
            ctc_loss=out.ctc_loss,
            align_loss=out.align_loss,
            contrastive_loss=closs,
            lm_energy_loss=eloss,
            codevector_perplexity=code_ppl,
            ctc_logits=out.ctc_logits,
            energy=out.energy,
            energy_raw=out.energy_raw,
            predicted_phone_ids=out.predicted_phone_ids,
            predicted_phone_attention_mask=out.predicted_phone_attention_mask,
            align_target_is_pred=out.align_target_is_pred,
            input_lengths=out.input_lengths,
            text_lengths=out.text_lengths,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )

    # ------------------------------------------------------------------
    # Diagnostic: is the head actually using the position features?
    # ------------------------------------------------------------------
    @torch.no_grad()
    def position_feature_usage(self):
        """Mean column norm of the head's first layer, split by input block.

        If the position block's norm stays far below the YA block's, the head
        is ignoring pos/dur and the experiment has failed — no need to wait
        for the F1 curve.
        """
        W = self.reconstruction_head[0].weight       # (1024, D + d + 64)
        D = self.config.hidden_size
        d = self.align_hidden_size
        return {
            "x_masked": float(W[:, :D].norm(dim=0).mean()),
            "ya": float(W[:, D:D + d].norm(dim=0).mean()),
            "pos": float(W[:, D + d:].norm(dim=0).mean()),
        }


@torch.no_grad()
def initialize_quantizer(model, dataloader, device, n_batches=200):
    """Seed the EMA codevectors from real encoder frames. Run this
    BEFORE trainer.train() or the codevectors start from noise."""
    model.eval()
    embeddings = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        input_values = batch["input_values"].to(device)
        am = batch.get("attention_mask")
        if am is not None:
            am = am.to(device)
        X = model.wav2vec2(input_values, attention_mask=am).last_hidden_state
        B, T, D = X.shape
        # Draw each frame from the utterance's real frames, not its padding.
        if am is not None:
            lens = model._get_feat_extract_output_lengths(am.sum(-1)).clamp(1, T).cpu()
        else:
            lens = torch.full((B,), T)
        idx = (torch.rand(B) * lens).long()
        embeddings.append(X[torch.arange(B), idx].cpu().float())

    embeddings = torch.cat(embeddings, dim=0)
    q = model.ema_quantizer
    perm = torch.randperm(embeddings.shape[0])[:q.num_vars]
    q.codevectors.copy_(embeddings[perm].to(device).to(q.codevectors.dtype))
    q.ema_embed.copy_(embeddings[perm].to(device).to(q.ema_embed.dtype))
    q.cluster_size.fill_(1.0)
    print(f"[train] EMA quantizer initialized from {embeddings.shape[0]} real frames")
    model.train()