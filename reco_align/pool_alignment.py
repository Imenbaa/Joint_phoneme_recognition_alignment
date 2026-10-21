#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligner with phone keys pooled from the encoder (Wav2Vec2ForCTCPoolAlignment).

wav2vec2 encoder -> CTC head (L_CTC), and a cross-attention aligner:
queries = ConvBank over the encoder frames, keys = one vector per phone of the
target sequence. Each phone's key is a weighted average of encoder frames, the
weights being that phone's CTC posterior. L_FS is the ForwardSum loss on the
attention energy plus a beta-binomial prior.

Pooling modes (config.pool_mode):

  "global"      P(p_n | t) over every frame of the utterance.

  "global_pos"  as "global", plus a learnable embedding of the phone's rank, so
                that two occurrences of the same phoneme get different keys.

  "window"      the pooling weights are multiplied by a Gaussian window
                (config.window_sigma_frames) centred on the phone's frame span:
                the CTC forced alignment of the ground truth in training, the
                greedy CTC path otherwise. Occurrences of the same phoneme
                then pool different frames.

Latent optional silence (config.use_sil, see sil_fs.py) replaces the constant
blank of the ForwardSum loss by optional SIL states with one learned key.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForPreTraining

from .modules import (
    ConvBank,
    Attention,
    ForwardSumLoss,
    Wav2Vec2AlignmentBase,
    Wav2Vec2ForCTCAlignmentOutput,
)
from .sil_fs import (
    NEG as SIL_NEG,
    slot_mask,
    batch_topology,
    forward_backward,
    fs_loss,
    unit_occupancy,
)


@dataclass
class SilAlignmentOutput(Wav2Vec2ForCTCAlignmentOutput):
    """Wav2Vec2ForCTCAlignmentOutput + latent-SIL outputs.
    All new fields are None when config.use_sil is off."""
    sil_energy_raw: Optional[torch.FloatTensor] = None     # (B, T) q . k(sil_key), scaled
    sil_frame_frac: Optional[torch.FloatTensor] = None     # expected share of frames on SIL
    sil_used_frac: Optional[torch.FloatTensor] = None      # share of SIL slots with >= 0.5 frame
    sil_mean_dur: Optional[torch.FloatTensor] = None       # mean frames per used slot
    sil_logp_value: Optional[torch.FloatTensor] = None     # current entry cost
    sil_code_top_share: Optional[torch.FloatTensor] = None # L_m target codes on SIL frames
    sil_code_ppl: Optional[torch.FloatTensor] = None
    sil_energy_db: Optional[torch.FloatTensor] = None      # mean waveform dB (re p95) of SIL frames
    lm_energy_loss: Optional[torch.FloatTensor] = None     # L_energy (None when off)


class GradScale(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by alpha in the
    backward pass. Used to let L_FS reach the encoder at a reduced strength."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.alpha, None


class Wav2Vec2ForCTCPoolAlignment(Wav2Vec2AlignmentBase):

    POOL_MODES = ("global_pos", "window","global")

    def __init__(self, config):
        # Wav2Vec2ForPreTraining builds the wav2vec2 backbone, quantizer and
        # projections; the CTC head and the aligner are added below.
        Wav2Vec2ForPreTraining.__init__(self, config)

        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.pool_mode = getattr(config, "pool_mode", "global_pos")
        if self.pool_mode not in self.POOL_MODES:
            raise ValueError(
                f"config.pool_mode must be one of {self.POOL_MODES}, "
                f"got {self.pool_mode!r}."
            )

        # Attention dimension (default: wav2vec2 hidden size // 2).
        align_dim = int(
            getattr(config, "align_hidden_size", config.hidden_size // 2)
        )
        self.align_hidden_size = align_dim

        # Acoustic ConvBank. Kernels in frames: [1, 3, 5] at 20 ms; a 10 ms
        # model sets config.convbank_kernels to keep the span in ms.
        self.cnn = ConvBank(
            config.hidden_size,
            align_dim,
            list(getattr(config, "convbank_kernels", None) or [1, 3, 5]),
            align_dim,
            align_dim,
            getattr(config, "hidden_dropout", 0.1),
        )

        # Pooled vectors live in encoder space (hidden_size). The projection
        # maps them into the attention space and gives the aligner a space of
        # its own rather than the one lm_head linearly separates.
        self.phone_proj = nn.Sequential(
            nn.Linear(config.hidden_size, align_dim),
            nn.GELU(),
            nn.LayerNorm(align_dim),
        )

        self.max_phones = int(getattr(config, "max_phones", 1024))

        if self.pool_mode == "global_pos":
            self.phone_pos = nn.Embedding(self.max_phones, config.hidden_size)
            nn.init.normal_(self.phone_pos.weight, std=0.02)
        else:
            self.phone_pos = None

        self.window_sigma = float(
            getattr(config, "window_sigma_frames", 5.0)
        )
        self.gt_span_source = getattr(
            config, "gt_span_source", "forced_align"
        )

        self.attention = Attention(align_dim, scale=True)

        self.prior_scaling_factor = float(
            getattr(config, "prior_scaling_factor", 1.0)
        )
        self.prior_type = getattr(config, "prior_type", "beta_binomial")
        self.prior_log_floor = float(
            getattr(config, "prior_log_floor", -18.42)
        )
        self._prior_cache = {}

        self.detach_alignment_input = bool(
            getattr(config, "detach_alignment_input", False)
        )

        self.align_loss = ForwardSumLoss()

        # ----- latent optional SIL (off by default) -----
        # One learned key in phone_proj output space shared by every SIL slot,
        # an entry cost (fixed buffer or learned scalar) and the slot layout.
        self.use_sil = bool(getattr(config, "use_sil", False))
        self.sil_positions = getattr(config, "sil_positions", "word")
        self.sil_eval_positions = getattr(config, "sil_eval_positions", "all")
        self.sil_min_frames = int(getattr(config, "sil_min_frames", 1) or 1)
        self.lm_att_source = getattr(config, "lm_att_source", "softmax")
        self._sil_state = None
        self._warned_sil_word = False
        if self.use_sil:
            self.sil_key = nn.Parameter(torch.zeros(align_dim))
            lp = float(getattr(config, "sil_logp", -2.0))
            if bool(getattr(config, "sil_logp_learned", False)):
                self.sil_logp = nn.Parameter(torch.tensor(lp))
            else:
                self.register_buffer("sil_logp", torch.tensor(lp))
            # 1 once sil_key has been initialised (persisted, so a resumed run
            # does not re-initialise it).
            self.register_buffer("sil_initialized", torch.zeros((), dtype=torch.long))

        self._warned_truncation = False
        self._warned_forced_align = False
        self._last_frame_hidden = None
        # Restrict the encoder part of L_FS / L_m to the top-k
        # transformer layers (0 = all layers).
        self.encoder_top_layers = int(getattr(config, "encoder_top_layers", 0) or 0)
        self._topk_capture = None
        self._topk_hooks = False
        self._last_top_hidden = None
        self._last_lm_alpha = None
        self._last_phone_hidden = None
        self.post_init()
    def _quadratic_log_prior(self, T, P, input_lengths, text_lengths, device, dtype):
        """Quadratic diagonal prior: -10 * (n/N - t/T)^2, per sample."""
        B = text_lengths.size(0)
        prior = torch.zeros((B, T, P), device=device, dtype=dtype)
        for b in range(B):
            N, M = int(text_lengths[b].item()), int(input_lengths[b].item())
            if N < 1 or M < 1:
                continue
            t = torch.arange(M, device=device, dtype=torch.float32) / max(M - 1, 1)
            n = torch.arange(N, device=device, dtype=torch.float32) / max(N - 1, 1)
            prior[b, :M, :N] = (-10.0 * (n.unsqueeze(0) - t.unsqueeze(1)) ** 2).to(dtype)
        return prior
    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------
    def get_alignment_parameters(self):
        params = (
            list(self.cnn.parameters())
            + list(self.attention.parameters())
            + list(self.phone_proj.parameters())
        )
        if self.phone_pos is not None:
            params += list(self.phone_pos.parameters())
        if self.use_sil:
            params.append(self.sil_key)
            if isinstance(self.sil_logp, nn.Parameter):
                params.append(self.sil_logp)
        return params

    # ------------------------------------------------------------------
    # Latent SIL helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset_sil_parameters(self, seed_std=0.02):
        """Call after from_pretrained: HF does not know these parameters, so
        their values must be set explicitly (they are missing from the base)."""
        if not self.use_sil:
            return
        self.sil_key.normal_(0.0, seed_std)
        self.sil_logp.fill_(float(getattr(self.config, "sil_logp", -2.0)))
        self.sil_initialized.zero_()

    def _frame_db(self, input_values, T):
        """Per encoder frame log-energy (dB) over its conv receptive field."""
        rf, hop = 1, 1
        for k, s in zip(self.config.conv_kernel, self.config.conv_stride):
            rf += (k - 1) * hop
            hop *= s
        x = input_values.float()
        need = (T - 1) * hop + rf
        if x.size(1) < need:
            x = F.pad(x, (0, need - x.size(1)))
        return 10.0 * torch.log10(x[:, :need].unfold(1, rf, hop).pow(2).mean(-1) + 1e-10)

    @torch.no_grad()
    def init_sil_key_lowenergy(self, input_values, attention_mask=None, frac=0.05):
        """Matched-filter init: sil_key along W_k^T q_sil, where q_sil is the mean
        query of the quietest `frac` of the batch's frames (dB relative to each
        utterance's 95th percentile), so q . k(sil_key) is largest on those
        frames; its norm is the median norm of phone_proj outputs (the space of
        the phone keys). Training data only."""
        was_training = self.training
        self.eval()
        try:
            X = self.wav2vec2(input_values, attention_mask=attention_mask)[0]
            T = X.size(1)
            if attention_mask is not None:
                L = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()
            else:
                L = torch.full((X.size(0),), T, device=X.device, dtype=torch.long)
            q = self.attention.q(self.cnn(X)).float()                  # (B, T, d)
            db = self._frame_db(input_values, T)
            valid = torch.arange(T, device=X.device)[None, :] < L[:, None]
            rel = torch.full_like(db, float("inf"))
            for b in range(db.size(0)):
                v = db[b, : int(L[b])]
                if v.numel():
                    rel[b, : v.numel()] = v - torch.quantile(v, 0.95)
            r = rel[valid]
            n_sel = max(1, int(frac * r.numel()))
            thr = r.kthvalue(n_sel).values
            sel = valid & (rel <= thr)
            qm = q[sel].mean(0)
            z = self.attention.k.weight.float().t() @ qm                  # d/dz of qm . (W z)
            key_norm = self.phone_proj(X[valid]).float().norm(dim=-1).median()
            z = z / z.norm().clamp(min=1e-6) * key_norm
            self.sil_key.copy_(z.to(self.sil_key.dtype))
            self.sil_initialized.fill_(1)
            return int(sel.sum()), float(thr)
        finally:
            self.train(was_training)

    def _sil_slot_list(self, target_phones, sil_slots, use_pred, mode):
        """Allowed SIL slots (bool array of length N+1) per utterance.
        sil_slots: (B, L) long, word-boundary slot indices padded with -1, or None."""
        out = []
        for i, p in enumerate(target_phones):
            n = int(p.size(0))
            ws = None
            if mode == "word" and sil_slots is not None and not bool(use_pred[i]):
                row = sil_slots[i]
                row = row[row >= 0]
                if row.numel():
                    ws = row.tolist()
            if mode == "word" and ws is None and not self._warned_sil_word:
                print("[sil] WARNING: --sil_positions word without word boundaries for "
                      "some utterances -> 'all' for those. (warned once)")
                self._warned_sil_word = True
            out.append(slot_mask(n, mode, ws))
        return out

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    def _pool_phone_embeddings(
        self,
        X,               # (B, T, D) encoder frames
        ctc_logits,      # (B, T, V)
        phone_ids,       # (B, N)
        phone_mask,      # (B, N)  1 for real phones
        frame_mask,      # (B, T)  1 for valid frames
        centres=None,    # (B, N)  float frame index, "window" mode only
    ):
        """One gather + one bmm. No Python loop over B or N."""
        B, T, D = X.shape

        # Posterior weights are always detached: L_FS must not reach lm_head
        # through the pooling weights, and the greedy decode is in any case
        # non-differentiable.
        probs = ctc_logits.detach().float().softmax(dim=-1)          # (B,T,V)

        idx = phone_ids.unsqueeze(1).expand(-1, T, -1)               # (B,T,N)
        W = probs.gather(2, idx)                                     # (B,T,N)

        W = W * frame_mask.unsqueeze(2).float()

        if centres is not None:
            t = torch.arange(
                T, device=X.device, dtype=torch.float32
            ).view(1, T, 1)
            win = torch.exp(
                -((t - centres.unsqueeze(1).float()) ** 2)
                / (2.0 * self.window_sigma ** 2)
            )
            win = win * frame_mask.unsqueeze(2).float()
            Ww = W * win
            # A phone whose posterior is ~0 inside its own window (e.g. a
            # ground-truth phone the model never emits) would otherwise get
            # noise/noise. Fall back to the window alone: a plain local
            # acoustic average.
            dead = Ww.sum(dim=1, keepdim=True) < 1e-6
            W = torch.where(dead.expand_as(Ww), win, Ww)

        W = W * phone_mask.unsqueeze(1).float()
        denom = W.sum(dim=1, keepdim=True)
        W = torch.where(denom > 1e-6, W / denom.clamp(min=1e-8), torch.zeros_like(W))
        Y = torch.bmm(W.transpose(1, 2).float(), X.float()).to(X.dtype)   # (B,N,D)

        if self.phone_pos is not None:
            N = phone_ids.size(1)
            pos = torch.arange(N, device=X.device).clamp(
                max=self.max_phones - 1
            )
            Y = Y + self.phone_pos(pos).unsqueeze(0).to(Y.dtype)

        return self.phone_proj(Y)                                    # (B,N,d)

    # ------------------------------------------------------------------
    # Frame spans
    # ------------------------------------------------------------------
    @staticmethod
    def _greedy_decode_with_spans(ctc_logits, blank_id, input_lengths):
        """Argmax -> collapse repeats -> drop blanks, keeping the frame run
        that produced each emitted token. Returns (ids, centres) per sample."""
        argmax = ctc_logits.argmax(dim=-1)
        device = ctc_logits.device

        ids_out, centres_out = [], []

        for i in range(argmax.size(0)):
            seq = argmax[i, : int(input_lengths[i])].tolist()

            ids, centres = [], []
            prev, run_start = -1, 0

            for t, tok in enumerate(seq):
                if tok != prev:
                    if prev != -1 and prev != blank_id:
                        ids.append(prev)
                        centres.append((run_start + t - 1) / 2.0)
                    prev, run_start = tok, t

            if prev != -1 and prev != blank_id:
                ids.append(prev)
                centres.append((run_start + len(seq) - 1) / 2.0)

            ids_out.append(
                torch.tensor(ids, dtype=torch.long, device=device)
            )
            centres_out.append(
                torch.tensor(centres, dtype=torch.float32, device=device)
            )

        return ids_out, centres_out

    @staticmethod
    def _diagonal_centres(n_phones, n_frames, device):
        """c_n = (n + 0.5) * M / N. Used when no alignment is available."""
        if n_phones < 1:
            return torch.zeros(0, dtype=torch.float32, device=device)
        n = torch.arange(n_phones, device=device, dtype=torch.float32)
        return (n + 0.5) * float(n_frames) / float(n_phones)

    def _gt_centres(self, ctc_logits, gt_phones, input_lengths):
        """Frame centre of each ground-truth phone, from a CTC forced
        alignment of the ground-truth sequence against the posteriors.
        Falls back to the diagonal if torchaudio is unavailable or the
        alignment fails for a sample."""
        device = ctc_logits.device
        blank_id = self.config.pad_token_id

        use_fa = self.gt_span_source == "forced_align"
        fa = merge = None

        if use_fa:
            try:
                from torchaudio.functional import forced_align, merge_tokens
                fa, merge = forced_align, merge_tokens
            except Exception as e:
                if not self._warned_forced_align:
                    print(
                        f"[Wav2Vec2ForCTCPoolAlignment] torchaudio "
                        f"forced_align unavailable ({e}); falling back to "
                        f"diagonal centres for ground-truth targets. "
                        f"(warned once)"
                    )
                    self._warned_forced_align = True
                use_fa = False

        with torch.no_grad():
            log_probs = F.log_softmax(
                ctc_logits.detach().float(), dim=-1
            )

            out = []
            for b, p in enumerate(gt_phones):
                N = int(p.size(0))
                M = int(input_lengths[b].item())

                if N < 1 or M < 1:
                    out.append(
                        torch.zeros(N, dtype=torch.float32, device=device)
                    )
                    continue

                centres = None
                if use_fa and N <= M:
                    try:
                        al, sc = fa(
                            log_probs[b : b + 1, :M],
                            p.unsqueeze(0).to(torch.int32),
                            blank=blank_id,
                        )
                        spans = merge(al[0], sc[0], blank=blank_id)
                        if len(spans) == N:
                            centres = torch.tensor(
                                [
                                    (s.start + s.end - 1) / 2.0
                                    for s in spans
                                ],
                                dtype=torch.float32,
                                device=device,
                            )
                    except Exception:
                        centres = None

                if centres is None:
                    centres = self._diagonal_centres(N, M, device)

                out.append(centres)

        return out

    # ------------------------------------------------------------------
    # Top-k re-computation (config.encoder_top_layers)
    # ------------------------------------------------------------------
    def _install_topk_hooks(self, k):
        """Forward-pre-hooks on the last k transformer layers. When a capture is
        active they record each layer's exact inputs and the RNG state, so the
        layers can be re-run on a detached input with identical dropout."""
        if self._topk_hooks == k:
            return
        if self._topk_hooks:
            raise RuntimeError("encoder_top_layers changed after the hooks were installed")
        layers = self.wav2vec2.encoder.layers
        if k > len(layers):
            raise ValueError(f"encoder_top_layers={k} > {len(layers)} layers")

        def make_hook(i):
            def hook(module, args, kwargs):
                cap = self._topk_capture
                if cap is not None:
                    h = args[0] if args else kwargs.get("hidden_states")
                    dev = h.device
                    cap[i] = (args, kwargs, torch.get_rng_state(),
                              torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None)
                return None
            return hook

        for i, layer in enumerate(layers[-k:]):
            layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        self._topk_hooks = k

    def _recompute_top(self, frame_hidden_raw):
        k = self.encoder_top_layers
        cap, self._topk_capture = self._topk_capture, None
        if not cap or len(cap) != k:
            # a layer was skipped (layerdrop > 0): fall back to all layers
            if not getattr(self, "_warned_topk", False):
                print("[Wav2Vec2ForCTCPoolAlignment] top-k capture incomplete "
                      "(layerdrop?); using all layers for this step. (warned once)")
                self._warned_topk = True
            return frame_hidden_raw
        enc = self.wav2vec2.encoder
        layers = enc.layers[-k:]
        args0, kw0, _, _ = cap[0]
        h = (args0[0] if args0 else kw0["hidden_states"]).detach()
        for i, layer in enumerate(layers):
            args, kwargs, cpu_s, cuda_s = cap[i]
            kwargs = dict(kwargs)
            if not args:
                kwargs.pop("hidden_states", None)
            dev = h.device
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.set_rng_state(cpu_s)
                if cuda_s is not None:
                    torch.cuda.set_rng_state(cuda_s, dev)
                out = layer(h, *args[1:], **kwargs)
            h = out[0] if isinstance(out, (tuple, list)) else out
        if getattr(self.config, "do_stable_layer_norm", False):
            h = enc.layer_norm(h)
        return h

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
        """sil_active: use the latent-SIL forward-sum in place of the
             RAD-TTS constant blank (needs config.use_sil). sil_slots: (B, L)
             word-boundary slot indices (-1 padded) for --sil_positions word.
        encoder_alpha (joint routing):
             None  -> legacy routing, unchanged: queries AND keys attached to
                      the encoder unless config.detach_alignment_input is set.
             float -> joint routing: keys always detached; queries receive
                      the L_FS gradient scaled by encoder_alpha (0 = detached).
                      config.detach_alignment_input=True forces it to 0.
           lm_encoder_alpha: same for L_m (used by the contrastive subclass);
                      None or 0 = L_m never reaches the encoder.
           config.encoder_top_layers = k > 0: the encoder gradient of L_FS and
                      L_m reaches only the last k transformer layers (and the
                      final layer norm). CTC still trains every layer.
        """
        self._last_encoder_alpha = None
        self._last_lm_alpha = None
        self._last_top_hidden = None
        # alphas are known before the encoder runs: decide whether the top-k
        # layers' inputs must be captured for re-computation.
        if encoder_alpha is not None:
            _a = 0.0 if self.detach_alignment_input else float(encoder_alpha)
            _la = 0.0 if lm_encoder_alpha is None else float(lm_encoder_alpha)
            _need = (max(_a, _la) > 0.0) and torch.is_grad_enabled() and not skip_alignment
        else:
            _a = _la = 0.0
            _need = False
        k = self.encoder_top_layers
        if _need and k > 0:
            self._install_topk_hooks(k)
            self._topk_capture = {}
        else:
            self._topk_capture = None
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
        )

        frame_hidden_raw = outputs[0]
        ctc_logits = self.lm_head(self.dropout(frame_hidden_raw))

        if attention_mask is not None:
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
        else:
            input_lengths = torch.full(
                (input_values.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=ctc_logits.device,
            )

        # ----- L_CTC -----
        ctc_loss = None

        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: "
                    f"{self.config.vocab_size}"
                )

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(
                ctc_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if skip_alignment:
            loss = (
                ctc_weight * ctc_loss if ctc_loss is not None else None
            )
            return Wav2Vec2ForCTCAlignmentOutput(
                loss=loss,
                ctc_loss=ctc_loss,
                ctc_logits=ctc_logits,
                input_lengths=input_lengths,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        B = input_values.size(0)
        device = ctc_logits.device
        windowed = self.pool_mode == "window"

        # ----- Alignment target: ground truth vs CTC prediction -----
        gt_phones = (
            self._labels_to_phone_list(labels)
            if labels is not None
            else None
        )

        need_pred = (gt_phones is None) or (float(pred_target_prob) > 0.0)

        if need_pred:
            with torch.no_grad():
                pred_phones, pred_centres = self._greedy_decode_with_spans(
                    ctc_logits, self.config.pad_token_id, input_lengths
                )
        else:
            pred_phones, pred_centres = None, None

        if gt_phones is None:
            use_pred = torch.ones(B, dtype=torch.bool, device=device)
        elif pred_phones is None:
            use_pred = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            use_pred = (
                torch.rand(B, device=device) < float(pred_target_prob)
            )

        gt_centres = None
        if windowed and gt_phones is not None and (~use_pred).any():
            gt_centres = self._gt_centres(
                ctc_logits, gt_phones, input_lengths
            )

        target_phones, target_centres = [], []
        for i in range(B):
            if bool(use_pred[i]):
                target_phones.append(pred_phones[i])
                target_centres.append(
                    pred_centres[i] if windowed else None
                )
            else:
                target_phones.append(gt_phones[i])
                target_centres.append(
                    gt_centres[i] if windowed else None
                )

        # ----- Truncation guard -----
        if self.pool_mode == "global_pos":
            cap = self.max_phones
            trunc = [p[:cap] for p in target_phones]

            if not self._warned_truncation:
                n_trunc = sum(
                    1
                    for a, b in zip(target_phones, trunc)
                    if a.size(0) != b.size(0)
                )
                if n_trunc > 0:
                    print(
                        f"[Wav2Vec2ForCTCPoolAlignment] WARNING: {n_trunc} "
                        f"phone sequence(s) truncated to max_phones={cap}. "
                        f"The alignment target then covers only part of the "
                        f"utterance while L_FS still spans every frame. "
                        f"Raise config.max_phones or filter long utterances. "
                        f"(warned once)"
                    )
                    self._warned_truncation = True

            target_phones = trunc
            if windowed:
                target_centres = [
                    c[:cap] if c is not None else None
                    for c in target_centres
                ]

        text_lengths = torch.tensor(
            [p.size(0) for p in target_phones],
            dtype=torch.long,
            device=device,
        )
        max_phones = max(int(text_lengths.max().item()), 1)

        phone_ids = torch.zeros(
            B, max_phones, dtype=torch.long, device=device
        )
        phone_attn_mask = torch.zeros(
            B, max_phones, dtype=torch.long, device=device
        )
        centres = (
            torch.zeros(B, max_phones, dtype=torch.float32, device=device)
            if windowed
            else None
        )

        for i, p in enumerate(target_phones):
            n = p.size(0)
            if n > 0:
                phone_ids[i, :n] = p
                phone_attn_mask[i, :n] = 1
                if windowed:
                    c = target_centres[i]
                    centres[i, :n] = c[:n].to(device)

        # ----- Alignment branch -----
        if encoder_alpha is None:
            # Legacy routing (encoder_alpha None): same tensor for queries and keys.
            align_input = (
                frame_hidden_raw.detach()
                if self.detach_alignment_input
                else frame_hidden_raw
            )
            query_input = key_input = align_input
        else:
            # Joint routing: keys never reach the encoder; the query
            # path carries L_FS into it, scaled by alpha.
            alpha, lm_alpha = (_a, _la) if _need else (0.0, 0.0)
            key_input = frame_hidden_raw.detach()
            if _need:
                # X_top: same values as frame_hidden_raw, but its graph reaches
                # only the top-k layers (all layers when k = 0).
                X_top = (self._recompute_top(frame_hidden_raw)
                         if k > 0 else frame_hidden_raw)
                self._last_top_hidden = X_top
            if alpha > 0.0:
                query_input = GradScale.apply(X_top, alpha)
            else:
                query_input = key_input
            self._last_encoder_alpha = alpha
            self._last_lm_alpha = lm_alpha

        # Stash the RNG state so a subclass can re-run the ConvBank on a
        # detached copy with the SAME dropout masks (see the contrastive model).
        self._cnn_rng_state = None
        if encoder_alpha is not None and _need:
            dev = frame_hidden_raw.device
            self._cnn_rng_state = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
            )

        frame_hidden = self.cnn(query_input)

        T_all = ctc_logits.size(1)
        frame_mask = (
            torch.arange(T_all, device=device).unsqueeze(0)
            < input_lengths.unsqueeze(1)
        ).long()

        phone_hidden = self._pool_phone_embeddings(
            X=key_input,
            ctc_logits=ctc_logits,
            phone_ids=phone_ids,
            phone_mask=phone_attn_mask,
            frame_mask=frame_mask,
            centres=centres,
        )
        self._last_frame_hidden = frame_hidden_raw
        self._last_phone_hidden = phone_hidden
        sil_energy_raw = None
        if self.use_sil:
            # One extra key column: phone energies are unchanged (each column is
            # q . k(key) on its own); column P is the SIL energy.
            key_ext = torch.cat(
                [phone_hidden,
                 self.sil_key.to(phone_hidden.dtype).view(1, 1, -1).expand(B, 1, -1)], 1)
            mask_ext = torch.cat(
                [phone_attn_mask, torch.ones_like(phone_attn_mask[:, :1])], 1)
            att_out, energy_ext = self.attention(frame_hidden, key_ext, mask_ext)
            energy_raw, sil_energy_raw = energy_ext[..., :-1], energy_ext[..., -1]
        else:
            att_out, energy_raw = self.attention(
                frame_hidden, phone_hidden, phone_attn_mask
            )

        # ----- RAD-TTS beta-binomial prior -----
        Bsz, T_, P_ = energy_raw.shape

        if float(prior_weight) > 0.0:
            prior_fn = (self._quadratic_log_prior
                        if self.prior_type == "quadratic"
                        else self._radtts_log_prior)
            log_prior = prior_fn(
                T=T_, P=P_, input_lengths=input_lengths,
                text_lengths=text_lengths, device=energy_raw.device,
                dtype=energy_raw.dtype,
            )
            energy = energy_raw + float(prior_weight) * log_prior
        else:
            energy = energy_raw

        contrastive_loss = None
        codevector_perplexity = None

        # ----- L_FS -----
        align_loss_value = None

        self._sil_state = None
        sil_stats = {}
        if self.training and self.use_sil and sil_active:
            valid = text_lengths > 0
            if valid.any():
                # Latent SIL: softmax over [phones ; SIL] per frame. The SIL column
                # takes the prior of the best-placed phone at that frame (neutral:
                # SIL is never favoured or penalised by the prior relative to the
                # diagonal); its position in time is constrained by the DP.
                pm_neg = (1 - phone_attn_mask).float() * SIL_NEG
                sil_col = sil_energy_raw.float()
                if float(prior_weight) > 0.0:
                    sil_col = sil_col + float(prior_weight) * (
                        log_prior.float() + pm_neg.unsqueeze(1)).max(-1).values
                full = torch.cat(
                    [(energy.float() + (1 - phone_attn_mask).float().unsqueeze(1) * -10000.0),
                     sil_col.unsqueeze(-1)], dim=-1)
                vidx = valid.nonzero(as_tuple=True)[0]
                slots = self._sil_slot_list(
                    [target_phones[i] for i in vidx.tolist()],
                    sil_slots[vidx] if sil_slots is not None else None,
                    use_pred[vidx], self.sil_positions)
                with torch.autocast(device_type=device.type, enabled=False):
                    logp_ext = torch.log_softmax(full[vidx], dim=-1)            # (Bv, T, P+1)
                    top = batch_topology(text_lengths[vidx].tolist(), slots,
                                         self.sil_min_frames, device, p_max=P_)
                    logZ, gamma = forward_backward(
                        logp_ext, top, input_lengths[vidx], self.sil_logp.float())
                    align_loss_value = fs_loss(logZ, text_lengths[vidx])
                n_units = 2 * P_ + 1
                self._sil_state = {"vidx": vidx, "gamma": gamma, "top": top,
                                   "logp_ext": logp_ext, "n_units": n_units}
                with torch.no_grad():
                    U = unit_occupancy(gamma.detach(), top, n_units)            # (Bv, T, 2P+1)
                    tv = (torch.arange(U.size(1), device=device)[None, :]
                          < input_lengths[vidx][:, None]).float()
                    s_t = U[..., P_:].sum(-1)
                    sil_stats["sil_frame_frac"] = (s_t * tv).sum() / tv.sum().clamp(min=1)
                    m = U[..., P_:].sum(1)                                      # (Bv, P+1)
                    allowed = torch.zeros_like(m, dtype=torch.bool)
                    unit_sil = top["unit"].masked_fill(~top["is_sil"], 0) - P_
                    allowed.scatter_(1, unit_sil.clamp(min=0), top["is_sil"])
                    used = (m >= 0.5) & allowed
                    sil_stats["sil_used_frac"] = used.sum() / allowed.sum().clamp(min=1)
                    sil_stats["sil_mean_dur"] = (m * used).sum() / used.sum().clamp(min=1)
                    sil_stats["sil_logp_value"] = self.sil_logp.detach().float()
        elif self.training:
            valid = text_lengths > 0
            if valid.any():
                expanded_mask = (
                    (1 - phone_attn_mask).float() * -10000.0
                )
                expanded_mask = expanded_mask.unsqueeze(1).expand(
                    -1, energy.size(1), -1
                )
                att = torch.log_softmax(energy + expanded_mask, dim=-1)
                att_v = att[valid].unsqueeze(1)
                align_loss_value = self.align_loss(
                    att_v, text_lengths[valid], input_lengths[valid]
                )

        # ----- Total -----
        loss = None
        if any(
            x is not None
            for x in (ctc_loss, align_loss_value, contrastive_loss)
        ):
            loss = torch.zeros((), device=ctc_logits.device)
            if ctc_loss is not None:
                loss = loss + ctc_weight * ctc_loss
            if align_loss_value is not None:
                loss = loss + align_weight * align_loss_value
            if contrastive_loss is not None:
                loss = loss + contrastive_weight * contrastive_loss

        return SilAlignmentOutput(
            sil_energy_raw=sil_energy_raw,
            **sil_stats,
            loss=loss,
            ctc_loss=ctc_loss,
            align_loss=align_loss_value,
            contrastive_loss=contrastive_loss,
            codevector_perplexity=codevector_perplexity,
            ctc_logits=ctc_logits,
            energy=energy,
            energy_raw=energy_raw,
            predicted_phone_ids=phone_ids,
            predicted_phone_attention_mask=phone_attn_mask,
            align_target_is_pred=use_pred,
            input_lengths=input_lengths,
            text_lengths=text_lengths,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )