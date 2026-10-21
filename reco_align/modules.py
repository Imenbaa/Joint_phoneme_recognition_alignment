#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Building blocks shared by the aligner: ForwardSum loss, ConvBank, cross-attention,
the model output dataclass and a base class with the CTC / prior helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import Wav2Vec2ForPreTraining
from transformers.file_utils import ModelOutput


class ForwardSumLoss(torch.nn.Module):
    """
    Implementation from:
    https://nv-adlr.github.io/one-tts-alignment

    LFS is computed over every encoder frame in the utterance.
    There is no VAD/speech-frame masking.
    """
    def __init__(self, blank_logprob=-1, prepend_blank=True):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.prepend_blank = prepend_blank
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
            attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
                batched tensor of attention log probabilities, padded.
            text_lens: batch-D vector of text sequence lengths.
            mel_lens: batch-D vector of encoder-frame sequence lengths.

        All valid encoder frames are used in the Forward-Sum loss.
        """
        if self.prepend_blank:
            attn_logprob_pd = F.pad(
                input=attn_logprob,
                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_logprob,
            )
        else:
            attn_logprob_pd = attn_logprob

        cost_total = 0.0
        B = attn_logprob.shape[0]

        for bid in range(B):
            target_seq = torch.arange(
                1,
                text_lens[bid] + 1,
                device=attn_logprob.device,
            ).unsqueeze(0)

            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[
                :mel_lens[bid], :, :text_lens[bid] + 1
            ]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]

            cost = self.CTCLoss(
                curr_logprob,
                target_seq,
                input_lengths=mel_lens[bid:bid + 1],
                target_lengths=text_lens[bid:bid + 1],
            )
            cost_total += cost

        return cost_total / B


class ConvBank(nn.Module):
    """
    Implementation from:
    https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/libri_phone/model.py
    """
    def __init__(
        self,
        input_dim,
        output_class_num,
        kernels,
        cnn_size,
        hidden_size,
        dropout,
        **kwargs
    ):
        super(ConvBank, self).__init__()
        self.drop_p = dropout

        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(
                nn.Conv1d(
                    latest_size,
                    cnn_size,
                    kernel,
                    padding=kernel // 2,
                )
            )
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(
            F.relu(self.in_linear(features)),
            p=self.drop_p,
            training=self.training,
        )

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()

        for cnn in self.cnns:
            conv_feats.append(cnn(hidden))

        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(
            F.relu(hidden),
            p=self.drop_p,
            training=self.training,
        )

        predicted = self.out_linear(hidden)
        return predicted


class Attention(nn.Module):
    """
    Cross-attention between frames (queries) and phones (keys).
    """
    def __init__(self, hidden_dim, scale=False):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = float(hidden_dim) ** -0.5 if scale else 1.0

    def forward(self, frame_hidden, phone_hidden, labels_attention_mask):
        frame_hidden = self.q(frame_hidden)
        phone_hidden = self.k(phone_hidden)

        energy = torch.bmm(
            frame_hidden,
            phone_hidden.transpose(2, 1),
        )
        if self.scale != 1.0:
            energy = energy * self.scale

        attention_mask = (1 - labels_attention_mask) * -10000.0
        energy = energy + attention_mask.unsqueeze(1).repeat(
            1, energy.size(1), 1
        )

        att_matrix = torch.softmax(energy, dim=-1)
        att_out = torch.bmm(att_matrix, phone_hidden)
        att_out = torch.cat([att_out, frame_hidden], dim=-1)

        return att_out, energy


@dataclass
class Wav2Vec2ForCTCAlignmentOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ctc_loss: Optional[torch.FloatTensor] = None
    align_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    codevector_perplexity: Optional[torch.FloatTensor] = None
    ctc_logits: torch.FloatTensor = None
    energy: torch.FloatTensor = None
    energy_raw: Optional[torch.FloatTensor] = None
    predicted_phone_ids: Optional[torch.LongTensor] = None
    predicted_phone_attention_mask: Optional[torch.LongTensor] = None
    align_target_is_pred: Optional[torch.BoolTensor] = None
    input_lengths: Optional[torch.LongTensor] = None
    text_lengths: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2AlignmentBase(Wav2Vec2ForPreTraining):
    """Helpers shared by the aligner models: backbone freezing and parameter
    groups, the RAD-TTS beta-binomial prior, and label handling. Subclasses
    build their own modules in __init__ and define forward()."""

    _PRIOR_CACHE_MAX = 8

    def freeze_feature_extractor(self):
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

    def get_backbone_parameters(self):
        return (
            list(self.wav2vec2.parameters())
            + list(self.lm_head.parameters())
        )

    # ------------------------------------------------------------------
    # RAD-TTS beta-binomial attention prior
    # ------------------------------------------------------------------
    def _beta_binomial_log_prior(self, M, N, device, dtype):
        """
        log P(n | t) for a beta-binomial prior over N phones and M frames.

        M now always corresponds to the complete encoder-frame sequence;
        there is no VAD/speech-frame selection.
        """
        s = self.prior_scaling_factor

        key = (
            int(M),
            int(N),
            s,
            str(device),
            str(dtype),
        )
        cached = self._prior_cache.get(key)
        if cached is not None:
            return cached

        i = torch.arange(
            1,
            M + 1,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        a = s * i
        b = s * (M + 1 - i)

        k = torch.arange(
            N,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        n = float(N - 1)
        lg = torch.lgamma

        log_binom = (
            lg(torch.tensor(n + 1.0, device=device))
            - lg(k + 1.0)
            - lg(n - k + 1.0)
        )

        log_beta_ratio = (
            lg(k + a)
            + lg(n - k + b)
            - lg(n + a + b)
            + lg(a + b)
            - lg(a)
            - lg(b)
        )

        log_pmf = (
            log_binom + log_beta_ratio
        ).clamp(min=self.prior_log_floor)

        log_pmf = log_pmf.to(dtype)

        if len(self._prior_cache) >= self._PRIOR_CACHE_MAX:
            self._prior_cache.clear()

        self._prior_cache[key] = log_pmf
        return log_pmf

    def _radtts_log_prior(
        self,
        T,
        P,
        input_lengths,
        text_lengths,
        device,
        dtype,
    ):
        """
        Batched [B, T, P] log-prior.

        The frame axis always runs over all valid encoder frames.
        """
        B = text_lengths.size(0)

        prior = torch.zeros(
            (B, T, P),
            device=device,
            dtype=dtype,
        )

        for b in range(B):
            N = int(text_lengths[b].item())

            if N < 1:
                continue

            M = int(input_lengths[b].item())

            if M < 1:
                continue

            lp = self._beta_binomial_log_prior(
                M,
                N,
                device,
                dtype,
            )

            if N < P:
                lp = F.pad(
                    lp,
                    (0, P - N),
                    value=0.0,
                )

            prior[b, :M, :] = lp

        return prior

    @staticmethod
    def _labels_to_phone_list(labels):
        """
        Ground-truth phone sequences from CTC labels.
        """
        out = []

        for i in range(labels.size(0)):
            row = labels[i]
            out.append(
                row[row >= 0].to(torch.long)
            )

        return out
