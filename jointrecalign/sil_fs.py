#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent optional silence for the forward-sum aligner (MFA / Kaldi style).

Extended state sequence of one utterance with N phones and SIL slots 0..N
(slot i sits before phone i; slot N after the last phone):

    [SIL_0] p_0 [SIL_1] p_1 ... p_{N-1} [SIL_N]

Only the allowed slots exist (word boundaries + edges, edges only, or all).
A SIL slot is a chain of k = sil_min_frames tied sub-states: entered at the
first, self-loop on the last, so a used SIL lasts >= k frames.

Transitions (log space):
  stay       phone states; the last sub-state of a SIL chain
  advance    s-1 -> s; entering a SIL chain costs sil_logp
  skip       phone -> next phone over an unused SIL chain (no cost)
Start: first SIL sub-state of slot 0 (pays the entry cost) or phone 0.
End: phone N-1 or the last sub-state of slot N.
Phones are mandatory (>= 1 frame each); SIL is optional.

Emissions: one log-probability column per phone plus ONE shared SIL column
(all slots use the same learned key), i.e. logp_ext is (B, T, P + 1) with the
SIL column last.

forward_backward() runs both recursions with ordinary autograd ops, so
  logZ  -> L_FS = -logZ / N (mean over the batch, zero_infinity like CTCLoss)
  gamma -> frame x state occupancy, differentiable (used by L_m when
           lm_att_source = fs_posterior)
viterbi() is the matching decoder (numpy), used at evaluation.
"""

import numpy as np
import torch

NEG = -1e30


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def slot_mask(n_phones, mode, word_slots=None):
    """Allowed SIL slots 0..N as a bool array of length N+1.

    mode: "all"   every slot
          "edges" slots 0 and N
          "word"  slots listed in word_slots (word boundaries + edges); falls
                  back to "all" when word_slots is None
          "none"  no slot
    """
    m = np.zeros(n_phones + 1, dtype=bool)
    if mode == "none" or n_phones == 0:
        return m
    if mode == "all" or (mode == "word" and word_slots is None):
        m[:] = True
    elif mode == "edges":
        m[0] = m[n_phones] = True
    elif mode == "word":
        ws = np.asarray(word_slots, dtype=np.int64)
        ws = ws[(ws >= 0) & (ws <= n_phones)]
        m[ws] = True
    else:
        raise ValueError(f"unknown sil position mode {mode!r}")
    return m


def build_topology(n_phones, slots, k, sil_unit_offset=None):
    """One utterance -> dict of numpy arrays over its S states.

    col      emission column: phone index n, or -1 for SIL
    unit     phone n -> n, SIL slot i -> sil_unit_offset + i (default n_phones);
             in a batch the offset is the padded phone count P, so units are
             [phones 0..P-1 | SIL slots P..2P] for every utterance
    stay     self-loop allowed
    adv      can be entered from s-1
    entry    first sub-state of a SIL chain (entering pays sil_logp)
    skip     phone that can be entered from the phone before it, over a SIL chain
    init     allowed first state;  final  allowed last state
    """
    k = max(1, int(k))
    off = n_phones if sil_unit_offset is None else int(sil_unit_offset)
    col, unit, stay, entry, skip, init, final = [], [], [], [], [], [], []
    for i in range(n_phones + 1):
        if slots[i]:
            for j in range(k):
                col.append(-1)
                unit.append(off + i)
                stay.append(j == k - 1)
                entry.append(j == 0)
                skip.append(False)
                init.append(i == 0 and j == 0)
                final.append(i == n_phones and j == k - 1)
        if i < n_phones:
            col.append(i)
            unit.append(i)
            stay.append(True)
            entry.append(False)
            skip.append(bool(slots[i]) and i > 0)
            init.append(i == 0)
            final.append(i == n_phones - 1)
    S = len(col)
    adv = np.ones(S, dtype=bool)
    adv[0] = False
    return {"col": np.asarray(col, np.int64), "unit": np.asarray(unit, np.int64),
            "stay": np.asarray(stay), "adv": adv, "entry": np.asarray(entry),
            "skip": np.asarray(skip), "init": np.asarray(init), "final": np.asarray(final),
            "k": k, "n_phones": n_phones}


def batch_topology(n_phones_list, slots_list, k, device, p_max=None):
    """Pad per-utterance topologies to (B, S_max) tensors. SIL units start at
    p_max (default: the largest phone count)."""
    p_max = max(n_phones_list) if p_max is None else int(p_max)
    tops = [build_topology(n, s, k, p_max) for n, s in zip(n_phones_list, slots_list)]
    B = len(tops)
    S = max(1, max(len(t["col"]) for t in tops))
    out = {}
    for key, fill, dt in (("col", -1, torch.long), ("unit", 0, torch.long)):
        a = np.full((B, S), fill, np.int64)
        for b, t in enumerate(tops):
            a[b, :len(t[key])] = t[key]
        out[key] = torch.as_tensor(a, dtype=dt, device=device)
    for key in ("stay", "adv", "entry", "skip", "init", "final"):
        a = np.zeros((B, S), bool)
        for b, t in enumerate(tops):
            a[b, :len(t[key])] = t[key]
        out[key] = torch.as_tensor(a, device=device)
    valid = np.zeros((B, S), bool)
    for b, t in enumerate(tops):
        valid[b, :len(t["col"])] = True
    out["valid"] = torch.as_tensor(valid, device=device)
    out["is_sil"] = (out["col"] < 0) & out["valid"]
    out["k"] = max(1, int(k))
    out["n_states"] = torch.as_tensor([len(t["col"]) for t in tops], device=device)
    return out


# ---------------------------------------------------------------------------
# Forward-backward (autograd)
# ---------------------------------------------------------------------------
def _shift_right(x, n, fill=NEG):
    """y[:, s] = x[:, s - n] (fill for s < n)."""
    if n >= x.size(1):
        return torch.full_like(x, fill)
    return torch.cat([torch.full_like(x[:, :n], fill), x[:, :-n]], dim=1)


def _shift_left(x, n, fill=NEG):
    """y[:, s] = x[:, s + n] (fill for s >= S - n)."""
    if n >= x.size(1):
        return torch.full_like(x, fill)
    return torch.cat([x[:, n:], torch.full_like(x[:, :n], fill)], dim=1)


def state_emissions(logp_ext, top):
    """logp_ext (B, T, P+1), SIL column last -> (B, T, S) per-state emissions."""
    B, T, P1 = logp_ext.shape
    col = top["col"].clone()
    col[col < 0] = P1 - 1
    E = logp_ext.gather(2, col.unsqueeze(1).expand(B, T, -1))
    return E.masked_fill(~top["valid"].unsqueeze(1), NEG)


def forward_backward(logp_ext, top, input_lengths, sil_logp, need_gamma=True):
    """-> logZ (B,), gamma (B, T, S) or None.

    sil_logp: float or 0-d tensor (learned), added on every SIL entry.
    Frames t >= input_lengths[b] are ignored; gamma is 0 there.
    """
    E = state_emissions(logp_ext.float(), top)                       # (B, T, S)
    B, T, S = E.shape
    k = top["k"]
    dev = E.device
    sil_logp = torch.as_tensor(sil_logp, dtype=E.dtype, device=dev)
    zero = torch.zeros((), dtype=E.dtype, device=dev)
    neg = torch.full((), NEG, dtype=E.dtype, device=dev)
    cost_in = torch.where(top["entry"], sil_logp, zero)               # paid when entering s
    stay_ok, adv_ok, skip_ok = top["stay"], top["adv"], top["skip"]
    L = input_lengths.to(dev).long()

    def step_fwd(a):
        a_stay = torch.where(stay_ok, a, neg)
        a_adv = torch.where(adv_ok, _shift_right(a, 1) + cost_in, neg)
        a_skip = torch.where(skip_ok, _shift_right(a, k + 1), neg)
        return torch.logsumexp(torch.stack([a_stay, a_adv, a_skip]), dim=0)

    alpha = torch.where(top["init"], E[:, 0] + cost_in, neg)
    alphas = [alpha]
    for t in range(1, T):
        new = step_fwd(alpha) + E[:, t]
        alpha = torch.where((t < L).unsqueeze(1), new, alpha)
        alphas.append(alpha)
    logZ = torch.logsumexp(torch.where(top["final"], alpha, neg), dim=1)   # alpha frozen at L-1

    if not need_gamma:
        return logZ, None

    beta_last = torch.where(top["final"], zero, neg)
    beta = beta_last
    betas = [None] * T
    betas[T - 1] = beta
    for t in range(T - 2, -1, -1):
        X = E[:, t + 1] + beta                                    # successor scores
        b_stay = torch.where(stay_ok, X, neg)
        b_adv = torch.where(_shift_left(adv_ok.float(), 1, 0.0) > 0,
                            _shift_left(X + cost_in, 1), neg)
        b_skip = torch.where(_shift_left(skip_ok.float(), k + 1, 0.0) > 0,
                             _shift_left(X, k + 1), neg)
        new = torch.logsumexp(torch.stack([b_stay, b_adv, b_skip]), dim=0)
        beta = torch.where((t < L - 1).unsqueeze(1), new, beta_last)
        betas[t] = beta
    A = torch.stack(alphas, 1)
    Bt = torch.stack(betas, 1)
    gamma = torch.exp((A + Bt - logZ[:, None, None]).clamp(max=0.0))
    tv = (torch.arange(T, device=dev)[None, :] < L[:, None])
    ok = (logZ > NEG / 2)[:, None, None]
    gamma = gamma * (tv.unsqueeze(-1) & top["valid"].unsqueeze(1) & ok).to(gamma.dtype)
    return logZ, gamma


def fs_loss(logZ, n_phones):
    """CTCLoss(reduction='mean', zero_infinity=True) convention of ForwardSumLoss:
    mean over utterances of -logZ / N; impossible utterances contribute 0."""
    n = n_phones.to(logZ.dtype).clamp(min=1)
    per = -logZ / n
    per = torch.where(logZ > NEG / 2, per, torch.zeros_like(per))
    return per.mean()


def unit_occupancy(gamma, top, n_units):
    """gamma (B, T, S) -> (B, T, n_units): phone n -> unit n, SIL slot i -> P + i
    (n_units = 2P + 1 with the topology built with p_max = P)."""
    B, T, S = gamma.shape
    idx = top["unit"].unsqueeze(1).expand(B, T, S)
    U = torch.zeros(B, T, n_units, dtype=gamma.dtype, device=gamma.device)
    return U.scatter_add(2, idx, gamma)


# ---------------------------------------------------------------------------
# Viterbi (evaluation)
# ---------------------------------------------------------------------------
def viterbi(logp_ext, n_phones, slots, k, sil_logp):
    """logp_ext (T, N+1) numpy (phones then SIL column) -> state topology dict and
    best state per frame (array of length T), or (top, None) if T < N."""
    top = build_topology(n_phones, slots, k)
    T = logp_ext.shape[0]
    col = top["col"].copy()
    col[col < 0] = logp_ext.shape[1] - 1
    E = logp_ext[:, col].astype(np.float64)                    # (T, S)
    S = E.shape[1]
    if T < n_phones or S == 0:
        return top, None
    cost_in = np.where(top["entry"], float(sil_logp), 0.0)
    D = np.full((T, S), NEG)
    Bk = np.zeros((T, S), np.int64)
    D[0] = np.where(top["init"], E[0] + cost_in, NEG)
    idx = np.arange(S)
    kk = top["k"]
    for t in range(1, T):
        prev = D[t - 1]
        c_stay = np.where(top["stay"], prev, NEG)
        c_adv = np.where(top["adv"], np.concatenate([[NEG], prev[:-1]]) + cost_in, NEG)
        sh = np.concatenate([np.full(min(kk + 1, S), NEG), prev[:-(kk + 1)]]) if S > kk + 1 else np.full(S, NEG)
        c_skip = np.where(top["skip"], sh, NEG)
        C = np.stack([c_stay, c_adv, c_skip])
        a = C.argmax(0)
        D[t] = C[a, idx] + E[t]
        Bk[t] = np.where(a == 0, idx, np.where(a == 1, idx - 1, idx - (kk + 1)))
    fin = np.where(top["final"], D[T - 1], NEG)
    s = int(fin.argmax())
    if fin[s] <= NEG / 2:
        return top, None
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = Bk[t, s]
    return top, path
