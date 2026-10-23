#!/usr/bin/env python3
"""CPU tests of jointrecalign.sil_fs (no model): python tests/test_sil_fs.py

1. logZ == brute force over all valid segmentations (positions all/edges/word/none,
   min frames 1-3, several entry costs)
2. gamma == d logZ / d emissions (autograd), rows sum to 1 over valid frames
3. batching / padding gives the per-utterance values
4. SIL everywhere, k = 1, entry cost 0, SIL column = RAD-TTS blank
   == ForwardSumLoss (CTCLoss) of jointrecalign/modules.py, i.e. L_FS without SIL
5. viterbi score == brute-force max, path is a valid segmentation
"""
import itertools
import math

import numpy as np
import torch

from jointrecalign import sil_fs as SF

torch.manual_seed(0)
rng = np.random.default_rng(0)


def segmentations(T, N, slots, k):
    """All label sequences (per frame: phone n, or ('S', i) for SIL slot i) of length T."""
    out = []

    def rec(i, t, seq, n_sil):
        # at slot i (before phone i), t frames used
        opts = [0]
        if slots[i]:
            opts += list(range(k, T - t + 1))
        for ls in opts:
            if t + ls > T:
                continue
            seq2 = seq + [("S", i)] * ls
            ns = n_sil + (ls > 0)
            if i == N:
                if t + ls == T:
                    out.append((seq2, ns))
                continue
            for lp in range(1, T - t - ls + 1):
                rec(i + 1, t + ls + lp, seq2 + [i] * lp, ns)

    rec(0, 0, [], 0)
    return out


def brute(logp, N, slots, k, c):
    T = logp.shape[0]
    scores = []
    for seq, ns in segmentations(T, N, slots, k):
        sc = sum(logp[t, N if isinstance(x, tuple) else x] for t, x in enumerate(seq)) + c * ns
        scores.append((sc, seq))
    if not scores:
        return -math.inf, -math.inf, None
    v = np.array([s for s, _ in scores])
    m = v.max()
    best = scores[int(v.argmax())][1]
    return m + math.log(np.exp(v - m).sum()), m, best


def test_logz_and_viterbi():
    n = 0
    for N in (1, 2, 3):
        for T in range(N, 8):
            for mode in ("all", "edges", "word", "none"):
                ws = sorted(set([0, N] + ([1] if N > 1 else []))) if mode == "word" else None
                slots = SF.slot_mask(N, mode, ws)
                for k in (1, 2, 3):
                    for c in (0.0, -2.0, -0.7):
                        logp = torch.log_softmax(torch.randn(T, N + 1, dtype=torch.float64) * 2, -1)
                        ref, ref_max, best = brute(logp.numpy(), N, slots, k, c)
                        top = SF.batch_topology([N], [slots], k, "cpu")
                        lz, g = SF.forward_backward(logp[None].float(), top, torch.tensor([T]), c)
                        if ref == -math.inf:
                            assert lz.item() < SF.NEG / 2, (N, T, mode, k, c)
                            continue
                        assert abs(lz.item() - ref) < 1e-3, (N, T, mode, k, c, lz.item(), ref)
                        tp, path = SF.viterbi(logp.numpy(), N, slots, k, c)
                        col = tp["col"]
                        lab = [("S", None) if col[s] < 0 else int(col[s]) for s in path]
                        sc = sum(logp[t, N if isinstance(x, tuple) else x].item() for t, x in enumerate(lab))
                        sc += c * sum(1 for t in range(T) if tp["entry"][path[t]] and
                                      (t == 0 or path[t - 1] != path[t]))
                        assert abs(sc - ref_max) < 1e-6, (N, T, mode, k, c, sc, ref_max)
                        n += 1
    print(f"[ok] logZ and viterbi match brute force on {n} configurations")


def test_gamma_is_gradient():
    N, T, k = 4, 12, 2
    slots = SF.slot_mask(N, "all")
    logp = torch.log_softmax(torch.randn(1, T, N + 1), -1).requires_grad_(True)
    top = SF.batch_topology([N], [slots], k, "cpu")
    lz, g = SF.forward_backward(logp, top, torch.tensor([T]), -1.5)
    lz.sum().backward()
    # d logZ / d logp[t, col] = sum of gamma over the states emitting that column
    G = torch.zeros(1, T, N + 1)
    col = top["col"][0].clone()
    col[col < 0] = N
    G.index_add_(2, col, g.detach())
    assert torch.allclose(G, logp.grad, atol=1e-4), (G - logp.grad).abs().max()
    assert torch.allclose(g.sum(-1), torch.ones(1, T), atol=1e-4)
    print("[ok] gamma == d logZ / d emissions; occupancy sums to 1 per frame")


def test_batching():
    Ns, Ts, k = [3, 5, 2], [9, 14, 4], 2
    P, Tm = max(Ns), max(Ts)
    slots = [SF.slot_mask(n, "all") for n in Ns]
    logp = torch.log_softmax(torch.randn(3, Tm, P + 1), -1)
    top = SF.batch_topology(Ns, slots, k, "cpu")
    lz, g = SF.forward_backward(logp, top, torch.tensor(Ts), -2.0)
    for b in range(3):
        sub = torch.cat([logp[b:b + 1, :Ts[b], :Ns[b]], logp[b:b + 1, :Ts[b], P:]], -1)
        t1 = SF.batch_topology([Ns[b]], [slots[b]], k, "cpu")
        l1, g1 = SF.forward_backward(sub, t1, torch.tensor([Ts[b]]), -2.0)
        assert abs(l1.item() - lz[b].item()) < 1e-4, (b, l1.item(), lz[b].item())
        S1 = g1.size(-1)
        assert torch.allclose(g1[0], g[b, :Ts[b], :S1], atol=1e-4)
        assert g[b, Ts[b]:].abs().max().item() == 0 if Ts[b] < Tm else True
    print("[ok] padded batch == per-utterance logZ and gamma")


def test_ctc_equivalence():
    from jointrecalign.modules import ForwardSumLoss
    B, Ns, Ts = 3, [4, 6, 3], [15, 20, 9]
    P, Tm = max(Ns), max(Ts)
    energy = torch.randn(B, Tm, P) * 3
    pm = torch.zeros(B, P)
    for b, n in enumerate(Ns):
        pm[b, :n] = 1
    att = torch.log_softmax(energy + (1 - pm[:, None, :]) * -10000.0, -1)
    ref = ForwardSumLoss()(att.unsqueeze(1), torch.tensor(Ns), torch.tensor(Ts))
    # RAD-TTS: blank -1 prepended to the phone log-softmax, renormalised
    blank = torch.full((B, Tm, 1), -1.0)
    pad = torch.log_softmax(torch.cat([blank, att], -1), -1)   # column 0 = blank
    ext = torch.cat([pad[..., 1:], pad[..., :1]], -1)          # phones, then SIL = blank
    slots = [SF.slot_mask(n, "all") for n in Ns]
    top = SF.batch_topology(Ns, slots, 1, "cpu", p_max=P)
    lz, _ = SF.forward_backward(ext, top, torch.tensor(Ts), 0.0, need_gamma=False)
    ours = SF.fs_loss(lz, torch.tensor(Ns))
    assert abs(ours.item() - ref.item()) < 1e-4, (ours.item(), ref.item())
    print(f"[ok] SIL everywhere + RAD-TTS blank == ForwardSumLoss: {ours.item():.5f} vs {ref.item():.5f}")


if __name__ == "__main__":
    test_logz_and_viterbi()
    test_gamma_is_gradient()
    test_batching()
    test_ctc_equivalence()
    print("all sil_fs tests passed")
