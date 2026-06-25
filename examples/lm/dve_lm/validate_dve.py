"""Correctness tests for the Data Value Embedding implementation.

These run on CPU in seconds and need no GPU or corpus:

  Test 1 (recursion fidelity): an independent, literal port of the reference recursion
      (store_train_grad.py:343-386) must match dve_recursion(lr_mode='none') exactly.

  Test 2 (projection approximates exact DVE): on a small linear model where the FULL
      parameter dimension is small enough that exact (un-projected) DVE is computable,
      the projected DVE values track the exact values, with correlation -> 1 as the
      projection rank grows (the JL guarantee). Note: exact full-dim DVE is infeasible
      on a real GPT (the vocab*d_model embedding makes M huge) — which is exactly why
      the method projects; the toy is the honest exact comparison.

End-to-end behavior on GPT2-Tiny is exercised by running main.py (see README); this file
covers the math.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from ghostEngines.gradProjection.dve_embedding import dve_recursion


def reference_recursion(grads_by_step):
    """Literal port of the reference reverse recursion (lr dropped), training order in/out.

    Mirrors store_train_grad.py: process reversed; first sets e=g and M=sum_b e_b g_b^T;
    subsequent steps e = g - (g @ M^T) reshaped, then M += sum_b e_b g_b^T.
    """
    n = len(grads_by_step)
    M = None
    embs = [None] * n
    for i, idx in enumerate(reversed(range(n))):
        g = grads_by_step[idx].float()
        B = g.shape[0]
        if i == 0:
            e = g.clone()
        else:
            M_times_grad = torch.mm(g, M.t())  # [B, D]
            e = g - M_times_grad
        outer = torch.einsum('bi,bj->bij', e, g).sum(dim=0)  # [D, D]
        M = outer if M is None else M + outer
        embs[idx] = e
    return embs


def test_recursion_fidelity(seed=0):
    print("\n[Test 1] Recursion fidelity vs literal reference port...")
    g = torch.Generator().manual_seed(seed)
    n_steps, B, D = 7, 4, 11
    # Small-magnitude grads keep the un-normalized recursion well-conditioned (it has no
    # lr/normalization damping, so large inputs blow up and fp32 rounding dominates).
    grads = [torch.randn(B, D, generator=g) * 0.1 for _ in range(n_steps)]

    ref = reference_recursion(grads)
    ours = dve_recursion(grads, lrs=None, lr_mode='none')

    rel_err = max((a - b).abs().max().item() / (a.abs().max().item() + 1e-12)
                  for a, b in zip(ref, ours))
    print(f"  max relative diff over {n_steps} steps: {rel_err:.3e}")
    assert rel_err < 1e-5, f"recursion mismatch: {rel_err}"
    print("  PASS")


def _spearman(a, b):
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean()
    br = br - br.mean()
    return float((ar @ br) / (ar.norm() * br.norm() + 1e-12))


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _run_linear_dve(grads, test_grads, lr_mode, lrs):
    """Full-dim DVE values for a single block: [n_test, n_steps*B]."""
    embs = dve_recursion(grads, lrs=lrs, lr_mode=lr_mode)  # each [B, D]
    E = torch.cat(embs, dim=0)              # [n_train, D]
    return test_grads @ E.t()              # [n_test, n_train]


def test_projection_approximates_exact(seed=0, lr_mode='scaled'):
    print(f"\n[Test 2] Projection approximates exact DVE (lr_mode={lr_mode})...")
    gen = torch.Generator().manual_seed(seed)
    # Full parameter dim small enough that exact (un-projected) DVE is feasible. Use
    # B > D and a modest lr so the SGD trajectory is contractive (I - lr*H stable) and
    # the recursion stays bounded — the regime real training lives in.
    D = 40
    n_steps, B = 10, 128
    n_test = 20
    base_lr = 0.1

    # Build a real SGD trajectory on a linear least-squares model so the captured
    # per-sample gradients are genuine trajectory gradients.
    w = torch.randn(D, generator=gen) * 0.1
    lrs = [base_lr] * n_steps
    grads = []
    for s in range(n_steps):
        X = torch.randn(B, D, generator=gen)
        t = torch.randn(B, generator=gen)
        resid = X @ w - t                       # [B]
        g = resid.unsqueeze(1) * X              # per-sample grad [B, D]
        grads.append(g)
        w = w - lrs[s] * g.mean(dim=0)          # SGD step

    # Test gradients at the final w.
    Xte = torch.randn(n_test, D, generator=gen)
    tte = torch.randn(n_test, generator=gen)
    g_test = (Xte @ w - tte).unsqueeze(1) * Xte  # [n_test, D]

    V_exact = _run_linear_dve(grads, g_test, lr_mode, lrs)  # [n_test, n_train]
    assert torch.isfinite(V_exact).all(), "exact DVE diverged; trajectory not stable"
    v_exact_flat = V_exact.reshape(-1)

    print(f"  {'rank':>6} {'pearson':>9} {'spearman':>9}")
    last_pearson = 0.0
    for k in [4, 16, 40, 200]:
        P = torch.randn(k, D, generator=gen) / (k ** 0.5)  # E[P^T P] ~= I
        proj_grads = [g @ P.t() for g in grads]            # [B, k]
        proj_test = g_test @ P.t()                         # [n_test, k]
        V_proj = _run_linear_dve(proj_grads, proj_test, lr_mode, lrs)
        pe = _pearson(v_exact_flat, V_proj.reshape(-1))
        sp = _spearman(v_exact_flat, V_proj.reshape(-1))
        print(f"  {k:>6} {pe:>9.4f} {sp:>9.4f}")
        last_pearson = pe

    assert last_pearson > 0.9, f"high-rank projection should track exact (got {last_pearson})"
    print("  PASS (correlation -> 1 as rank grows)")


def main():
    test_recursion_fidelity()  # covers lr_mode='none' exactly vs the reference port
    test_projection_approximates_exact(lr_mode='scaled')  # realistic, stable regime
    print("\nAll DVE correctness tests passed.")


if __name__ == '__main__':
    main()
