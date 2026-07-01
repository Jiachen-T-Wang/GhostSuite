"""Rank-convergence analysis for the DVE value matrices.

Loads the per-rank ``value/values.pt`` produced by running the DVE pipeline at
several ``--proj_rank_total`` settings on the SAME trajectory (same seed/data/optimizer),
and checks that the value matrices converge as rank grows -- the JL guarantee on the
real model. Because every run shares the seed, the n_train columns (training windows,
in training order) and n_test rows line up across ranks, so the matrices are directly
comparable element-for-element.

Reports, for each consecutive rank pair (k, 2k):
  - Pearson / Spearman over the flattened [n_test, n_train] value matrix,
  - mean Jaccard overlap of each test row's top-k most-valuable training points.

Convergence toward 1.0 as rank grows = pass.

Usage:
  python analyze_rank_convergence.py --paths RANK64_VALUES RANK128_VALUES ... \
      --ranks 64 128 256 512 [--topk 10]
"""

import argparse

import torch


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _spearman(a, b):
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean()
    br = br - br.mean()
    return float((ar @ br) / (ar.norm() * br.norm() + 1e-12))


def _topk_jaccard(A, B, k):
    """Mean over test rows of |top-k(A_row) ∩ top-k(B_row)| / |union|."""
    n_test, n_train = A.shape
    k = min(k, n_train)
    ja = 0.0
    for i in range(n_test):
        sa = set(torch.topk(A[i], k).indices.tolist())
        sb = set(torch.topk(B[i], k).indices.tolist())
        ja += len(sa & sb) / len(sa | sb)
    return ja / n_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', nargs='+', required=True,
                    help='values.pt paths, in the same order as --ranks')
    ap.add_argument('--ranks', nargs='+', type=int, required=True)
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()
    assert len(args.paths) == len(args.ranks), "paths and ranks must align"

    mats, shapes = [], []
    for r, p in zip(args.ranks, args.paths):
        d = torch.load(p, map_location='cpu')
        V = d['values'].float()
        mats.append(V)
        shapes.append(tuple(V.shape))
        finite = torch.isfinite(V).all().item()
        print(f"rank {r:>5}: values {tuple(V.shape)}  finite={finite}  "
              f"|V|_mean={V.abs().mean():.3e}  std={V.std():.3e}")

    base = shapes[0]
    for r, s in zip(args.ranks, shapes):
        if s != base:
            raise SystemExit(f"shape mismatch at rank {r}: {s} vs {base} "
                             "(runs must share seed/steps/batch/n_test)")

    print(f"\nConsecutive-rank convergence (top-{args.topk} Jaccard over {base[0]} test rows):")
    print(f"  {'pair':>14} {'pearson':>9} {'spearman':>9} {'topk_jacc':>10}")
    for i in range(len(args.ranks) - 1):
        A, B = mats[i], mats[i + 1]
        pe = _pearson(A.reshape(-1), B.reshape(-1))
        sp = _spearman(A.reshape(-1), B.reshape(-1))
        jac = _topk_jaccard(A, B, args.topk)
        print(f"  {args.ranks[i]:>5}->{args.ranks[i+1]:<6} {pe:>9.4f} {sp:>9.4f} {jac:>10.4f}")

    # Also correlate each rank against the highest rank (the best available proxy
    # for the exact limit) to show monotone approach.
    ref = mats[-1]
    print(f"\nVs highest rank ({args.ranks[-1]}, proxy for the exact limit):")
    print(f"  {'rank':>6} {'pearson':>9} {'spearman':>9} {'topk_jacc':>10}")
    for i in range(len(args.ranks)):
        pe = _pearson(mats[i].reshape(-1), ref.reshape(-1))
        sp = _spearman(mats[i].reshape(-1), ref.reshape(-1))
        jac = _topk_jaccard(mats[i], ref, args.topk)
        print(f"  {args.ranks[i]:>6} {pe:>9.4f} {sp:>9.4f} {jac:>10.4f}")


if __name__ == '__main__':
    main()
