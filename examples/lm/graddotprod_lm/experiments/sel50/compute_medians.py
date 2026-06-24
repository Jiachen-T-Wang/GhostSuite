#!/usr/bin/env python3
"""Compute global median thresholds for dot-product and cosine selection.

Reads every dot_prod_log_iter_*.pt under a scoring run's grad_dotprods/ dir,
gathers the per-sample dot-product and cosine scores across the whole run, and
writes their medians to <run_dir>/sel50_medians.env so the replay driver can
split the data into top-50% (metric >= median) and bottom-50% (metric < median).

cosine_i = dot_i / (train_grad_norm_i * val_grad_norm); requires the scoring run
to have been launched with --log_grad_norms.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch


def resolve_grad_dir(run_dir: str) -> str:
    direct = os.path.join(run_dir, "grad_dotprods")
    if os.path.isdir(direct):
        return direct
    for d in sorted(os.listdir(run_dir)):
        cand = os.path.join(run_dir, d, "grad_dotprods")
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"No grad_dotprods/ under {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    gd = resolve_grad_dir(args.run_dir)
    files = glob.glob(os.path.join(gd, "dot_prod_log_iter_*.pt"))
    files.sort(key=lambda f: int(re.search(r"iter_(-?\d+)", f).group(1)))
    if not files:
        raise FileNotFoundError(f"No dot_prod logs under {gd}")

    dots, coss = [], []
    have_norms = True
    for f in files:
        log = torch.load(f, map_location="cpu")
        for e in log:
            dp = e["dot_product"].float().numpy()
            dots.append(dp)
            if "train_grad_norm" in e and "val_grad_norm" in e:
                tn = e["train_grad_norm"].float().numpy()
                vn = float(e["val_grad_norm"])
                denom = np.clip(tn * vn, 1e-12, None)
                coss.append(dp / denom)
            else:
                have_norms = False

    dots = np.concatenate(dots)
    dot_med = float(np.median(dots))
    n = dots.size
    n_top_dot = int((dots >= dot_med).sum())

    lines = [
        f"DOT_MEDIAN={dot_med:.10e}",
        f"N_SAMPLES={n}",
        f"N_TOP_DOT={n_top_dot}",
    ]
    print(f"[medians] samples={n}  dot_median={dot_med:.6e}  "
          f"top(dot>=med)={n_top_dot} ({100*n_top_dot/n:.2f}%)")

    if have_norms and coss:
        coss = np.concatenate(coss)
        cos_med = float(np.median(coss))
        n_top_cos = int((coss >= cos_med).sum())
        lines.append(f"COS_MEDIAN={cos_med:.10e}")
        lines.append(f"N_TOP_COS={n_top_cos}")
        lines.append("HAVE_COSINE=1")
        print(f"[medians] cosine_median={cos_med:.6e}  "
              f"top(cos>=med)={n_top_cos} ({100*n_top_cos/n:.2f}%)")
    else:
        lines.append("HAVE_COSINE=0")
        print("[medians] WARNING: grad norms absent -> cosine unavailable "
              "(scoring run needs --log_grad_norms)", file=sys.stderr)

    out = os.path.join(args.run_dir, "sel50_medians.env")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[medians] wrote {out}")


if __name__ == "__main__":
    main()
