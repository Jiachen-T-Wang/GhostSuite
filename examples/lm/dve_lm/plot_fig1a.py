"""Reproduce Figure 1(a) of Data Value Embedding (arXiv:2412.09538).

Figure 1(a): average DVE influence per training batch vs training iteration, measured
against the (final) model's loss on the validation set. The DVE value matrix
``values.pt`` (from stage 3, ``compute_values``) holds ``values[n_test, n_train]`` =
``<g_val_j, e_s>`` for every (val example j, training sample s), with ``train_order``
giving each column's training step. We average over the validation rows and over the
samples within each training step to get one scalar per step, and plot it vs the step.

With ``--lr_mode scaled`` (the recommended long-run setting) the stored embedding already
folds the per-step learning rate (``e_hat = lr_s * e_s``), so the raw per-step average is
the *lr-weighted* influence. Dividing by the per-step lr recovers the *intrinsic*
(schedule-independent) influence. The paper normalizes by the per-batch lr, so we emit
both curves (raw + lr-normalized) and a CSV; the lr-normalized one is the closest match.

Usage:
  python plot_fig1a.py --values RUN/value/values.pt --out RUN/fig1a \
      --lr_mode scaled --learning_rate 3e-4 --warmup_steps 2000 --max_steps 60000 \
      --lr_schedule linear [--smooth 50]
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lr_at(step, lr, warmup, max_steps, schedule):
    """Mirror dve_train_loop.compute_lr for the schedules used here."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
    if schedule == 'constant':
        return lr
    if schedule == 'linear':
        return lr * (1.0 - progress)
    # cosine (decays to ~0; min_lr not threaded here)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--values', required=True, help='path to stage-3 values.pt')
    ap.add_argument('--out', required=True, help='output path prefix (writes .png and .csv)')
    ap.add_argument('--lr_mode', default='scaled', choices=['none', 'scaled'])
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--warmup_steps', type=int, default=2000)
    ap.add_argument('--max_steps', type=int, default=60000)
    ap.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'cosine'])
    ap.add_argument('--smooth', type=int, default=0, help='moving-average window for display')
    args = ap.parse_args()

    d = torch.load(args.values, map_location='cpu')
    values = d['values'].float()                      # [n_test, n_train]
    order = np.asarray(d['train_order'], dtype=np.int64)  # [n_train] step per column
    n_test, n_train = values.shape
    print(f"values {tuple(values.shape)}  finite={torch.isfinite(values).all().item()}  "
          f"steps {order.min()}..{order.max()}")

    # Mean over validation rows -> per-training-sample influence, then group by step.
    col_mean = values.mean(dim=0).numpy()             # [n_train]
    n_steps = int(order.max()) + 1
    sums = np.bincount(order, weights=col_mean, minlength=n_steps)
    counts = np.bincount(order, minlength=n_steps)
    valid = counts > 0
    steps = np.nonzero(valid)[0]
    raw = sums[valid] / counts[valid]                 # lr-weighted influence per step

    # lr-normalized: divide out the per-step lr that scaled mode folded in.
    lrs = np.array([lr_at(int(s), args.learning_rate, args.warmup_steps,
                          args.max_steps, args.lr_schedule) for s in steps])
    if args.lr_mode == 'scaled':
        norm = raw / np.clip(lrs, 1e-12, None)
    else:
        norm = raw  # 'none' mode: lr was never folded in; raw already lr-independent

    # CSV
    csv_path = args.out + '.csv'
    np.savetxt(csv_path, np.column_stack([steps, lrs, raw, norm]),
               delimiter=',', header='step,lr,influence_raw,influence_lr_normalized',
               comments='', fmt=['%d', '%.8e', '%.8e', '%.8e'])
    print(f"wrote {csv_path}")

    # Plot
    r_disp = moving_average(raw, args.smooth)
    n_disp = moving_average(norm, args.smooth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, r_disp, lw=0.8, color='C0')
    axes[0].set_title('Avg DVE influence per batch (lr-weighted)')
    axes[0].set_xlabel('training iteration'); axes[0].set_ylabel('influence')
    axes[0].axhline(0, color='k', lw=0.5, alpha=0.4)
    axes[1].plot(steps, n_disp, lw=0.8, color='C1')
    axes[1].set_title('Avg DVE influence per batch (lr-normalized)')
    axes[1].set_xlabel('training iteration'); axes[1].set_ylabel('influence / lr')
    axes[1].axhline(0, color='k', lw=0.5, alpha=0.4)
    sm = f" (MA{args.smooth})" if args.smooth > 1 else ""
    fig.suptitle(f'DVE Figure 1(a) reproduction — GPT2-Small / Pile{sm}')
    fig.tight_layout()
    png_path = args.out + '.png'
    fig.savefig(png_path, dpi=140)
    print(f"wrote {png_path}")


if __name__ == '__main__':
    main()
