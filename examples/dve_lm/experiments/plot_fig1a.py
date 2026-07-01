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
(schedule-independent) influence. The paper normalizes by the per-batch lr, so the
lr-normalized curve is the faithful match; we emit both plus CSVs.

Outputs (``--out PREFIX``):
  PREFIX.png            clean paper-style panel: binned lr-normalized influence +
                        inter-quartile band, symlog-y so the warmup spike and the
                        post-warmup basin/ascent are both legible.
  PREFIX_diagnostic.png two raw panels (lr-weighted + lr-normalized), lightly smoothed.
  PREFIX.csv            per-step step,lr,influence_raw,influence_lr_normalized.
  PREFIX_binned.csv     per-bin center,lr,mean,q25,q75 of the lr-normalized influence.

Read the value matrix with ``--values RUN/value/values.pt`` (recomputes the per-step
series), or re-plot quickly from an existing per-step CSV with ``--from_csv PREFIX.csv``.

Usage:
  python plot_fig1a.py --values RUN/value/values.pt --out RUN/fig1a \
      --lr_mode scaled --learning_rate 3e-4 --warmup_steps 2000 --max_steps 60000 \
      --lr_schedule linear --bins 200
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lr_at(step, lr, warmup, decay_steps, schedule):
    """Mirror dve_train_loop.compute_lr for the schedules used here.

    ``decay_steps`` is the LR-decay horizon (the training ``--lr_decay_steps``, or ``max_steps``
    when that is unset); it must match the run's schedule so the lr-normalization divides by the
    same per-step lr that was folded into the embedding.
    """
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, decay_steps - warmup)))
    if schedule == 'constant':
        return lr
    if schedule == 'linear':
        return lr * (1.0 - progress)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine (to ~0)


def moving_average(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode='same')


def per_step_series(args):
    """Return (steps, lrs, raw, norm) one value per training step."""
    if args.from_csv:
        d = np.genfromtxt(args.from_csv, delimiter=',', names=True)
        steps = d['step'].astype(np.int64)
        return steps, d['lr'], d['influence_raw'], d['influence_lr_normalized']

    d = torch.load(args.values, map_location='cpu')
    values = d['values'].float()                          # [n_test, n_train]
    order = np.asarray(d['train_order'], dtype=np.int64)  # step per column
    print(f"values {tuple(values.shape)}  finite={torch.isfinite(values).all().item()}  "
          f"steps {order.min()}..{order.max()}")

    col_mean = values.mean(dim=0).numpy()                 # mean over val rows
    n_steps = int(order.max()) + 1
    sums = np.bincount(order, weights=col_mean, minlength=n_steps)
    counts = np.bincount(order, minlength=n_steps)
    valid = counts > 0
    steps = np.nonzero(valid)[0]
    raw = sums[valid] / counts[valid]                     # lr-weighted influence/step
    decay_steps = args.lr_decay_steps if args.lr_decay_steps > 0 else args.max_steps
    lrs = np.array([lr_at(int(s), args.learning_rate, args.warmup_steps,
                          decay_steps, args.lr_schedule) for s in steps])
    norm = raw / np.clip(lrs, 1e-12, None) if args.lr_mode == 'scaled' else raw

    np.savetxt(args.out + '.csv', np.column_stack([steps, lrs, raw, norm]),
               delimiter=',', header='step,lr,influence_raw,influence_lr_normalized',
               comments='', fmt=['%d', '%.8e', '%.8e', '%.8e'])
    print(f"wrote {args.out}.csv")
    return steps, lrs, raw, norm


def bin_curve(steps, vals, nbins):
    """Bin steps into nbins equal-width groups; return center/mean/sem per bin.

    The signed per-batch influence is small relative to its step-to-step spread, so the
    *mean* (the temporal-influence signal in Fig 1a) is reported with a standard-error
    band, not an inter-quartile band (the IQR is dominated by noise and would bury the
    signal). SEM = std(step values in bin) / sqrt(count).
    """
    edges = np.linspace(steps.min(), steps.max() + 1, nbins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, nbins - 1)
    c, m, sem = [], [], []
    for b in range(nbins):
        sel = idx == b
        if not sel.any():
            continue
        v = vals[sel]
        c.append(steps[sel].mean()); m.append(v.mean())
        sem.append(v.std() / np.sqrt(len(v)))
    return (np.array(c), np.array(m), np.array(sem))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--values', help='path to stage-3 values.pt')
    src.add_argument('--from_csv', help='re-plot from a prior PREFIX.csv (fast)')
    ap.add_argument('--out', required=True, help='output path prefix')
    ap.add_argument('--lr_mode', default='scaled', choices=['none', 'scaled'])
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--warmup_steps', type=int, default=2000)
    ap.add_argument('--max_steps', type=int, default=60000)
    ap.add_argument('--lr_decay_steps', type=int, default=-1,
                    help='LR-decay horizon; -1 uses max_steps. Must match the training run.')
    ap.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'cosine'])
    ap.add_argument('--bins', type=int, default=60, help='bins for the clean panel')
    ap.add_argument('--smooth', type=int, default=200, help='MA window for the diagnostic panel')
    ap.add_argument('--logy', action='store_true',
                    help='symlog y on the clean panel (default linear, which reads cleaner)')
    args = ap.parse_args()

    steps, lrs, raw, norm = per_step_series(args)
    print(f"per-step series: {len(steps)} steps, finite={np.isfinite(norm).all()}")

    # ---- Clean paper-style panel: binned mean lr-normalized influence + SEM band ----
    c, m, sem = bin_curve(steps, norm, args.bins)
    np.savetxt(args.out + '_binned.csv', np.column_stack([c, m, sem]),
               delimiter=',', header='step_center,norm_mean,norm_sem',
               comments='', fmt='%.8e')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(c, m - sem, m + sem, color='C1', alpha=0.3, lw=0, label='±1 SEM')
    ax.plot(c, m, color='C1', lw=1.8, marker='o', ms=2.5, label=f'binned mean ({args.bins} bins)')
    ax.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax.axvline(args.warmup_steps, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.text(args.warmup_steps, ax.get_ylim()[1], ' end of warmup', color='gray',
            va='top', ha='left', fontsize=8)
    if args.logy:
        # symlog keyed to the MEAN signal (not the noise spread).
        ax.set_yscale('symlog', linthresh=max(1e-9, float(np.median(np.abs(m)))))
    ax.set_xlabel('training iteration')
    ax.set_ylabel('avg influence per batch  (/ lr)')
    steps_lbl = f'{args.max_steps // 1000}k' if args.max_steps >= 1000 else str(args.max_steps)
    ax.set_title(f'DVE Figure 1(a) reproduction — GPT2-Small / Pile (1% subset, {steps_lbl} steps)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=150)
    print(f"wrote {args.out}.png")

    # ---- Diagnostic: raw (lr-weighted) vs lr-normalized, lightly smoothed ----
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, moving_average(raw, args.smooth), lw=0.7, color='C0')
    axes[0].set_title('lr-weighted (raw)'); axes[0].axhline(0, color='k', lw=0.5, alpha=0.4)
    axes[1].plot(steps, moving_average(norm, args.smooth), lw=0.7, color='C1')
    axes[1].set_title('lr-normalized'); axes[1].axhline(0, color='k', lw=0.5, alpha=0.4)
    for a in axes:
        a.set_xlabel('training iteration'); a.set_ylabel('influence')
    fig2.suptitle(f'DVE Fig 1(a) diagnostic — GPT2-Small / Pile (MA{args.smooth})')
    fig2.tight_layout()
    fig2.savefig(args.out + '_diagnostic.png', dpi=140)
    print(f"wrote {args.out}_diagnostic.png")


if __name__ == '__main__':
    main()
