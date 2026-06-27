"""Plot GREATS pretraining val/test loss vs steps (trialrun.png-style, two panels).

Parses the raw run logs in ./logs/ and writes ./greats_pretrain_2026-06-25.png.
Run from the repo root with this worktree's venv:
    .venv/bin/python examples/greats/pretrain/experiments/plot_greats_pretrain_loss.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))  # examples/greats/pretrain/experiments
LOG_DIR = os.path.join(_HERE, "logs")
OUT = os.path.join(_HERE, "greats_pretrain_2026-06-25.png")

ARMS = [
    ("Regular", "regular_10275913.log", "tab:blue"),
    ("GREATS (excl wte/lm_head)", "greats_excl_wte_lmhead_10275914.log", "tab:orange"),
    ("GREATS (incl tied score)", "greats_incl_10275915.log", "tab:green"),
]
_LINE = re.compile(r"^step (\d+): train loss [\d.]+, val loss ([\d.]+), test loss ([\d.]+)")


def parse(path):
    steps, val, test = [], [], []
    with open(path) as f:
        for line in f:
            m = _LINE.match(line)
            if m:
                steps.append(int(m.group(1)))
                val.append(float(m.group(2)))
                test.append(float(m.group(3)))
    return steps, val, test


fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(11, 4.2))
for label, fname, color in ARMS:
    steps, val, test = parse(os.path.join(LOG_DIR, fname))
    ax_val.plot(steps, val, label=label, color=color, lw=1.6)
    ax_test.plot(steps, test, label=label, color=color, lw=1.6)

for ax, title in ((ax_val, "val loss, Pile (GPT2-Small)"),
                  (ax_test, "test loss, Pile (GPT2-Small)")):
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss (nats)")
    ax.legend()
    ax.grid(alpha=0.3)

# Zoomed inset focus on the steady-state tail where arms separate.
for ax, idx in ((ax_val, 1), (ax_test, 2)):
    ax.set_ylim(2.85, 3.6)
    ax.set_xlim(2000, 20000)

fig.suptitle("GREATS pretraining — online batch selection on Pile (equal update size k=16)", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
