"""Plot OPUS-vs-GREATS-vs-Regular pretraining val/test loss vs steps (two panels).

Parses the raw run logs in ./logs/ (OPUS arms) plus the committed GREATS experiment logs
(examples/greats/pretrain/experiments/logs/ — same protocol, same seed, same eval windows)
and writes ./opus_pretrain_2026-07-03.png. Run from the repo root with this worktree's venv:
    .venv/bin/python examples/opus/experiments/plot_opus_pretrain_loss.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))           # examples/opus/experiments
LOG_DIR = os.path.join(_HERE, "logs")
GREATS_LOGS = os.path.abspath(os.path.join(
    _HERE, "..", "..", "greats", "pretrain", "experiments", "logs"))
OUT = os.path.join(_HERE, "opus_pretrain_2026-07-03.png")

def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

ARMS = [
    ("Regular", os.path.join(GREATS_LOGS, "regular_10275913.log"), "tab:blue"),
    ("GREATS (excl wte/lm_head)",
     os.path.join(GREATS_LOGS, "greats_excl_wte_lmhead_10275914.log"), "tab:orange"),
    ("OPUS stochastic (adamw_scalar, T=1e-9)",
     _first(os.path.join(LOG_DIR, "opus_stochastic_*.log")), "tab:red"),
    ("OPUS greedy (raw units)",
     _first(os.path.join(LOG_DIR, "opus_greedy_*.log")), "tab:purple"),
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
    if fname is None or not os.path.exists(fname):
        print(f"skipping {label}: log not found")
        continue
    steps, val, test = parse(fname)
    ax_val.plot(steps, val, label=label, color=color, lw=1.6)
    ax_test.plot(steps, test, label=label, color=color, lw=1.6)
    if steps:
        print(f"{label}: final step {steps[-1]}  val {val[-1]:.3f}  test {test[-1]:.3f}")

for ax, title in ((ax_val, "val loss, Pile (GPT2-Small)"),
                  (ax_test, "test loss, Pile (GPT2-Small)")):
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss (nats)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(2.85, 3.6)
    ax.set_xlim(2000, 20000)

fig.suptitle("OPUS vs GREATS pretraining — online selection on Pile (equal update size k=16)",
             y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
