"""Plot OPUS-vs-GREATS-vs-Regular pretraining val/test loss vs steps (two panels).

All six curves are SAME-CODE runs: the Regular / GREATS-excl baselines were re-run alongside
the OPUS arms because the 2026-06-25 GREATS experiment logs predate an LR-schedule fix (the
old config hardcoded a 10k cosine horizon, flatlining at min-lr for the second half of a 20k
run) and are not comparable with newer runs — see this folder's README.

Parses the raw run logs in ./logs/ and writes ./opus_pretrain_2026-07-04.png. Run from the
repo root:
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
OUT = os.path.join(_HERE, "opus_pretrain_2026-07-04.png")


def _first(pattern):
    hits = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    return hits[0] if hits else None

ARMS = [
    ("Regular",                          _first("regular_rerun_*.log"),     "tab:blue",   "-"),
    ("GREATS TopK (excl wte/lm_head)",   _first("greats_excl_rerun_*.log"), "tab:orange", "-"),
    ("OPUS topk (adamw_scalar)",         _first("opus_topk_1*.log"),        "tab:red",    "-"),
    ("OPUS topk (raw scores)",           _first("opus_topk-raw_*.log"),     "tab:brown",  "--"),
    ("OPUS stochastic (T=1e-9)",         _first("opus_stochastic_*.log"),   "tab:purple", "-"),
    ("OPUS greedy (raw units)",          _first("opus_greedy_*.log"),       "tab:gray",   "--"),
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
for label, fname, color, ls in ARMS:
    if fname is None or not os.path.exists(fname):
        print(f"skipping {label}: log not found")
        continue
    steps, val, test = parse(fname)
    ax_val.plot(steps, val, label=label, color=color, lw=1.6, ls=ls)
    ax_test.plot(steps, test, label=label, color=color, lw=1.6, ls=ls)
    if steps:
        print(f"{label}: final step {steps[-1]}  val {val[-1]:.3f}  test {test[-1]:.3f}")

for ax, title in ((ax_val, "val loss, Pile (GPT2-Small)"),
                  (ax_test, "test loss, Pile (GPT2-Small)")):
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss (nats)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(2.75, 3.6)
    ax.set_xlim(2000, 20000)

fig.suptitle("OPUS vs GREATS pretraining — online selection on Pile "
             "(equal update size k=16, same-code arms)", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
