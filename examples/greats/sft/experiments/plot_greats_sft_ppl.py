"""Plot GREATS SFT eval/test perplexity vs steps (mirrors upstream trialrun.png).

Parses the raw run logs in ./logs/ and writes ./greats_sft_ppl_2026-06-25.png.
Run from the repo root with this worktree's venv:
    .venv/bin/python examples/greats/sft/experiments/plot_greats_sft_ppl.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))  # examples/greats/sft/experiments
LOG_DIR = os.path.join(_HERE, "logs")
OUT = os.path.join(_HERE, "greats_sft_ppl_2026-06-25.png")

# (label, log filename, color)
ARMS = [
    ("Regular", "regular_10272824.log", "tab:blue"),
    ("GREATS", "greats_first_order_10272825.log", "tab:orange"),
]
_EVAL = re.compile(
    r"\[eval\] step (\d+) \|.*?eval_ppl ([\d.]+) \| test_ppl ([\d.]+)")


def parse(path):
    steps, eval_ppl, test_ppl = [], [], []
    with open(path) as f:
        for line in f:
            m = _EVAL.search(line)
            if m:
                steps.append(int(m.group(1)))
                eval_ppl.append(float(m.group(2)))
                test_ppl.append(float(m.group(3)))
    return steps, eval_ppl, test_ppl


fig, (ax_eval, ax_test) = plt.subplots(1, 2, figsize=(11, 4.2))
for label, fname, color in ARMS:
    steps, ev, te = parse(os.path.join(LOG_DIR, fname))
    ax_eval.plot(steps, ev, label=label, color=color, marker="o", ms=3)
    ax_test.plot(steps, te, label=label, color=color, marker="o", ms=3)

for ax, title in ((ax_eval, "eval_perplexity, subject=sociology"),
                  (ax_test, "test_perplexity, subject=sociology")):
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Perplexity")
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("GREATS SFT — Llama-2-7b LoRA (MMLU sociology answer perplexity)", y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
