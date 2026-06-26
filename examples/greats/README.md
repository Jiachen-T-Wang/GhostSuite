# GREATS — Online Batch Selection with Ghost Inner Products

[GREATS](https://github.com/Jiachen-T-Wang/GREATS) (Wang et al., NeurIPS 2024,
"Online Selection of High-Quality Data for LLM Training in Every Iteration") selects, at
**every** training step, the subset of a candidate batch whose gradients best reduce the
validation loss — using the cheap per-sample gradient dot products that the GhostSuite
`GradDotProd` engine already computes in a single backward pass.

This example implements the **first-order** variant of GREATS: candidates are ranked by
their train↔validation gradient alignment `s_i = <g_i, g_val>` (no pairwise train–train
Gram / greedy second-order term yet). It is *not* a static GradNorm/TracIN baseline:
scores are recomputed **online, every step, against the current model**, and only the
selected subset is used for the update.

## Layout
- `pretrain/` — **Phase A (implemented).** Online selection during GPT-2 pretraining,
  reusing the `examples/lm/` shared model + data utilities (synthetic / Pile).
- `sft/` — **Phase B (planned).** Online selection during LoRA instruction tuning,
  toward the paper's torchtune/LESS + MMLU setting. See `sft/README.md`.

Design notes and the full plan: `docs/plans/greats_example_implementation_2026-06-25.md`.
