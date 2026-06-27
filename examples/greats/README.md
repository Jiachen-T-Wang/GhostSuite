# GREATS — Online Batch Selection with Ghost Inner Products

[GREATS](https://github.com/Jiachen-T-Wang/GREATS) (Wang et al., NeurIPS 2024,
"Online Selection of High-Quality Data for LLM Training in Every Iteration") selects, at
**every** training step, the subset of a candidate batch whose gradients best reduce the
validation loss — using the cheap per-sample gradient dot products that the GhostSuite
`GradDotProd` engine already computes in a single backward pass.

GREATS scores are recomputed **online, every step, against the current model**, and only
the selected subset is used for the update — it is *not* a static GradNorm/TracIN baseline.
Candidates are ranked by their train↔validation gradient alignment `s_i = <g_i, g_val>`
(first-order), optionally with the redundancy-aware greedy second-order term (the
candidate–candidate Gram), depending on the example and `--selection` mode below.

## Layout
- `pretrain/` — Online **first-order** selection during GPT-2 pretraining, reusing the
  `examples/lm/` shared model + data utilities (synthetic / Pile). See `pretrain/README.md`.
- `sft/` — Online selection during LoRA instruction tuning, toward the paper's
  torchtune/LESS + MMLU setting. Supports both `--selection first_order` and the
  **second-order** Gram-based greedy variant (default, "true GREATS"). See `sft/README.md`.
