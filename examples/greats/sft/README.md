# GREATS SFT — Online Selection for LoRA Instruction Tuning (planned)

**Status: not yet implemented.** This directory is a placeholder for Phase B of the GREATS
example: online batch selection during **LoRA instruction tuning**, toward the paper's
torchtune / LESS + MMLU setting.

The same first-order online selection as `../pretrain/` applies, scoring candidates by
`<g_i, g_val>` over the **trainable LoRA parameters only** (base weights frozen) against a
held-out instruction/MMLU dev batch. The `GradDotProd` engine already supports `nn.Linear`,
and LoRA adapters are `nn.Linear`, so no new core ghost computation is required.

Two integration options to decide at Phase B planning time:
- **B1 (recommended):** a self-contained minimal LoRA SFT loop in-repo (HF model + a small
  LoRA), no torchtune dependency — keeps it smoke-testable like the rest of `examples/`.
- **B2 (high fidelity):** wrap the upstream torchtune/LESS recipe to reproduce paper
  numbers; heavier external deps.

Phase B will get its own implementation plan once Phase A (`../pretrain/`) is verified. See
`docs/plans/greats_example_implementation_2026-06-25.md`.
