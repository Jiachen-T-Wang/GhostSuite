# OPUS Pretraining — Sketched, Diversity-Aware Online Data Selection

A re-implementation of **OPUS** (Optimizer-induced Projected Utility Selection,
[github.com/gszfwsb/OPUS](https://github.com/gszfwsb/OPUS), ICML 2026 — the follow-up line to
GREATS) on the GhostSuite `GradProjLora` engine, sharing the model/data stack and training
protocol of [`examples/greats/pretrain/`](../greats/pretrain/README.md) so the two selection
rules are directly comparable.

## Algorithm (per training step)
Let `N` = candidate pool (`--candidate_batch_size`), `k` = trained subset (`--batch_size`),
`m` = proxy batch (`--val_batch_size`).

1. **Draw** `N` candidates and a fresh proxy batch of `m` (from the fixed eval window pool by
   default — the GREATS protocol).
2. **Sketching pass** — ONE forward/backward over `[candidates ++ proxy]` with the projection
   engine's hooks: each scored layer's per-sample gradient is captured directly in projected
   space, `P_o (Σ_t b_t a_tᵀ) P_iᵀ`, **never materializing a per-sample gradient**. Per-layer
   sketches concatenate to `[N+m, K]`.
3. **Score + Gram** — two small GEMMs: `s_i = <S_i, mean_j S_vj>` and `sim_ij = <S_i, S_j>`.
   With `--opus_preconditioner adamw_scalar` (default) the train-side sketches carry OPUS's
   per-layer AdamW factors `C_t/sqrt(numel)` (scores get `c_l`, the Gram `c_l²`).
4. **Select** `k` of `N` — Boltzmann **stochastic-greedy** with the Gram redundancy penalty
   (the OPUS default; `--opus_selection_method greedy|topk` for the deterministic /
   GREATS-style ablations).
5. **Update** — a plain forward/backward + optimizer step on the selected `k` (identical to the
   GREATS example's subset update; scoring hooks are detached here).

## vs the OPUS reference implementation
Compared with the reference implementation at
[github.com/gszfwsb/OPUS](https://github.com/gszfwsb/OPUS):

- The reference's default mode CountSketches per-sample gradients by **materializing them in
  row chunks**; our two-sided factorized projection computes the sketch inside the backward
  with no materialization. Both are unbiased inner-product estimators, but at GPT2-Small
  (`N=32, m=16`, bf16, H200) our scoring round takes **88 ms (seq 512) / 172 ms (seq 1024)**
  vs **537 / 862 ms** for the reference default (CountSketch + AdamW diagonal), at ~11 GiB
  less peak memory — 5–6× faster, with score fidelity vs exact dots of Spearman 0.97 / 0.995.
- The reference's *element-wise* AdamW diagonal `1/(sqrt(v_t)+eps)` is **not supported** — it
  cannot pass through a factorized projection without materializing the gradient, which is
  the very cost this port avoids. The per-layer AdamW scalars (`adamw_scalar`) are supported
  exactly. (Folding a rank-1-factored diagonal into `P_o/P_i` is possible future work.)
- Scored layers default to the transformer-block Linears (`--proj_layers c_attn,c_fc,c_proj`),
  i.e. the GREATS experiment's recommended `excl wte,lm_head` configuration; the reference
  likewise scores Linear layers only (no embedding).
- The reference trainer wraps its model in `torch.compile`, but gradients w.r.t. activations
  captured as module attributes *inside* a compiled graph are unavailable
  (`torch.autograd.grad` returns `None` for side outputs of a compiled region — the scores
  silently zero out), so capture-based scoring must run eager. This port keeps the scoring
  pass hook-based and eager by design, and the update step stays a plain step.

## Quick start

```bash
# Smoke test (synthetic data, 1 GPU, no corpus)
python examples/opus/main.py --method OPUS --train_set synthetic \
    --architecture GPT2-Tiny --candidate_batch_size 16 --batch_size 8 \
    --val_batch_size 4 --max_steps 12 --eval_interval 4 \
    --model_dtype float32 --train_dtype float32

# Real training (tokenized Pile; see examples/lm/graddotprod_lm/README.md for tokenization)
cd examples/opus
./train.sh --architecture GPT2-Small --candidate_batch_size 32 --batch_size 16
./train.sh --method Regular --batch_size 16     # no-selection baseline
```

## Correctness
The scorer is validated against a naive per-sample-gradient oracle and against the reference
implementation's scoring kernels on the same model and batch: with full per-layer dimensions
and calibrated row-orthonormal projections the sketch inner products are *exact*, and scores
and Gram match both references to ~1e-6 relative error (fp32); greedy selection returns
indices identical to the reference. At the default compressed budget (`--proj_dim 8192`),
sketched scores rank-correlate with exact scores at Spearman 0.995 (GPT2-Small, seq 1024).

## Key flags
- `--method {OPUS,Regular}` — online selection vs no-selection baseline.
- `--candidate_batch_size N` / `--batch_size k` / `--val_batch_size m`.
- `--opus_selection_method {stochastic,greedy,topk}`, `--opus_temperature T`.
- `--opus_preconditioner {adamw_scalar,none}`.
- `--proj_dim` — per-layer sketch budget `k_i·k_o` (default 8192, matching the reference's
  CountSketch dimension).
- `--score_seq_len L` — score on an `L`-token prefix (the reference's `score_len` efficiency
  knob); default scores the full sequence like GREATS.

## Experiment & results
The OPUS-vs-GREATS-vs-Regular comparison on Pile (same protocol as the committed GREATS
experiment) lives in [`experiments/`](experiments/README.md).
