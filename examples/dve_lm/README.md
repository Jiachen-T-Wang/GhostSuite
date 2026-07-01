# Data Value Embedding (DVE)

Re-implementation of **Data Value Embedding** ([arXiv:2412.09538](https://arxiv.org/abs/2412.09538))
on top of GhostSuite's `gradProjection` engine. It attributes a trained model's behavior on
**test** examples back to each **training** example *and the training step at which it appeared*
(in-run / temporal influence), via a compact per-training-point embedding.

The expensive part — per-sample gradients — is captured cheaply by the ghost projection
`P = P_i ⊗ P_o` (no per-sample gradient is ever materialized). This replaces the reference's
random-LoRA Kronecker sketch with GhostSuite's Gaussian/orthonormal JL projection.

## Method

For a model trained with SGD `θ_{t+1} = θ_t − η_t (1/B) Σ_b g_{t,b}`, the first-order in-run
influence of training point `(s,b)` on a test point is `⟨g_test, e_{s,b}⟩`, where the **data
value embedding** is

```
e_s = g_s − M_{>s} g_s ,      M_{>s−1} = M_{>s} + (η_s / B_s) Σ_b e_{s,b} g_{s,b}ᵀ
```

i.e. `e_s` is the training gradient `g_s` after being transformed by the unrolled SGD Jacobian
`∏_{t>s}(I − η_t H_t)` of all *later* steps (`H_t ≈ (1/B) Σ_b g_t g_tᵀ`). `M` is maintained
**per layer** (block-diagonal across layers), and all gradients live in the projected space.

`--lr_mode`:
- `scaled` (default): folds the recorded per-step `η_s` and the `1/B_s` Gauss-Newton norm into
  the recursion — numerically consistent with the actual (small-lr, contractive) trajectory.
- `none`: reproduces the released reference exactly (`η = 1`, sum reduction). Only well-behaved
  for short runs / tiny gradients, since it drops the lr damping.

## Capture path & precision defaults

Stage-1 capture defaults to the **decoupled in-graph + `torch.compile`** fast path
(`--decoupled_compile`, on by default) at **`--train_dtype bfloat16`** (fp32 master weights, bf16
autocast; `--proj_dtype float32` for projection fidelity). On an H200 this is **~25% faster** than
the eager hook engine (GPT2-Small, bs 8, block 1024), and the resulting DVE values match the fp32
hook reference to **Pearson r = 1.0 / identical top-k rankings** (~1% L2 magnitude difference from
bf16 rounding).

- `--no_decoupled_compile` — fall back to the eager hook engine (`GradProjLoraEngine`). Required for
  **Conv1D**-based models (the decoupled path only supports `nn.Linear`/`nn.Embedding`). On a
  non-CUDA device the fast path auto-falls-back to the hook engine (compile is CUDA-only).
- `--ac_budget b` (in (0,1]) — compile-native activation checkpointing (Inductor min-cut): lower
  recomputes more in backward to cut peak memory (e.g. `0.5` ≈ −26% peak at ~+7% step time).

## Pipeline (4 stages)

Stages share config-derived paths, so they can run in one command or separately.

```bash
# All four stages, synthetic smoke (CPU-friendly, GPT2-Tiny; explicit fp32 for the CPU path):
python main.py --data_source synthetic --architecture GPT2-Tiny --device cpu \
    --optimizer sgd --learning_rate 0.05 --max_steps 8 --batch_size 4 \
    --n_test 8 --test_batch_size 4 --proj_rank_total 64 \
    --model_dtype float32 --train_dtype float32 --proj_dtype float32 \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

1. `--train_and_store_grad` — train the model (real optimizer steps) and capture per-step,
   per-sample projected gradients (`capture/proj_iter_*.pt`, one per step, tagged with `lr`,
   `order`, `batch_idx`) plus the final checkpoint (`capture/final_model.pt`).
2. `--compute_embedding` — reverse-recursion → `embedding/embed_iter_*.pt`.
3. `--compute_value` — load the final checkpoint, project the **test** gradients with the same
   `P`, dot against the embeddings → `value/values.pt` (`[n_test, n_train]`).
4. `--attribute` — print the top/bottom training points per test point.

### Real Pile run

The Pile loader (`shared/dataloader.py`) reads per-domain GPT-2 `.bin` files via env vars:

```bash
export PILE_DATA_DIR_TRAIN=/path/to/pile/pile-train
export PILE_DATA_DIR_VAL=/path/to/pile/pile-val-gpt2
export PILE_DATA_DIR_TEST=/path/to/pile/pile-test-gpt2

python main.py --data_source pile --architecture GPT2-Small --device cuda \
    --optimizer adamw --learning_rate 3e-4 --max_steps 500 --batch_size 8 \
    --n_test 64 --proj_rank_total 256 --lr_mode scaled \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

## Correctness

```bash
python validate_dve.py        # recursion + projection math (CPU, seconds)
python validate_dve_model.py  # direct-on-model checks (GPT2 + MLP)
```

`validate_dve.py`:
- **Test 1** — the recursion matches a literal port of the reference exactly (`lr_mode=none`).
- **Test 2** — on a small linear model where exact full-dim DVE is feasible, the projected DVE
  values track the exact values, with correlation → 1 as the projection rank grows (the JL
  guarantee). Full-dim exact DVE is infeasible on a real GPT (the `vocab × d_model` embedding
  makes `M` enormous) — which is exactly why the method projects.

`validate_dve_model.py` (stronger, directly on real models):
- **Test A — ghost capture on GPT2.** The engine's per-sample *projected* gradient must equal a
  brute-force projected gradient assembled from per-sample backward passes (`allclose`). Validates
  the `P_i ⊗ P_o` ghost trick, token-sum, batch rescale, and per-layer dispatch on GPT2.
  Measured: max per-sample relative L2 error **3.7e-7**.
- **Test B — exact unrolled-SGD influence on an MLP.** The semantic ground truth (which the
  reference repo lacks): the *true* first-order influence `dL_test/dw_{s,b}` is computed by
  autodiff *through the SGD trajectory* (double-backprop, no Gauss-Newton, no projection). DVE
  values must track it and beat a plain gradient-dot (TracIn-style) baseline. Measured:
  Pearson(DVE, −influence) **0.994** vs grad-dot baseline **0.979** — the reverse recursion adds
  real signal.

## Experiments

- [`experiments/`](experiments/README.md) — per-batch temporal influence vs training iteration
  (signed, lr-normalized; GPT2-Small on Pile, 10k steps). Holds the figure, diagnostic, binned
  CSV, the `plot_temporal_influence.py` plotter, and the launch commands.

## Notes / caveats

- **SGD vs AdamW.** The unrolling is derived for SGD; `--optimizer adamw` applies the same
  recursion to an AdamW trajectory (as the reference does). Validate correctness with
  `--optimizer sgd` first.
- **Projection consistency.** All stages must use the same `--proj_seed`, `--architecture`, and
  projection params so `P` is reconstructed identically between the train and test passes.
- **`none`-mode stability.** See `--lr_mode` above; prefer `scaled` for real runs.
- **Scale.** Capture writes one file per step (`n_steps × B × total_proj_dim`); per-layer `M` is
  `Σ_layer (k_i·k_o)²`. Fine for GPT2-Tiny/Small; very long runs may want memmap capture.
- **Precision / reproducibility.** The default `bf16` trajectory diverges slightly from `fp32`; the
  DVE values are effectively unchanged (r = 1.0, identical top-k, ~1% magnitude). `bf16` and `fp32`
  runs write to **different** result dirs (`_tdt_bfloat16` suffix), so they never overwrite each
  other. For a bit-for-bit fp32 reference reproduction, pass `--train_dtype float32`.
