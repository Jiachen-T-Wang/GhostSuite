# Data Value Embedding (DVEmb)

Re-implementation of **Data Value Embedding** ([arXiv:2412.09538](https://arxiv.org/abs/2412.09538))
on top of GhostSuite's `gradProjection` engine. DVEmb answers: *which training examples — and at which
training step — most shaped the trained model's behavior on a given test example?* (its "in-run" or
temporal influence). It does this by attaching a compact **value embedding** to every training
point, then scoring test examples against those embeddings.

The expensive part — per-sample gradients along the *whole* training run — is captured cheaply by
the ghost projection `P = P_i ⊗ P_o`, which never materializes a per-sample gradient. (This replaces
the reference paper's random-LoRA Kronecker sketch with GhostSuite's Gaussian/orthonormal JL
projection.)

## The pipeline: four stages

DVEmb runs as a **pipeline of four stages**. Here a *stage* is one step of the computation, turned on
by a boolean flag on `main.py`. The stages run **in order**, each consuming the previous stage's
output, and you can run them all in a single command or as separate invocations. Separate
invocations still find each other's files because every stage derives the **same run directory**
from the config (architecture, projection params, optimizer, steps, dtype, …) and writes into
per-stage subfolders beneath it.

| # | Stage flag | What it does | Reads → writes (under the run dir) |
|---|------------|--------------|------------------------------------|
| 1 | `--train_and_store_grad` | Train the model with real optimizer steps; at **every** step, capture the per-sample *projected* gradients of that step's batch. This captured trajectory is what the rest of the pipeline unrolls. | training data → `capture/proj_iter_*.pt` (one file per step, tagged with the step's `lr`, `order`, and `batch_idx`) + `capture/final_model.pt` |
| 2 | `--compute_embedding` | The core DVEmb step: a reverse recursion over the captured gradients that folds each step's contribution *backward* through the later steps' SGD Jacobian, producing one **value embedding** per training point. | `capture/` → `embedding/embed_iter_*.pt` |
| 3 | `--compute_value` | Load the final checkpoint, project the **test** gradients with the *same* `P`, and dot them against the training embeddings to form the value matrix. | `embedding/` + test data → `value/values.pt`, shape `[n_test, n_train]` |
| 4 | `--attribute` | For each test example, rank and print the most- and least-valuable training examples (and the step at which each appeared). | `value/values.pt` → stdout |

### Quick start (synthetic smoke, no corpus)

```bash
# Run all four stages end to end on random tokens (CPU-friendly, GPT2-Tiny):
python main.py --data_source synthetic --architecture GPT2-Tiny --device cpu \
    --optimizer sgd --learning_rate 0.05 --max_steps 8 --batch_size 4 \
    --n_test 8 --test_batch_size 4 --proj_rank_total 64 \
    --model_dtype float32 --train_dtype float32 --proj_dtype float32 \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

Pass only some of the four flags to run a subset of stages — e.g. re-run just `--attribute` against
an existing `value/values.pt`, or redo `--compute_value` for a new test set without retraining.

## Method (what Stage 2 computes)

For a model trained with SGD `θ_{t+1} = θ_t − η_t (1/B) Σ_b g_{t,b}`, the first-order in-run
influence of training point `(s,b)` (example `b` at step `s`) on a test point is `⟨g_test, e_{s,b}⟩`,
where the **data value embedding** is

```
e_s = g_s − M_{>s} g_s ,      M_{>s−1} = M_{>s} + (η_s / B_s) Σ_b e_{s,b} g_{s,b}ᵀ
```

i.e. `e_s` is the training gradient `g_s` after being transformed by the unrolled SGD Jacobian
`∏_{t>s}(I − η_t H_t)` of all *later* steps (`H_t ≈ (1/B) Σ_b g_t g_tᵀ`). Stage 2 evaluates this
recursion in reverse (last step → first), maintaining `M` **per layer** (block-diagonal across
layers); all gradients live in the projected space, which is what makes `M` tractable. Each step's
recorded learning rate `η_s` and `1/B_s` Gauss-Newton norm are folded into the recursion, keeping it
numerically consistent with the actual (small-lr, contractive) trajectory.

## Capture backend & precision (Stage 1)

Stage 1 is where the per-sample gradients are captured, and it has two interchangeable backends:

- **Default — decoupled in-graph + `torch.compile`** (`--decoupled_compile`, on for CUDA) at
  **`--train_dtype bfloat16`** (fp32 master weights + bf16 autocast; `--proj_dtype float32` keeps the
  projection itself in fp32). On an H200 this is **~25% faster** than the eager backend (GPT2-Small,
  bs 8, block 1024), and the DVEmb values it produces match the fp32 eager reference to
  **Pearson r = 1.0 / identical top-k rankings** (~1% difference in value *magnitude* from bf16
  rounding).
- **Eager fallback** — `--no_decoupled_compile` uses the hook-based `GradProjLoraEngine`. Required
  for **Conv1D**-based models (the decoupled backend supports only `nn.Linear`/`nn.Embedding`). On a
  non-CUDA device Stage 1 auto-selects this backend (compile is CUDA-only).

`--ac_budget b` (in `(0, 1]`, decoupled backend only) turns on compile-native activation
checkpointing (Inductor min-cut): lower `b` recomputes more in the backward pass to cut peak memory
(e.g. `0.5` ≈ −26% peak at ~+7% step time).

The two backends are numerically equivalent (bit-exact in fp32), so this choice affects speed and
memory, not the DVEmb values.

## Real Pile run

The Pile loader (`shared/dataloader.py`) reads per-domain GPT-2 `.bin` files via env vars:

```bash
export PILE_DATA_DIR_TRAIN=/path/to/pile/pile-train
export PILE_DATA_DIR_VAL=/path/to/pile/pile-val-gpt2
export PILE_DATA_DIR_TEST=/path/to/pile/pile-test-gpt2

python main.py --data_source pile --architecture GPT2-Small --device cuda \
    --optimizer adamw --learning_rate 3e-4 --max_steps 500 --batch_size 8 \
    --n_test 64 --proj_rank_total 256 \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

## Experiments

- [`experiments/`](experiments/README.md) — per-batch temporal influence vs training iteration
  (signed, lr-normalized; GPT2-Small on Pile, 10k steps). Holds the figure, the
  `plot_temporal_influence.py` plotter, and the launch commands.

## Notes / caveats

- **SGD vs AdamW.** The unrolling is derived for SGD; `--optimizer adamw` applies the same recursion
  to an AdamW trajectory (as the reference does), which is an approximation — prefer `--optimizer
  sgd` when you need the influence to be exact.
- **Projection consistency.** All stages must use the same `--proj_seed`, `--architecture`, and
  projection params so `P` is reconstructed identically between the train (Stage 1) and test
  (Stage 3) passes. (The run directory encodes these, so a mismatch resolves to a *different* dir and
  fails loudly rather than silently mixing incompatible projections.)
- **Scale.** Stage 1 writes one file per step (`n_steps × B × total_proj_dim`); per-layer `M` is
  `Σ_layer (k_i·k_o)²`. Fine for GPT2-Tiny/Small; very long runs may want memmap capture.
- **Precision / reproducibility.** The default `bf16` trajectory diverges slightly from `fp32`; the
  DVEmb values are effectively unchanged (r = 1.0, identical top-k, ~1% magnitude). `bf16` and `fp32`
  runs write to **different** run dirs (a `_tdt_bfloat16` suffix), so they never overwrite each
  other. For a bit-for-bit fp32 reference reproduction, pass `--train_dtype float32`.
