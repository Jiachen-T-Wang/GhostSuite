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

## Pipeline (4 stages)

Stages share config-derived paths, so they can run in one command or separately.

```bash
# All four stages, synthetic smoke (CPU-friendly, GPT2-Tiny):
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
export PILE_DATA_DIR_TRAIN=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-train
export PILE_DATA_DIR_VAL=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-val-gpt2
export PILE_DATA_DIR_TEST=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-test-gpt2

python main.py --data_source pile --architecture GPT2-Small --device cuda \
    --optimizer adamw --learning_rate 3e-4 --max_steps 500 --batch_size 8 \
    --n_test 64 --proj_rank_total 256 --lr_mode scaled \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

## Correctness

```bash
python validate_dve.py
```
- **Test 1** — the recursion matches a literal port of the reference exactly (`lr_mode=none`).
- **Test 2** — on a small linear model where exact full-dim DVE is feasible, the projected DVE
  values track the exact values, with correlation → 1 as the projection rank grows (the JL
  guarantee). Full-dim exact DVE is infeasible on a real GPT (the `vocab × d_model` embedding
  makes `M` enormous) — which is exactly why the method projects.

## Notes / caveats

- **SGD vs AdamW.** The unrolling is derived for SGD; `--optimizer adamw` applies the same
  recursion to an AdamW trajectory (as the reference does). Validate correctness with
  `--optimizer sgd` first.
- **Projection consistency.** All stages must use the same `--proj_seed`, `--architecture`, and
  projection params so `P` is reconstructed identically between the train and test passes.
- **`none`-mode stability.** See `--lr_mode` above; prefer `scaled` for real runs.
- **Scale.** Capture writes one file per step (`n_steps × B × total_proj_dim`); per-layer `M` is
  `Σ_layer (k_i·k_o)²`. Fine for GPT2-Tiny/Small; very long runs may want memmap capture.
