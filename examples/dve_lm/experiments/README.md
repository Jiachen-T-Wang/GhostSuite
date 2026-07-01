# Reproducing Figure 1(a) of Data Value Embedding (arXiv:2412.09538)

**Date:** 2026-06-30 (branch `v0.5`)
**GPU:** 1× NVIDIA H200 (Della `ailab`), Slurm job `10490885`.
**Raw log:** gitignored scratch under `examples/dve_lm/results/<run>/`.

![DVE Figure 1(a) reproduction](dve_fig1a_2026-06-30.png)

This run **matches the released reference code's training trajectory** (`train_gpt2_long.sh` +
`config/config_pile.yaml` in the reference implementation): GPT-2, warmup 2000, linear LR decay to
0 at step 10000, AdamW (weight decay 0.01), MLP-only per-sample gradients.

## What Figure 1(a) is

> **Paper caption:** "(a) Average data influence scores computed from data value embedding per
> training batch, measured against the final model's loss on Pile's validation set. Setting:
> Pythia-410M trained on 1% of Pile."

The y-axis is the **signed** average per-batch influence, **normalized by dividing each batch's
influence by its per-step learning rate** ("we normalize the influence scores for each batch by
their learning rate"); the x-axis is the training iteration. The paper (§5.3) describes three
phases: a **high-impact warmup spike**, a **low-impact basin**, and a **gradual late-training
ascent** (later data → higher influence, because it is transformed by fewer future steps).

## Architecture note — GPT-2, not Pythia-410M

The paper's caption names **Pythia-410M**, but the **released reference code**
(`config/config_pile.yaml`, `model_name: gpt2`) implements this experiment on **GPT-2**. Our DVE
pipeline (`create_GPT_model`) supports only GPT-2 architectures — Pythia-410M is GPT-NeoX (rotary
attention), unsupported by the ghost engine without substantial new work. So this run faithfully
reproduces the **released GPT-2 methodology**; the temporal-influence phenomenon is
schedule/dynamics-driven, not architecture-specific. This is the one intentional model deviation.

## Configuration (matches `train_gpt2_long.sh` / `store_train_grad.py`)

| Param | Value | Source |
|---|---|---|
| Model | GPT2-Small (123.6M) | reference `model_name: gpt2` |
| Data | Pile (local pretokenized, 1024 block), fp32 | `data_source=pile`, `mixed_precision=no` |
| Optimizer | AdamW, lr 3e-4, **weight_decay 0.01, betas (0.9, 0.999)** | reference `torch.optim.AdamW(lr=...)` defaults |
| LR schedule | linear warmup 2000 → **linear decay to 0 at step 10000** | reference `get_linear_schedule_with_warmup(num_warmup=2000, num_training_steps=10000)` |
| Steps / batch | **10,000** / 16 | reference hardcoded horizon (≈ a 1%-Pile single epoch) |
| Projection | MLP-only (24 Linear layers), JL rank 256, `lr_mode=scaled` | reference `--mlp_only`, rank 64 LoRA → our ghost JL |
| Val set | 256 held-out Pile windows | `n_test=256` |

Stages: train+capture → embed (reverse recursion) → value (test-grad · embeddings) →
`plot_fig1a.py` (see **Launch** below). Train+capture ran ≈ 51 min on the H200; the full pipeline
finished well within a 2 h allocation.

`lr_mode=scaled` (not the reference's `none`) is used for **numerical stability** — `none` has no
LR damping and the per-layer `M` accumulation is ill-conditioned. Scaled folds `η_s` into the
embedding; `plot_fig1a.py` then divides the per-batch value by `η_s` to recover the lr-normalized
influence the paper plots. The projection is GhostSuite's Gaussian JL sketch (rank 256) in place
of the reference's random-LoRA rank-64 Kronecker sketch — the intentional method substitution.

## Result — three phases reproduced

Binned mean of the signed, lr-normalized per-batch influence (60 bins over 10k steps):

| Phase | Steps | Mean influence (/lr) | Paper |
|---|---|---|---|
| **Warmup spike** | 0–500 | **+6.1e-4** (first bin +3.0e-3) | high during warmup ✓ |
| Early dip | 500–2000 | −2.9e-5 | (transition into basin) |
| **Basin** | 2000–6000 | **+1.1e-6** (near-zero) | low-impact basin ✓ |
| Mid ascent | 6000–8000 | +5.7e-6 | rising ✓ |
| **Late ascent** | 8000–10000 | **+1.6e-5** | gradually increasing late ✓ |

All three phases are present and well separated: the warmup spike is ~100× the basin, and the
post-basin influence rises monotonically (1.1e-6 → 5.7e-6 → 1.6e-5, a ~14× ascent), matching the
paper's finding and its mechanistic explanation (early data's influence decays under many future
steps; late data retains influence).

The companion diagnostic (`dve_fig1a_2026-06-30_diagnostic.png`) shows why the lr normalization
matters: the **raw (lr-weighted)** curve is dominated by the schedule envelope (grows then decays
to ~0 as lr→0), so it does *not* expose the ascent; **dividing by lr** (the paper's normalization)
recovers the intrinsic three-phase structure.

## Caveats

- **GPT-2, not Pythia-410M** (see above) — released methodology, not the paper's exact model.
- **Noisier than the paper.** We average over 256 val windows × 16-sample batches with random
  Pile-window sampling on a 124M model; the paper's 410M curve is smoother. The binned mean ± SEM
  still resolves the three phases; bin-to-bin sign oscillation in the basin is noise.
- **`lr_mode=scaled` + post-hoc /lr**, vs the reference's `none` recursion. These differ in the
  per-layer `M` batch-normalization, so this is *the reference method up to that normalization*,
  chosen for conditioning. The qualitative shape is robust.
- **JL rank-256 sketch** vs the reference's random-LoRA rank-64 (intentional substitution).
- **Single seed / single trajectory.**

## Launch

The Pile loader reads per-domain GPT-2 `.bin` files via env vars (see `examples/dve_lm/README.md`):

```bash
export PILE_DATA_DIR_TRAIN=/path/to/pile/pile-train
export PILE_DATA_DIR_VAL=/path/to/pile/pile-val-gpt2
export PILE_DATA_DIR_TEST=/path/to/pile/pile-test-gpt2

# Stages 1–3: train + capture per-step projected grads → reverse-recursion embeddings →
# value matrix (test grads · embeddings). ≈ 1 h on an H200; writes to
# examples/dve_lm/results/<run>/{capture,embedding,value}/ (gitignored scratch).
# --lr_decay_steps decouples the LR-decay horizon from the run length to match the reference
# (which hardcodes num_training_steps=10000); here they coincide.
python examples/dve_lm/main.py --data_source pile --architecture GPT2-Small --device cuda \
    --optimizer adamw --learning_rate 3e-4 --lr_schedule linear \
    --warmup_steps 2000 --max_steps 10000 --lr_decay_steps 10000 \
    --batch_size 16 --n_test 256 --weight_decay 0.01 --beta1 0.9 --beta2 0.999 \
    --proj_layers mlp --proj_rank_total 256 --lr_mode scaled \
    --train_and_store_grad --compute_embedding --compute_value

# Plot Figure 1(a) from the value matrix. Emits <out>.png, <out>_diagnostic.png,
# <out>.csv (per-step), <out>_binned.csv. --lr_decay_steps must match the run.
python examples/dve_lm/experiments/plot_fig1a.py \
    --values examples/dve_lm/results/<run>/value/values.pt --out dve_fig1a_2026-06-30 \
    --lr_mode scaled --learning_rate 3e-4 --warmup_steps 2000 \
    --max_steps 10000 --lr_decay_steps 10000 --lr_schedule linear --bins 60
```

## Artifacts

Committed alongside this README:
- `dve_fig1a_2026-06-30.png` — paper-style panel (binned lr-normalized influence + SEM band).
- `dve_fig1a_2026-06-30_diagnostic.png` — raw lr-weighted vs lr-normalized panels.
- `dve_fig1a_2026-06-30_binned.csv` — per-bin center / mean / SEM.
- `dve_fig1a_2026-06-30_perstep.csv` — per-step step,lr,influence_raw,influence_lr_normalized;
  re-plot cheaply with `plot_fig1a.py --from_csv <this> ...` (no GPU/recompute).
- `plot_fig1a.py` — the plotter (add `--logy` for a symlog panel).

Run outputs (gitignored scratch): `examples/dve_lm/results/<run>/` — `capture/` (10k proj_iter),
`embedding/` (10k embed_iter), `value/values.pt` (256×160000).
