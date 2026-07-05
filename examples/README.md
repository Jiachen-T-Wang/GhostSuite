# Examples Directory

This directory contains runnable examples demonstrating the Ghost Engine framework for
efficient per-sample gradient computation. It is organized into six subfolders, each
covered by a section below:

1. [`minimal/`](#1-minimal-examples-minimal) — smallest end-to-end demos, no data prep.
2. [`lm/`](#2-standalone-language-model-examples-lm) — standalone language-model training.
3. [`torchtitan/`](#3-llm-pretraining-with-torchtitan-torchtitan) — large-scale LLM pretraining.
4. [`greats/`](#4-online-batch-selection-with-greats-greats) — online batch selection built
   on the ghost dot-products.
5. [`opus/`](#5-sketched-online-data-selection-with-opus-opus) — sketched, diversity-aware
   online data selection built on the ghost gradient projections.
6. [`dvemb_lm/`](#6-data-value-embedding-dvemb_lm) — trajectory-specific data valuation built on
   the ghost gradient projections.


## 1. Minimal Examples (`minimal/`)

Simplified, dependency-light implementations on synthetic data (no corpus, runs on CPU)
that demonstrate the core engine usage:

- **`minimal/ghost_mlp.py`**: Basic GradDotProd usage for MLP models
  - Trains for 10 steps on synthetic data
  - Prints per-parameter gradient dot-products

- **`minimal/ghost_gradproj_mlp.py`**: Per-sample gradient projection computation and storage for MLP

- **`minimal/ghost_gradproj_lm.py`**: Per-sample gradient projection computation and storage for language models
  - Projects gradients for transformer layers
  - Demonstrates similarity computation from saved projections

**Run minimal examples:**
```bash
python examples/minimal/ghost_mlp.py
python examples/minimal/ghost_gradproj_mlp.py --mode project --proj_rank_total 64
python examples/minimal/ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc"
```


## 2. Standalone Language Model Examples (`lm/`)

Self-contained language-model training scripts that wrap the ghost engines. Both ship a
built-in **synthetic data mode** (random tokens, no tokenized corpus required) that runs
the example end-to-end as a quick smoke test on a tiny model. To train on real data,
tokenize the Pile (see each example's `README.md`) and pass `--train_set pile` /
`--data_source pile`.

### `lm/graddotprod_lm/` — online gradient dot-products
Pair-wise gradient dot-product computation **during** language-model training on the Pile.
Useful for research that needs gradient similarities computed online as the model trains
(e.g., online data selection, reweighting, curriculum learning).

```bash
# Synthetic smoke test (needs a GPU)
python examples/lm/graddotprod_lm/main.py --method GradDotProd \
    --train_set synthetic --architecture GPT2-Tiny \
    --batch_size 8 --val_batch_size 4 --max_steps 12 \
    --model_dtype float32 --train_dtype float32
```

By default this example uses the decoupled in-graph + `torch.compile` fast path (~+9% step time on
GPT-2-Small; it auto-falls back to the eager engine for incompatible configs). Pass `--eager` for
the eager engine, or add `--decoupled_mem_budget` for activation-checkpointing memory savings — see
the README's "Performance" section.

See [`lm/graddotprod_lm/README.md`](lm/graddotprod_lm/README.md) for detailed instructions.

### `lm/gradproj_lm/` — offline per-sample gradient projection
Per-sample gradient projection computation and storage for a **fixed** model checkpoint.
Useful for research that needs gradient similarities for a fixed checkpoint across a large
dataset (e.g., offline data selection, clustering), where the whole dataset cannot fit in a
single batch.

```bash
# Synthetic smoke test (runs on CPU too)
python examples/lm/gradproj_lm/main.py --data_source synthetic \
    --architecture GPT2-Tiny --device cuda --batch_size 4 --max_samples 8 \
    --proj_dtype float32 --model_dtype float32 --train_dtype float32 \
    --output_dir ./outputs_smoke
```

See [`lm/gradproj_lm/README.md`](lm/gradproj_lm/README.md) for detailed instructions.


## 3. LLM Pretraining with TorchTitan (`torchtitan/`)

[TorchTitan](https://github.com/pytorch/torchtitan) is a PyTorch-native training stack for
large-scale model development. This subfolder integrates the GradDotProd engine into
TorchTitan. Run the integration with the Llama 3 130M ghost config:

```bash
CONFIG_FILE="./examples/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" \
    ./examples/torchtitan/run_train_with_ghost.sh
```

Data prerequisite: set `C4_LOCAL_DIR` (training corpus) and `PILE_TEST_DIR` (ghost/validation
batches) to your local dataset directories before running.

### Dot-product levers and the speed ↔ memory tradeoff

The TorchTitan GradDotProd integration computes the train↔val gradient dot-products during
training. By default it uses the **separate-val two-pass engine**: each optimizer step runs one
plain backward on the fixed val batch (harvesting autograd's `.grad` as the cached per-param val
gradient) and then the train microbatches *without* appended val rows, their in-graph dots
projecting against the cache. Training loss and gradients are **bit-consistent with regular
training** (the loss curve matches `torchtitan.train` step-for-step), the val cost is paid once
per step instead of once per microbatch, and peak memory is roughly half the older combined-batch
engine's. Dots equal the combined-batch dots up to a constant per-config rescale
(`(T_tr+T_v)²/(T_tr·T_v)` at grad-accum 1). `--ghost.no-separate_val` restores the combined-batch
engine (one forward/backward on the concatenated `train + val` batch). Runtime levers
(defaults live in `llama3_130m_ghost.toml`):

| `--ghost.*` lever | default | effect |
|---|---|---|
| `separate_val` | on | two-pass engine: val backward once per step + train-only microbatches (faster under grad accumulation, ~half the memory, loss matches regular training) |
| `compile_loss` | on | compile the loss on the ghost path (otherwise it runs EAGER fp32 full-vocab CE — ~17 ms/step here — because the deferred compile builds it with compile disabled) |
| `subtract_val` | on | (combined path) recover train grads after backward instead of masking activations |
| `decoupled_fn` | on | graph-clean decoupled-Function path so `torch.compile` can compile the model |
| `compile_toplevel` | on | also compile the output Linear's dot (memory-free; the rest of the gain) |
| `opsac_mm_every` | 1 | op-SAC mm save-fraction: recompute every N-th matmul (1 = all → min memory; ↑N = more memory, faster). Active when `selective_ac_option="op"` |
| `regional_compile` | off | regional compile of RoPE/SwiGLU (helps A100, regresses H200) |

Measured on a single H200 / Llama-3 130M / seq 4096 at the standard pretraining shape (global
batch 12 = 6 accumulation microbatches of local bs 2, val bs 2, no AC): regular training
285.9 ms/step, separate-val ghost 429.3 ms (**1.50×**), combined-batch ghost 479.0 ms (1.68×);
at global batch 24 the ratio drops to 1.40× as the val pass amortizes. The remaining overhead is
the algorithm's floor — one val backward per step (amortizing as 1/N with the accumulation
factor) plus one projection GEMM per Linear per microbatch (the dot-product readout itself,
~25% of the model's Linear FLOPs) — not implementation slack: kernel-level profiling shows the
extra GPU time is exactly those GEMMs and the val pass, at the same efficiency as the model's
own kernels.

The default also sets `[compile] enable = true` and `[activation_checkpoint] mode = "selective",
selective_ac_option = "op"`. Together these run a compiled fast path with **op-level selective
activation checkpointing** that is faster than the eager engine **at the same loss** (bit-identical
*and* dot-product-identical). Activation checkpointing then gives a single speed↔memory dial. The
default is **op-SAC `mme1`** — the runtime-memory frontier point that runs *below* the eager
engine's peak memory while still ~15% faster. Numbers below were measured on the **combined-batch
engine** (`--ghost.no-separate_val --ghost.no-compile_loss`); the separate-val default shifts
every point down in memory by roughly half and the AC dial works the same way. Llama-3 130M, seq 4096,
train bs2 + val bs2, single H200, one session; throughput is tokens/s, loss-identical in every row:

| config | flags | throughput vs eager | peak memory |
|---|---|---:|---:|
| **default** (op-SAC `mme1`) | *(none)* | **+15%** | **−9%** (below eager) |
| op-SAC, more memory | `--ghost.opsac_mm_every=4` | +18% | +14% |
| op-SAC, more memory | `--ghost.opsac_mm_every=8` | +19% | +16% |
| max speed (no AC) | `--activation_checkpoint.mode=none` | **+24%** | +38% |
| absolute min memory | `--activation_checkpoint.mode=full` | +8% | −10% |

op-SAC strictly dominates the older layer-frequency dial: it sits on the pareto frontier at every
memory level, whereas `--activation_checkpoint.selective_ac_option=2` and the `torch.compile`
`memory_budget` partitioner are both off-frontier here (the latter gives no peak-memory reduction —
the fp32 logits tensor floors it).

Caveats: the combined `train + val` batch and the compiled path both raise peak memory, so at the
*max speed* (no-AC) setting a ghost run hits the memory ceiling **earlier** than a same-train-batch
baseline (the fp32 logits tensor, `batch · seq · vocab · 4` bytes, dominates for large-vocab
models). The default op-SAC `mme1` keeps peak memory *below* the eager engine, so it has the most
headroom — prefer it (or `mode=full`) on an 80 GB A100. To fall back to the pure eager engine, run
with `--ghost.no-decoupled_fn` (compile is auto-disabled on the eager path).


## 4. Online Batch Selection with GREATS (`greats/`)

[GREATS](https://github.com/Jiachen-T-Wang/GREATS) (Wang et al., NeurIPS 2024) selects, at
**every** training step, the subset of a candidate batch whose gradients best reduce the
validation loss — using the per-sample gradient dot products the `GradDotProd` engine already
computes in one backward pass. This is an *online, model-dependent* selector (scores are
recomputed each step), not a static GradNorm/TracIN baseline.

- **`greats/pretrain/`** — online **first-order** selection during GPT-2 pretraining, reusing
  the `lm/` shared model + data utilities (synthetic / Pile).
- **`greats/sft/`** — online selection during LoRA instruction tuning (MMLU target),
  supporting both first-order and the default **second-order** Gram-based greedy variant.

```bash
# Synthetic smoke test (1 GPU, no corpus)
python examples/greats/pretrain/main.py --method GREATS --train_set synthetic \
    --architecture GPT2-Tiny --candidate_batch_size 16 --batch_size 8 \
    --val_batch_size 4 --max_steps 12 --eval_interval 4 \
    --model_dtype float32 --train_dtype float32
```

See [`greats/README.md`](greats/README.md), [`greats/pretrain/README.md`](greats/pretrain/README.md),
and [`greats/sft/README.md`](greats/sft/README.md) for details.


## 5. Sketched Online Data Selection with OPUS (`opus/`)

[OPUS](https://github.com/gszfwsb/OPUS) (Wang et al., ICML 2026) extends the GREATS line:
candidates are scored by **sketched, optimizer-preconditioned** gradient inner products
against a proxy batch, plus a candidate-candidate similarity (Gram) matrix that feeds a
diversity-aware Boltzmann **stochastic-greedy** selection. This port computes the per-sample
sketches with the `GradProjLora` engine's two-sided factorized projection inside the backward
— no per-sample gradient is ever materialized, making the scoring pass 5–6× faster than the
reference implementation's CountSketch at matched fidelity — and shares the `lm/` model/data
stack and the GREATS pretrain protocol, so the two selection rules are directly comparable
(see [`opus/experiments/`](opus/experiments/README.md) for the Pile comparison).

```bash
# Synthetic smoke test (1 GPU, no corpus)
python examples/opus/main.py --method OPUS --train_set synthetic \
    --architecture GPT2-Tiny --candidate_batch_size 16 --batch_size 8 \
    --val_batch_size 4 --max_steps 12 --eval_interval 4 \
    --model_dtype float32 --train_dtype float32
```

See [`opus/README.md`](opus/README.md) for the algorithm, flags, and the differences from the
reference implementation.


## 6. Data Value Embedding (`dvemb_lm/`)

[Data Value Embedding](https://arxiv.org/abs/2412.09538) (DVEmb) attributes a trained model's
behavior on **test** examples back to each **training** example *and the training step at which it
appeared* (in-run / temporal influence), via a compact per-training-point embedding. Unlike the
GradDotProd examples (pairwise similarities) and GREATS (online selection), DVEmb captures per-step
per-sample **projected** gradients along the *whole* training trajectory, then a reverse recursion
unrolls the SGD Jacobian to turn them into value embeddings — built on the same ghost projection
`P = P_i ⊗ P_o` as `lm/gradproj_lm/` (no per-sample gradient is ever materialized).

It is a 4-stage pipeline (`--train_and_store_grad` → `--compute_embedding` → `--compute_value` →
`--attribute`), sharing config-derived paths so the stages can run together or separately.

```bash
# All four stages, synthetic smoke (CPU-friendly, GPT2-Tiny)
python examples/dvemb_lm/main.py --data_source synthetic --architecture GPT2-Tiny --device cpu \
    --optimizer sgd --learning_rate 0.05 --max_steps 8 --batch_size 4 \
    --n_test 8 --test_batch_size 4 --proj_rank_total 64 \
    --model_dtype float32 --train_dtype float32 --proj_dtype float32 \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

Stage-1 capture defaults (on CUDA) to a **decoupled in-graph + `torch.compile`** fast path at
`--train_dtype bfloat16` — numerically matching the eager hook engine while ~25% faster on an H200;
pass `--no_decoupled_compile` to use the eager engine, or `--ac_budget b` for compile-native
activation checkpointing. See [`dvemb_lm/README.md`](dvemb_lm/README.md) for the method, the LR-mode /
projection options, and the real-Pile run.
