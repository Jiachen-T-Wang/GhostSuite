# Examples Directory

This directory contains runnable examples demonstrating the Ghost Engine framework for
efficient per-sample gradient computation. It is organized into three subfolders, each
covered by a section below:

1. [`minimal/`](#1-minimal-examples-minimal) — smallest end-to-end demos, no data prep.
2. [`lm/`](#2-standalone-language-model-examples-lm) — standalone language-model training.
3. [`torchtitan/`](#3-llm-pretraining-with-torchtitan-torchtitan) — large-scale LLM pretraining.


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

For faster runs, this example also supports the compiled fast path (`--decoupled_fn
--decoupled_compile`, ~+9% step time on GPT-2-Small), the batched lever (`GHOST_BATCHED_DOTPROD=1`),
and activation checkpointing (`--decoupled_mem_budget`) — see the README's "Performance" section.

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

### Dot-product levers and the speed ↔ memory tradeoff

The TorchTitan GradDotProd integration computes the train↔val gradient dot-products by running a
single forward/backward on a **combined `train + val` batch**. Its runtime levers are exposed as
`--ghost.*` flags (defaults live in `llama3_130m_ghost.toml`):

| `--ghost.*` lever | default | effect |
|---|---|---|
| `subtract_val` | on | recover train grads after backward instead of masking activations |
| `decoupled_fn` | on | graph-clean decoupled-Function path so `torch.compile` can compile the model |
| `compile_toplevel` | on | also compile the output Linear's dot (memory-free; the rest of the gain) |
| `opsac_mm_every` | 1 | op-SAC mm save-fraction: recompute every N-th matmul (1 = all → min memory; ↑N = more memory, faster). Active when `selective_ac_option="op"` |
| `batched_dotprod` | off | eager grouped dot-product — the no-compile fallback (superseded by `decoupled_fn`) |
| `regional_compile` | off | regional compile of RoPE/SwiGLU (helps A100, regresses H200) |

The default also sets `[compile] enable = true` and `[activation_checkpoint] mode = "selective",
selective_ac_option = "op"`. Together these run a compiled fast path with **op-level selective
activation checkpointing** that is faster than the eager engine **at the same loss** (bit-identical
*and* dot-product-identical). Activation checkpointing then gives a single speed↔memory dial. The
default is **op-SAC `mme1`** — the runtime-memory frontier point that runs *below* the eager
engine's peak memory while still ~15% faster. Numbers: Llama-3 130M, seq 4096, train bs2 + val bs2,
single H200, one session; throughput is tokens/s, loss-identical in every row:

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
with `--ghost.no-decoupled_fn --ghost.no-batched_dotprod` (compile is auto-disabled on the eager
path).
