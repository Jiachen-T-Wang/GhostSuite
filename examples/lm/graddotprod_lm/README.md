# GradDotProd Language Model Training

This example demonstrates efficient gradient dot product computation during language model pretraining on the Pile dataset using the GradDotProd ghost engine.

## Overview

The GradDotProd engine enables computation of gradient similarities between validation loss and individual training samples in a single backpropagation pass, eliminating the need to materialize any model-sized vectors. At a high-level, this is achieved by cleverly exploiting information already computed during standard backpropagation. For technical details, please refer to Section 4.2 in [Data Shapley in One Training Run](https://openreview.net/pdf?id=HD6bWcj87Y).


## How the Engine Works

1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered separately and stored in `.grad` before optimizer step. 

The per-step update rule is a pluggable `ghostEngines.SelectionPolicy` run by the shared
`ghostEngines.online_selection_step` driver. This example uses `UpdateAll`
(score the batch, log the dots, update on all via subtract-val recovery); `Regular` uses
`NoSelection`. Online-selection variants (`TopK`/`BottomK`/`Threshold`) reuse the same driver — see
`examples/greats/`.


## Quick Start

### Smoke test (synthetic data, no corpus needed)
The fastest way to check the example runs end-to-end is the built-in synthetic
data mode (random tokens) on the tiny model. Requires a GPU but no dataset:

```bash
python examples/lm/graddotprod_lm/main.py --method GradDotProd \
    --train_set synthetic --architecture GPT2-Tiny \
    --batch_size 8 --val_batch_size 4 --max_steps 12 \
    --model_dtype float32 --train_dtype float32
```

A pass prints finite, non-diverging losses and writes a per-sample
`dot_product` tensor under `results/<run>/grad_dotprods/`.

### Get Tokenized Dataset (for real training)
Process the Pile dataset by domain:
```bash
python examples/lm/shared/tokenize_pile_by_domain.py
```
*Note: This process can take ~24 hours depending on your system. For a minimal example, see `examples/minimal/ghost_mlp.py`.* 

### Usage

```bash
cd examples/lm/graddotprod_lm

# Run with gradient dot product computation (default)
./train.sh --batch_size 16

# Run standard training without gradient computation
./train.sh --batch_size 16 --method Regular

# Custom training parameters
./train.sh --batch_size 16 --learning_rate 1e-4 --max_steps 100000

# Replay a previous run while filtering out negative similarity samples
./train.sh --method GradDotProd --replay_run_dir /path/to/old/run --replay_filter_metric dot_product --replay_filter_threshold 0.0

# Replay with shuffled filtered samples
./train.sh --method GradDotProd --replay_run_dir /path/to/old/run --replay_filter_metric dot_product --replay_filter_threshold 0.0 --replay_shuffle --replay_shuffle_seed 42
```

### Key Parameters

- `--method`: Training method (`GradDotProd` or `Regular`)
- `--architecture`: Model architecture (default to `GPT2-Small`)
- `--batch_size`: Training batch size
- `--val_batch_size`: Validation batch size for gradient computation
- `--dot_prod_save_interval`: How often to save gradient metrics
- `--replay_run_dir`: Load train batches from a past GradDotProd run (stored dot-product logs) instead of the live dataset
- `--replay_filter_metric`: Filtering metric for replay data (`dot_product` or `cosine`)
- `--replay_filter_threshold`: Drop samples below this threshold (default 0.0 drops negatives)
- `--replay_rebatch_size`: Batch size to repackage replayed samples (defaults to `--batch_size`)
- `--replay_drop_last`: Drop the final incomplete batch when replay data ends
- `--replay_shuffle`: Shuffle filtered replay samples (loads all filtered samples into memory)
- `--replay_shuffle_seed`: Seed for replay shuffling (defaults to `--seed`)

### Key Configurations

- `RESULTS_DIR` (in `config_file.py`): where training results and metrics are saved.
- Tokenized-Pile locations come from the `PILE_DATA_DIR_TRAIN` / `PILE_DATA_DIR_VAL` /
  `PILE_DATA_DIR_TEST` environment variables (see `../shared/dataloader.py`). No tokenized
  corpus is needed with `--train_set synthetic`.


## Performance: optimized dot-product paths

**The decoupled in-graph + `torch.compile` fast path with the separate-val two-pass engine is
the default** for GradDotProd. Separate-val runs one plain backward on the val batch per step
(harvesting autograd's `.grad` as the cached per-param val gradient) and scores train-only
batches against the cache, instead of carrying the val rows through every combined
forward/backward. Consequences: the training loss/gradients are bit-consistent with regular
training, the val cost is paid once per step instead of once per microbatch, peak memory drops,
and the tied `wte`/`lm_head` moves from the eager capture path (fp32 stashes of the vocab-sized
logits gradient every step) to the compiled in-graph dot. Logged dot-products equal the
combined-batch ones up to a constant positive rescale (`N·(T_tr+T_v)²/(T_tr·T_v)` over
per-microbatch train tokens `T_tr`, val tokens `T_v`, grad-accum `N`) — sign, ranking, and
threshold-0 filtering are unchanged.

| path | how to select | notes |
|---|---|---|
| **decoupled + compile + separate-val** | *(default)* | native layer backwards, dots folded into the `torch.compile`d blocks, val gradient harvested once per step |
| **decoupled + compile, combined batch** | `--no_separate_val` | val rides every scoring batch in one concatenated forward/backward (~+9% step time vs eager on GPT-2-Small); restores the combined-batch dot scale |
| **eager** | `--eager` | per-layer saved-tensor hooks; the reference path. Use for incompatible configs (also selected automatically — see below) |
| **activation checkpointing** | add `--decoupled_mem_budget 0.5` | recompute in backward to cut peak memory (≈−27% on GPT-2-Medium for +29% time); tunable in `(0,1]`. The lever for scaling to GPT-2-Medium/Large |

```bash
# Default run is already the optimized fast path:
python examples/lm/graddotprod_lm/main.py --method GradDotProd --train_set synthetic \
    --architecture GPT2-Small --batch_size 16 --val_batch_size 1 --max_steps 20

# Force the eager engine:
python examples/lm/graddotprod_lm/main.py --method GradDotProd ... --eager
```

**Fast-path scope + automatic fallback.** The decoupled fast path applies to GPT-2 token models on a
single GPU with bf16/fp32. For runs that can't use it — multi-GPU/DDP, float16 (`GradScaler`), or a
non-GPT-2 architecture (e.g. LLaVA) — `main.py` prints a notice and falls back to the **eager** engine
automatically. Pass `--eager` to select eager explicitly. (`--no_decoupled_compile` keeps the decoupled
path but skips compile; usually slower than `--eager`, for debugging.)

**Gradient accumulation (`--gradient_accumulation_steps > 1`) is supported** on all paths. Each
microstep draws a distinct train sub-batch of `--batch_size` and the per-sample dot-products are
collected per microstep. On the separate-val default the val gradient is harvested once per step
and each train-only microstep projects against it (the training gradient is autograd-native — no
recovery). On the combined paths (`--no_separate_val` / `--eager`) the validation batch rides in
every microstep's combined forward and the per-microstep validation gradients are summed for a
single subtract-val recovery before the optimizer step; either way the applied training gradient
equals the mean over all `N * batch_size` train samples. Note the dot-product scores carry
loss-rescale factors (`1/N^2` on the combined paths; `1/(N·T_tr·T_v)`-style on separate-val) —
consistent within a run, not comparable in absolute scale across different `N` or paths.

**Tied weights:** the token-embedding ↔ LM-head tie (standard GPT-2) is handled by all paths
(including the gradient cross-terms). `--no_tie_weights` unties if desired.

**Not recommended:** `--decoupled_compile_toplevel` (compile the output `lm_head` + final norm) —
implemented for generality but it *regresses* GPT-2 (the 50304-vocab `lm_head` compiles to a slower
kernel than eager cuBLAS); leave it off.


## Training Loop Integration

The training loop (`training_loop.py`) delegates each step to the shared selection driver,
which runs the ghost scoring pass over the combined train ++ val batch, logs the per-sample
dot-products, and performs the update prescribed by the selection policy:

```python
from ghostEngines import online_selection_step

_scores, _idx, loss = online_selection_step(
    manager=ghost_engine, model=model, optimizer=optimizer,
    scaler=scaler, ctx=ctx, forward_fn=forward_fn,
    policy=policy, iter_num=iter_num, grad_clip=grad_clip,
    grad_accum=gradient_accumulation_steps,
    draw_microbatch=draw_microbatch, ddp=ddp,
)
```

For custom loops the manager also exposes the underlying step primitives
(`attach_train_batch`, `prepare_forward_input`, `prepare_gradients`, `aggregate_and_log`).
