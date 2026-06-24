# "Ghost" Suites for Fast Gradient Information Calculation


## Introduction
Computing per-sample gradient information and pair-wise gradient similarity is often the computational bottleneck for data-centric research (e.g., data selection, synthetic data generation). A naive approach would require setting the batch size to 1, backpropagating on the loss of each training sample, and storing all the huge gradient vectors. Consequently, this approach would be computationally prohibitive for practical applications. 

In [Data Shapley in One Training Run](https://openreview.net/pdf?id=HD6bWcj87Y) (ICLR'25 Outstanding Paper Runner-up), we proposed a highly efficient method to obtain per-sample gradient information. It turns out that we can compute the gradient dot-product between every pair of data points within a large batch in just a single backpropagation. At high level, the technique exploits information that's already being computed during standard backpropagation with respect to the aggregated loss on a batch of data points. 

This repository provides a clean, drop-in implementation of "ghost"-based techniques for fast per-sample gradient information calculation. Our goal is to enable per-sample gradient computation and extraction with **minimal code changes**—often just a few lines added to your existing model training loop.


## Available Engines
- `GradDotProdEngine`
  - Purpose: Online computation of gradient similarities between validation loss and individual training samples in a single backprop pass.
  - Core idea: Reuse activations and output gradients already computed during backprop to obtain per‑parameter dot products without materializing model‑sized gradients; typically concatenates a small validation batch with the training batch.
  - Best for: computing pair-wise gradient similarities through the entire training process (e.g., online data selection, reweighting, curriculum learning, or analyzing training dynamics). 

- `GradProjLoRAEngine`
  - Purpose: Offline, corpus‑scale analysis by storing low‑dimensional per‑sample gradient projections to disk for later similarity analysis.
  - Core idea: Similar to `GradDotProdEngine`, we can reuse activations and output gradients already computed during backprop. Instead of directly computing gradient similarity, we store these per-sample info to disks. Specifically, we can apply a Kronecker‑structured random projection $P = P_i \otimes P_o$. This can be elegantly implemented through a zero‑impact [LoRA‑style side branch](https://arxiv.org/pdf/2405.13954); no changes to model behavior.
  - Best for: computing pair-wise gradient similarities for a large dataset w.r.t. a fixed model checkpoint.  

Logic and when to use which
- Both engines exploit the same gradient structure to avoid instantiating full gradients and add minimal training overhead.
- Use GradDotProd when you need on‑the‑fly similarities within a step (e.g., *online data selection or reweighting, curriculum learning, auditing training dynamics*).
- Use GradProjLoRA when you need reusable per‑sample representations across many batches or the whole corpus (e.g., *offline data selection, clustering, etc*). These projections preserve inner products up to JL distortion. 


## Installation
This project uses [uv](https://docs.astral.sh/uv/) for environment management.
```bash
# Install uv if you don't have it (see https://docs.astral.sh/uv/ for options):
curl -LsSf https://astral.sh/uv/install.sh | sh

source init.sh   # runs 'uv sync' and activates this checkout's .venv
```


## Quick Start

In `examples/minimal/`, we provide three minimal examples for demonstrating core usage of GhostEngines:

- **`ghost_mlp.py`**: Basic GradDotProd usage for MLP models
  - Trains for 10 steps on synthetic data
  - Prints per-parameter gradient dot-products

- **`ghost_gradproj_mlp.py`**: Per-sample gradient projection computation and storage for MLP

- **`ghost_gradproj_lm.py`**: Per-sample gradient projection computation and storage for language models
  - Projects gradients for transformer layers
  - Demonstrates similarity computation from saved projections

### LLM pretraining with TorchTitan
TorchTitan (https://github.com/pytorch/torchtitan) is a PyTorch-native training stack for large-scale model development and experimentation.

Run the TorchTitan GradDotProd integration with the Llama 3 130M ghost config:

```bash
CONFIG_FILE="./examples/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" ./examples/torchtitan/run_train_with_ghost.sh
```

#### Dot-product levers and the speed ↔ memory tradeoff

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
single H200, one session; throughput is tokens/s, loss-identical in every row
(`docs/analysis/ac_frontier_2026-06-21.md`):

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
the fp32 logits tensor floors it). See the analysis doc for the full frontier and the negative
results.

Caveats: the combined `train + val` batch and the compiled path both raise peak memory, so at the
*max speed* (no-AC) setting a ghost run hits the memory ceiling **earlier** than a same-train-batch
baseline (the fp32 logits tensor, `batch · seq · vocab · 4` bytes, dominates for large-vocab
models). The default op-SAC `mme1` keeps peak memory *below* the eager engine, so it has the most
headroom — prefer it (or `mode=full`) on an 80 GB A100. To fall back to the pure eager engine, run
with `--ghost.no-decoupled_fn --ghost.no-batched_dotprod` (compile is auto-disabled on the eager
path).

### Standalone language-model examples

Two standalone LM examples live under `examples/lm/graddotprod_lm/` (online gradient
dot-products during training) and `examples/lm/gradproj_lm/` (offline per-sample
gradient projection to disk). Both ship with a built-in **synthetic data mode**
(random tokens, no tokenized corpus required), which makes them a quick smoke
test on a tiny model:

```bash
# GradDotProd: online dot-products (tiny model, random tokens) — needs a GPU
python examples/lm/graddotprod_lm/main.py --method GradDotProd \
    --train_set synthetic --architecture GPT2-Tiny \
    --batch_size 8 --val_batch_size 4 --max_steps 12 \
    --model_dtype float32 --train_dtype float32

# GradProj: offline projections (tiny model, random tokens) — runs on CPU too
python examples/lm/gradproj_lm/main.py --data_source synthetic \
    --architecture GPT2-Tiny --device cuda --batch_size 4 --max_samples 8 \
    --proj_dtype float32 --model_dtype float32 --train_dtype float32 \
    --output_dir ./outputs_smoke
```

To train on real data, tokenize the Pile (see each example's `README.md`) and
pass `--train_set pile` / `--data_source pile`. The example READMEs document the
full set of options.


## How the Ghost Engines Work

### GradDotProd Engine
1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered separately and stored in `.grad` before optimizer step. 

### GradProj Engine
- Uses LoRA-style low-rank projection matrices
- Projects high-dimensional gradients to lower-dimensional space
- Enables efficient per-sample gradient storage without materializing full gradients
- Supports both MLP and attention layer projections

See individual example directories for detailed documentation and configuration options.



## Integrating Ghost Engine with Your Training Loop

The `GhostEngineManager` provides a convenient interface for integrating gradient computation engines into your training loop. This is the **generic eager API** for custom loops; the TorchTitan integration above wraps the same engines but adds the compiled fast path (`decoupled_fn` + `torch.compile`), which bypasses the `saved_tensors_context()` hook shown here. Here's an overview of how to modify your training loop:

```python
from ghostEngines import GhostEngineManager

# 1. Initialize the Ghost Engine Manager
ghost_engine = GhostEngineManager(
    config=config,                    # Your training configuration
    model=model,                      # PyTorch model
    optimizer=optimizer,              # Model optimizer
    ddp_info={"master_process": is_master},  # Distributed info for logging/saving
    val_data=(X_val, Y_val),          # Validation data (required for GradDotProd)
)

# 2. Training loop with Ghost Engine integration
for iteration in range(max_steps):
    # Get training batch
    X_train, Y_train, batch_idx = get_batch()

    optimizer.zero_grad(set_to_none=True)

    # Attach batch information to engine
    ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)

    # Prepare input (concatenates val data for GradDotProd method)
    X_forward, Y_forward = ghost_engine.prepare_forward_input(X_train, Y_train)
    
    # Forward and backward pass (capture saved tensors for GradDotProd)
    with ghost_engine.saved_tensors_context():
        outputs = model(input_ids=X_forward, labels=Y_forward)
        loss = outputs.loss
        loss.backward()

    # Ghost engine gradient processing
    ghost_engine.prepare_gradients()    # Move accumulated gradients to .grad

    # Optimizer step
    optimizer.step()

    # Ghost engine post-processing
    ghost_engine.aggregate_and_log()    # Compute and log gradient metrics
    ghost_engine.clear_gradients()      # Clean up stored gradients
    
    # Periodic metric saving
    if ghost_engine.should_save_metrics(iteration):
        ghost_engine.save_metrics(iteration)
```

Notes:
- `saved_tensors_context()` is required for `GradDotProd` and is a no-op for other methods.
- With gradient accumulation, call `aggregate_and_log()` after each microbatch and move `prepare_gradients()`/`optimizer.step()` to the end of the accumulation window.
- For no-grad evaluation, use `ghost_engine.detach_for_evaluation()` and `ghost_engine.reattach_after_evaluation()`.


## Citation

```bibtex
@article{wang2024data,
  title={Data shapley in one training run},
  author={Wang, Jiachen T and Mittal, Prateek and Song, Dawn and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2406.11011},
  year={2024}
}
```
