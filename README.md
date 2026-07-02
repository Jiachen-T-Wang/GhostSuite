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

- `GradProjLoraEngine`
  - Purpose: Offline, corpus‑scale analysis by storing low‑dimensional per‑sample gradient projections to disk for later similarity analysis.
  - Core idea: Similar to `GradDotProdEngine`, we can reuse activations and output gradients already computed during backprop. Instead of directly computing gradient similarity, we store these per-sample info to disks. Specifically, we can apply a Kronecker‑structured random projection $P = P_i \otimes P_o$. This can be elegantly implemented through a zero‑impact [LoRA‑style side branch](https://arxiv.org/pdf/2405.13954); no changes to model behavior.
  - Best for: computing pair-wise gradient similarities for a large dataset w.r.t. a fixed model checkpoint.  

Logic and when to use which
- Both engines exploit the same gradient structure to avoid instantiating full gradients and add minimal training overhead.
- Use GradDotProd when you need on‑the‑fly similarities within a step (e.g., *online data selection or reweighting, curriculum learning, auditing training dynamics*).
- Use GradProjLoRA when you need reusable per‑sample representations across many batches or the whole corpus (e.g., *offline data selection, clustering, etc*). These projections preserve inner products up to JL distortion. 


## Installation
Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
The lockfile pins CUDA 12.8 builds of `torch` on Python 3.12, so installation needs a Linux
machine with an NVIDIA GPU driver (many engine correctness checks can then run on CPU).
```bash
git clone https://github.com/Jiachen-T-Wang/GhostSuite.git
cd GhostSuite
uv sync
source .venv/bin/activate
```


## Quick Start

The `examples/` directory holds five groups of runnable demos, one per subfolder.
Each is summarized below; see [`examples/README.md`](examples/README.md) for the full
walkthrough, all options, and the TorchTitan speed↔memory tuning guide.

### Minimal examples (`examples/minimal/`)
Smallest end-to-end demos on synthetic data — no data prep, runs on CPU.

```bash
python examples/minimal/ghost_mlp.py                                       # GradDotProd on an MLP
python examples/minimal/ghost_gradproj_mlp.py --mode project --proj_rank_total 64
python examples/minimal/ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc"
```

### Standalone LM examples (`examples/lm/`)
Language-model training with the ghost engines: `graddotprod_lm/` (online train↔val
dot-products during training) and `gradproj_lm/` (offline per-sample projections to
disk). Both ship a synthetic data mode (random tokens, no corpus) for a quick smoke test:

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

To train on real data, tokenize the Pile and pass `--train_set pile` / `--data_source pile`
(see each example's `README.md`).

### LLM pretraining with TorchTitan (`examples/torchtitan/`)
End-to-end GradDotProd integration in [TorchTitan](https://github.com/pytorch/torchtitan),
PyTorch's native large-scale training stack. Run the Llama 3 130M ghost config:

```bash
CONFIG_FILE="./examples/torchtitan/torchtitan/models/llama3/train_configs/llama3_130m_ghost.toml" \
    ./examples/torchtitan/run_train_with_ghost.sh
```

Data prerequisite: set `C4_LOCAL_DIR` (training corpus) and `PILE_TEST_DIR` (ghost/validation
batches) to your local dataset directories before running.

The `--ghost.*` flags expose a speed↔memory tradeoff (a compiled fast path plus op-level
selective activation checkpointing). See [`examples/README.md`](examples/README.md#3-llm-pretraining-with-torchtitan-torchtitan)
for the levers, benchmarks, and tuning guidance.

### Online batch selection with GREATS (`examples/greats/`)
[GREATS](https://github.com/Jiachen-T-Wang/GREATS) (Wang et al., NeurIPS 2024) uses the ghost
dot-products to pick, every step, the candidate subset that best reduces the validation loss —
online data selection during pretraining (`greats/pretrain/`) and LoRA instruction tuning
(`greats/sft/`).

```bash
# Synthetic smoke test (1 GPU, no corpus)
python examples/greats/pretrain/main.py --method GREATS --train_set synthetic \
    --architecture GPT2-Tiny --candidate_batch_size 16 --batch_size 8 \
    --val_batch_size 4 --max_steps 12 --model_dtype float32 --train_dtype float32
```

See [`examples/greats/README.md`](examples/greats/README.md) for both variants and the
first-order vs. second-order selection modes.

### Data Value Embedding (`examples/dvemb_lm/`)
[Data Value Embedding](https://arxiv.org/abs/2412.09538) (DVEmb) attributes a trained model's
test-time behavior back to each training example *and the step it appeared at* (in-run / temporal
influence). It captures per-step per-sample projected gradients over the whole trajectory with the
`GradProjLoRA` engine, then unrolls the SGD dynamics into per-training-point value embeddings — a
4-stage pipeline.

```bash
# All four stages, synthetic smoke (CPU-friendly, GPT2-Tiny)
python examples/dvemb_lm/main.py --data_source synthetic --architecture GPT2-Tiny --device cpu \
    --optimizer sgd --max_steps 8 --batch_size 4 --n_test 8 --proj_rank_total 64 \
    --train_and_store_grad --compute_embedding --compute_value --attribute
```

On CUDA, capture defaults to a decoupled in-graph + `torch.compile` fast path at bf16 (~25% faster
than the eager engine, values unchanged). See [`examples/dvemb_lm/README.md`](examples/dvemb_lm/README.md).


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

The loop has a per-step lifecycle that also covers gradient accumulation: `begin_step()` once
before the microbatch loop, `collect_microbatch()` after each microbatch's backward (it reads that
microbatch's per-sample dots and folds its validation gradient), and a single `prepare_gradients()`
after the loop. The fixed validation batch rides in **every** microbatch's combined forward, so the
per-microbatch val gradients must be **summed** for one end-of-step subtract-val recovery — that is
what `collect_microbatch()` accumulates. Set `N = gradient_accumulation_steps` (use `N = 1` for a
plain single-microbatch step); when `N > 1`, each microstep must draw a **distinct** train sub-batch
and scale its loss by `1/N`.

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
N = config.gradient_accumulation_steps            # 1 for a plain single-microbatch step
for iteration in range(max_steps):
    optimizer.zero_grad(set_to_none=True)
    ghost_engine.begin_step()                     # reset per-step dot/grad accumulation

    for micro in range(N):
        X_train, Y_train, batch_idx = get_batch()  # DISTINCT sub-batch per microstep
        ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)
        # prepare_forward_input concatenates the val batch for GradDotProd (no-op otherwise)
        X_forward, Y_forward = ghost_engine.prepare_forward_input(X_train, Y_train)
        # saved_tensors_context captures per-sample activations for GradDotProd (no-op otherwise)
        with ghost_engine.saved_tensors_context():
            loss = model(input_ids=X_forward, labels=Y_forward).loss / N
            loss.backward()
        ghost_engine.collect_microbatch()          # this microstep's dots + fold its val grad

    ghost_engine.prepare_gradients()               # ONE subtract-val recovery from the summed val grad
    optimizer.step()
    ghost_engine.clear_gradients()                 # clean up stored gradients
    if ghost_engine.should_save_metrics(iteration):
        ghost_engine.save_metrics(iteration)
```

The per-microstep dots are pooled into a single `[N * batch_size]` score vector; read them with
`ghost_engine.read_scores()`. Note the dots carry a `1/N^2` loss-rescale factor — consistent within a
run (sign / ranking preserved), but not comparable in absolute scale across different `N`. For
selection workflows, the higher-level `examples/lm/shared/selection_trainer.online_selection_step`
driver wraps this whole lifecycle (scoring, selection, recovery) — pass it a `draw_microbatch`
callable and `grad_accum=N`.

Notes:
- `saved_tensors_context()` is required for `GradDotProd` and is a no-op for other methods.
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
