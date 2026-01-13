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
```bash
pip install -r requirements.txt
```


## Quick Start

In `examples/`, we provide three minimal examples for demonstrating core usage of GhostEngines:

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

The language-model examples under `examples/GradDotProd_LM/` and `examples/GradProj_LM/` are deprecated in v0.33 and will be fixed soon. If you need to run them, please use v0.2: https://github.com/Jiachen-T-Wang/GhostSuite/tree/v0.2


## How the Ghost Engines Work

### GradDotProd Engine
1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered seperately and stored in `.grad` before optimizer step. 

### GradProj Engine
- Uses LoRA-style low-rank projection matrices
- Projects high-dimensional gradients to lower-dimensional space
- Enables efficient per-sample gradient storage without materializing full gradients
- Supports both MLP and attention layer projections

See individual example directories for detailed documentation and configuration options.



## Integrating Ghost Engine with Your Training Loop

The `GhostEngineManager` provides a convenient interface for integrating gradient computation engines into your training loop. Here's an overview of how to modify your training loop:

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
