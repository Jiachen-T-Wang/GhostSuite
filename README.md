# "Ghost" Suites for Fast Gradient Information Calculation

## Introduction
Computing per-sample gradient information and pair-wise gradient similarity is often the computational bottleneck for data-centric research (e.g., data selection, synthetic data generation). A naive approach would require setting the batch size to 1, backpropagating on the loss of each training sample, and storing all the huge gradient vectors. Consequently, this approach would be computationally prohibitive for practical applications. 

In [Data Shapley in One Training Run](https://openreview.net/pdf?id=HD6bWcj87Y) (ICLR'25 Outstanding Paper Runner-up), we proposed a highly efficient method to obtain per-sample gradient information. It turns out that we can compute the gradient dot-product between every pair of data points within a large batch in just a single backpropagation. At high level, the technique exploits information that's already being computed during standard backpropagation with respect to the aggregated loss on a batch of data points. 

This repository provides a clean implementation for "ghost"-based techniques for fast gradient information calculation during language model training. The "ghost" engines enable efficient computation of gradient dot-products between validation loss and individual training samples without fully materializing intermediate gradients.


## Installation
```bash
pip install -r requirements.txt
```

## Project Structure

```
GhostSuite/
├── ghostEngines/           # Core ghost engine library
│   ├── graddotprod_engine.py
│   ├── gradProjection/
│   └── engine_manager.py
├── Examples/               # Training examples and implementations
│   ├── shared/            # Shared utilities across examples
│   ├── GradDotProd_LM/    # GradDotProd language model training
│   ├── GradProj_LM/       # Gradient projection LM example
│   └── [MLP examples]     # Minimal MLP demonstrations
├── Test/                  # Unit tests
└── Results/               # Training outputs and metrics
```

## Quick Start

### 1. Get Tokenized Dataset
Process the Pile dataset by domain:
```bash
python Examples/shared/data_processing/tokenize_pile_by_domain.py
```
*Note: This process can take ~24 hours depending on your system.*

### 2. Run Training Examples

#### GradDotProd Language Model Training
```bash
cd Examples/GradDotProd_LM
./train.sh --batch_size 2  # Run with gradient dot product computation
./train.sh --batch_size 2 --method Regular  # Run without gradient computation
```

#### Gradient Projection Language Model
```bash
cd Examples/GradProj_LM
./train.sh --batch_size 2  # Compute gradient projections
```

During training, the engines automatically:
- Log gradient metrics at specified intervals
- Save results to the configured output directory


## Integrating Ghost Engine with Your Training Loop

The `GhostEngineManager` provides a convenient interface for integrating gradient computation engines into your training loop. Here's an overview of how to modify your training loop:

```python
from ghostEngines import GhostEngineManager

# 1. Initialize the Ghost Engine Manager
ghost_engine = GhostEngineManager(
    config=config,                    # Your training configuration
    model=model,                      # PyTorch model
    optimizer=optimizer,              # Model optimizer
    val_data=(X_val, Y_val)          # Validation data (required for GradDotProd)
)

# 2. Training loop with Ghost Engine integration
for iteration in range(max_steps):
    # Get training batch
    X_train, Y_train, batch_idx = get_batch()
    
    # Attach batch information to engine
    ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)
    
    # Prepare input (concatenates val data for GradDotProd method)
    X_forward, Y_forward = ghost_engine.prepare_forward_input(X_train, Y_train)
    
    # Forward and backward pass
    outputs = model(input_ids=X_forward, labels=Y_forward)
    loss = outputs.loss
    loss.backward()
    
    # Ghost engine gradient processing
    ghost_engine.prepare_gradients()    # Move accumulated gradients to .grad
    
    # Optional: gradient clipping can be added here
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Optimizer step
    optimizer.step()
    
    # Ghost engine post-processing
    ghost_engine.aggregate_and_log()    # Compute and log gradient metrics
    ghost_engine.clear_gradients()      # Clean up stored gradients
    
    # Standard cleanup
    optimizer.zero_grad(set_to_none=True)
    
    # Periodic metric saving
    if ghost_engine.should_save_metrics(iteration):
        ghost_engine.save_metrics(iteration)
```


## Citation

```bibtex
@article{wang2024data,
  title={Data shapley in one training run},
  author={Wang, Jiachen T and Mittal, Prateek and Song, Dawn and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2406.11011},
  year={2024}
}
```
