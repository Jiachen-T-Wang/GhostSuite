# GradDotProd Language Model Training

This example demonstrates efficient gradient dot product computation during language model pretraining on the Pile dataset using the GradDotProd ghost engine.

## Overview

The GradDotProd engine enables computation of gradient similarities between validation loss and individual training samples in a single backpropagation pass, eliminating the need to materialize any model-sized vectors. At a high-level, this is achieved by cleverly exploiting information already computed during standard backpropagation. For technical details, please refer to Section 4.2 in [Data Shapley in One Training Run](https://openreview.net/pdf?id=HD6bWcj87Y).

## Quick Start

### Get Tokenized Dataset
Process the Pile dataset by domain:
```bash
python Examples/shared/data_processing/tokenize_pile_by_domain.py
```
*Note: This process can take ~24 hours depending on your system. For a minimal example, see `Examples/ghost_mlp.py`.* 

### Usage

```bash
cd Examples/GradDotProd_LM

# Run with gradient dot product computation (default)
./train.sh --batch_size 16

# Run standard training without gradient computation
./train.sh --batch_size 16 --method Regular

# Custom training parameters
./train.sh --batch_size 16 --learning_rate 1e-4 --max_steps 100000
```

### Key Parameters

- `--method`: Training method (`GradDotProd` or `Regular`)
- `--architecture`: Model architecture (default to `GPT2-Small`)
- `--batch_size`: Training batch size
- `--val_batch_size`: Validation batch size for gradient computation
- `--dot_prod_save_interval`: How often to save gradient metrics

### Key Configurations

Edit `config_file.py` to adjust:
- `RESULTS_DIR`: Where training results and metrics are saved
- `PILE_DATA_DIR`: Path to tokenized Pile dataset

## How It Works

1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered seperately and stored in `.grad` before optimizer step. 


## Training Loop Integration

The training loop (`training_loop.py`) integrates the ghost engine via hooks:

```python
# Attach batch information
ghost_engine.attach_train_batch(X_train, Y_train, iteration, batch_idx)

# Prepare concatenated input
X_forward, Y_forward = ghost_engine.prepare_forward_input(X_train, Y_train)

# Forward/backward pass
loss = model(X_forward, Y_forward).loss
loss.backward()

# Process gradients
ghost_engine.prepare_gradients()
optimizer.step()

# Aggregate and save metrics
ghost_engine.aggregate_and_log()
```