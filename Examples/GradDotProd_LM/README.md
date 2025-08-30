# GradDotProd Language Model Training

This example demonstrates efficient gradient dot product computation during GPT-2 training on the Pile dataset using the GradDotProd ghost engine.

## Overview

The GradDotProd engine enables computation of gradient similarities between validation loss and individual training samples in a single backpropagation pass, eliminating the need to materialize per-sample gradients. This is achieved by cleverly exploiting information already computed during standard backpropagation.

## Features

- **Efficient Gradient Computation**: Computes gradient dot products without storing full per-sample gradients
- **Model Support**: GPT-2 Small, Medium, and Large architectures
- **Distributed Training**: Full support for multi-GPU training via DDP
- **Automatic Logging**: Gradient metrics saved at configurable intervals
- **Memory Optimized**: Processes large batches efficiently

## Quick Start

### Basic Usage

```bash
# Run with gradient dot product computation (default)
./train.sh --batch_size 2

# Run standard training without gradient computation
./train.sh --batch_size 2 --method Regular

# Custom training parameters
./train.sh --batch_size 2 --learning_rate 1e-4 --max_steps 100000
```

### Key Parameters

- `--method`: Training method (`GradDotProd` or `Regular`)
- `--architecture`: Model architecture (`GPT2-Small`, `GPT2-Medium`, `GPT2-Large`)
- `--batch_size`: Training batch size
- `--val_batch_size`: Validation batch size for gradient computation
- `--max_steps`: Maximum training steps
- `--learning_rate`: Learning rate
- `--dot_prod_save_interval`: How often to save gradient metrics

## Configuration

Edit `config_file.py` to adjust:
- `RESULTS_DIR`: Where training results and metrics are saved
- `PILE_DATA_DIR`: Path to tokenized Pile dataset
- Model precision settings (`model_dtype`, `train_dtype`)

## How It Works

1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples
   - Accumulated training gradients stored separately
3. **Gradient Processing**: Training gradients moved to `.grad` before optimizer step
4. **Metric Aggregation**: Per-parameter dot products aggregated and saved periodically

## Output Structure

```
Results/
└── [experiment_name]/
    ├── grad_dotprods/      # Gradient dot product metrics
    │   ├── iter_000010.pkl
    │   ├── iter_000020.pkl
    │   └── ...
    ├── training_log.json   # Training metrics
    └── config.json         # Experiment configuration
```

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

## Memory Considerations

- Use `--batch_size 2` for GPUs with limited memory
- The engine processes validation batch alongside training batch
- Gradient metrics are cleared after aggregation to save memory

## Advanced Usage

### Distributed Training

```bash
# Multi-GPU training (automatically detected)
./train.sh --batch_size 4
```

### Evaluation Only

```bash
# Run evaluation without training
./train.sh --eval_only
```

## Troubleshooting

- **OOM Errors**: Reduce batch size or use smaller model architecture
- **Slow Training**: Adjust `dot_prod_save_interval` to save metrics less frequently
- **Import Errors**: Ensure you're running from the `GradDotProd_LM/` directory

## Citation

If you use this code, please cite:

```bibtex
@article{wang2024data,
  title={Data shapley in one training run},
  author={Wang, Jiachen T and Mittal, Prateek and Song, Dawn and Jia, Ruoxi},
  journal={arXiv preprint arXiv:2406.11011},
  year={2024}
}
```