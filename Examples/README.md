# Examples Directory

This directory contains example implementations demonstrating the use of the Ghost Engine framework for efficient gradient computation.

## Directory Structure

```
Examples/
├── shared/                # Shared utilities and data processing
│   ├── dataloader.py     # Dataset loading utilities
│   ├── model_setup.py    # Model initialization
│   ├── training_utils.py # Training helper functions
│   ├── utils.py          # General utilities
│   ├── domain_list.py    # Domain definitions
│   └── data_processing/  # Data preprocessing scripts
├── GradDotProd_LM/        # GradDotProd language model training
│   ├── main.py           # Training entry point
│   ├── config_file.py    # Configuration management
│   ├── training_loop.py  # Main training loop
│   └── train.sh          # Launch script
├── GradProj_LM/           # Gradient projection for language models
│   ├── main.py           # Projection computation entry
│   ├── config_file.py    # Projection configuration
│   ├── gradproj_loop.py  # Projection computation loop
│   └── train.sh          # Launch script
└── Minimal MLP Examples
    ├── ghost_mlp.py       # Basic GradDotProd for MLP
    ├── ghost_gradproj_mlp.py  # Gradient projection for MLP
    └── ghost_gradproj_lm.py   # Gradient projection for LM
```

## Available Examples

### 1. GradDotProd Language Model (`GradDotProd_LM/`)
Full implementation of gradient dot product computation during GPT-2 training on the Pile dataset. This example demonstrates how to efficiently compute gradient similarities between validation loss and training samples in a single backpropagation pass.

**Features:**
- Efficient gradient dot product computation
- Support for GPT-2 Small/Medium/Large
- Distributed training support
- Automatic metric logging and saving

**Quick Start:**
```bash
cd GradDotProd_LM
./train.sh --batch_size 2 --method GradDotProd  # With gradient computation
./train.sh --batch_size 2 --method Regular      # Standard training
```

### 2. Gradient Projection Language Model (`GradProj_LM/`)
Implementation of gradient projection using LoRA-based projection matrices for GPT-2 models. This example shows how to compute low-dimensional gradient projections efficiently.

**Features:**
- LoRA-based gradient projection
- Configurable projection rank and layers
- Support for MLP and attention layers
- Batch processing with memory optimization

**Quick Start:**
```bash
cd GradProj_LM
./train.sh --batch_size 2 --max_samples 1000
```

### 3. Minimal MLP Examples
Simplified implementations demonstrating core concepts:

- **`ghost_mlp.py`**: Basic GradDotProd implementation for MLP models
  - Trains for 10 steps on synthetic data
  - Prints per-parameter gradient dot-products
  - Demonstrates core engine usage pattern

- **`ghost_gradproj_mlp.py`**: Gradient projection for MLP with visualization
  - Three execution modes: project, non_interf, naive_check
  - Shows efficient per-sample gradient storage
  - Includes comparison with naive computation

- **`ghost_gradproj_lm.py`**: Language model gradient projection example
  - Projects gradients for transformer layers
  - Demonstrates similarity computation from saved projections
  - Integrates with transformers support utilities

**Run minimal examples:**
```bash
python ghost_mlp.py
python ghost_gradproj_mlp.py --mode project --proj_rank_total 64
python ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc"
```

## Shared Utilities

The `shared/` directory contains common components used across examples:

- **Data Processing**: Tokenization and dataset preparation scripts
- **Model Setup**: Unified model initialization for different architectures
- **Training Utilities**: Common training loop components and helpers
- **Dataloaders**: Efficient data loading for Pile and other datasets

## Getting Started

1. **Prepare Data**: First tokenize your dataset using the shared processing scripts
   ```bash
   python shared/data_processing/tokenize_pile_by_domain.py
   ```

2. **Choose Example**: Select the appropriate example based on your use case

3. **Configure**: Adjust settings in the example's `config_file.py`

4. **Run**: Execute the `train.sh` script with desired parameters

## How the Ghost Engines Work

### GradDotProd Engine
- Concatenates train and validation batches for single forward/backward pass
- Computes per-parameter gradient dot-products between validation and training samples
- Accumulates training gradients separately, then moves to `.grad` before optimizer step
- Aggregates per-parameter dot-products into single vectors for efficient storage

### GradProj Engine
- Uses LoRA-style low-rank projection matrices
- Projects high-dimensional gradients to lower-dimensional space
- Enables efficient per-sample gradient storage without materializing full gradients
- Supports both MLP and attention layer projections

See individual example directories for detailed documentation and configuration options.