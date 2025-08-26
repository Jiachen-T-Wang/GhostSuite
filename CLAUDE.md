# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Training (Let's only use GPT2-Small trained on Pile now, don't need to worry about LLAVA)
**Important: when running evaluation, always use --batch_size 2 due to our small GPU memory**
```bash
# Run training with default settings (GradDotProd method)
./Scripts/train.sh --batch_size 2

# Run regular training without gradient computation
./Scripts/train.sh --batch_size 2 --method Regular

# Specify custom parameters
./Scripts/train.sh --batch_size 2 --learning_rate 1e-4 --max_steps 100000
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize environment
./init.sh
```

## Architecture Overview

### Ghost Engine System
The core innovation is the "ghost" engine framework for efficient gradient computation:

- **GhostEngineManager** (`ghostEngines/engine_manager.py`): Unified interface that abstracts method-specific initialization. Automatically configures engines based on training method.

- **GradDotProdEngine** (`ghostEngines/graddotprod_engine.py`): Computes gradient dot products between validation loss and training samples in a single backpropagation pass by exploiting already-computed gradient information.

- **Integration Pattern**: Ghost engines attach to the model and optimizer, intercepting gradient computation without modifying the core training loop.

### Training Pipeline

1. **Configuration** (`config_file.py`): 
   - Manages all training parameters via `TrainingConfig` class
   - Key paths: `RESULTS_DIR` for outputs, `PILE_DATA_DIR` for tokenized data
   - Supports multiple model architectures (GPT2, Pythia, LLaVA)

2. **Main Entry** (`main.py`):
   - Orchestrates setup: distributed training, data loading, model initialization
   - Creates `Trainer` instance with configured ghost engine

3. **Training Loop** (`training_loop.py`):
   - Integrates ghost engine via simple hooks:
     - `attach_train_batch()`: Register current batch
     - `prepare_forward_input()`: Concatenate validation data if needed  
     - `prepare_gradients()`: Move accumulated gradients to `.grad`
     - `aggregate_and_log()`: Compute and save gradient metrics

4. **Model Setup** (`model_setup.py`):
   - Handles model initialization for different architectures
   - Configures precision (bfloat16/float32) and distributed training

### Data Processing

- **Dataloader** (`dataloader.py`, `llava_dataloader.py`): Handles tokenized datasets with domain-specific sampling
- **Processing Scripts** (`data_processing/`): Convert raw datasets to tokenized format organized by domain

### Key Configuration Variables

- `config.method`: "GradDotProd" or "Regular" - determines if ghost engine is used
- `config.dot_prod_save_interval`: How often to save gradient metrics (default: 10 iterations)
- `config.result_dir`: Where gradient dot products and training metrics are saved
- `config.model_dtype` / `config.train_dtype`: Precision settings (typically bfloat16)

## Important Notes

- Ghost engines require validation data (`X_val`, `Y_val`) for gradient computation
- Gradient metrics are saved to `{result_dir}/grad_dotprods/` directory
- The system supports distributed training via DDP


# Code Length and Structure Guidelines
- **Reuse code blocks whenever possible.** If similar functionality exists in previously generated files within this project, reference and extend that code rather than rewriting from scratch. Build incrementally on existing code patterns.
- **Do not fallback anywhere.** Raise errors and terminate program rather than silently falling back to default values. Always require explicit configuration values rather than silently using defaults. 