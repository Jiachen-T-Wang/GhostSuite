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



# Code Review and Testing Guidelines

### Core Principles
- **Review-Only Mode**: When conducting code reviews, analyze and provide feedback without modifying the existing code
- **Isolated Testing**: Write standalone, focused tests for individual components rather than executing the entire codebase
- **Documentation**: Document all findings systematically for future reference

### Review Process
1. **Analysis Phase**
   - Examine code structure, logic, and patterns
   - Identify potential issues, bugs, or areas for improvement
   - Assess code quality, readability, and maintainability
   - **Mathematical Verification**: If the code implements mathematical derivations or formulas, carefully verify that the implementation faithfully represents the mathematical concepts, including:
     - Correct formula translation
     - Proper handling of edge cases and numerical stability
     - Appropriate precision and rounding considerations
     - Accurate implementation of mathematical operations and their order

2. **Testing Phase**
   - Create minimal, isolated test cases for specific functions or modules
   - Focus on unit tests that validate individual pieces of functionality
   - Avoid running the full application unless explicitly necessary

3. **Documentation Phase**
   - Summarize findings in a markdown file within `Development/CodeReview/`
   - Use descriptive filenames that indicate the review scope (e.g., `dataloader-review-2025-08-27.md`)
   - **Important**: Never overwrite existing review documents; always create new files with unique names


