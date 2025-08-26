# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Entry point; parses args and runs training.
- `config_file.py`: CLI defaults and paths (`RESULTS_DIR`, `PILE_DATA_DIR`, `LLAVA_DATASET_DIR`).
- `training_loop.py`, `training_utils.py`: Trainer, scheduling, evaluation, I/O.
- `model_setup.py`: Model/optimizer/scaler setup.
- `ghostEngines/`: Ghost engines (e.g., gradient dot-product) and manager.
- `data_processing/`: Dataset utilities (e.g., `tokenize_pile_by_domain.py`).
- `dataloader.py`, `llava_dataloader.py`: Batch utilities for Pile and LLaVA.
- `Scripts/`: SLURM launcher (`train.sh`).
- `Demo/`: Notebook and example scripts.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Tokenize Pile: `python data_processing/tokenize_pile_by_domain.py`
- Quick smoke eval (no training): `python main.py --eval_only --eval_iter 5 --eval_bs 8 --train_set pile`
- Train locally (example):
  `python main.py --method GradDotProd --architecture GPT2-Small --batch_size 16 --max_steps 1000`
- SLURM job: `./Scripts/train.sh --method GradDotProd --max_steps 50000`
Note: Adjust paths in `config_file.py` before running.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; wrap lines ~100 chars.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Add docstrings for public functions; prefer type hints for new/edited APIs.
- Keep imports ordered: stdlib, third-party, local.

## Testing Guidelines

The following are the command lines for testing the codebase. 

**Important: (Let's only use GPT2-Small trained on Pile now, don't need to worry about LLAVA)**
**Important: when running evaluation, always use --batch_size 2 due to our small GPU memory**
```bash
# Run training with default settings (GradDotProd method)
./Scripts/train.sh --batch_size 2

# Run regular training without gradient computation
./Scripts/train.sh --batch_size 2 --method Regular

# Specify custom parameters
./Scripts/train.sh --batch_size 2 --learning_rate 1e-4 --max_steps 100000
```

# Code Length and Structure Guidelines
- **Reuse code blocks whenever possible.** If similar functionality exists in previously generated files within this project, reference and extend that code rather than rewriting from scratch. Build incrementally on existing code patterns.
- **Do not fallback anywhere.** Raise errors and terminate program rather than silently falling back to default values. Always require explicit configuration values rather than silently using defaults. 

