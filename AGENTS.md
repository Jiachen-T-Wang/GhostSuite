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
- No formal test suite yet. Use deterministic smoke runs with `--seed` and `--eval_only`.
- Validate data paths and result artifacts under `RESULTS_DIR` created by `utils.build_result_dir`.
- When adding tests, place under `tests/` and favor `pytest`-style unit tests mocking I/O.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (≤72 chars), descriptive body if needed.
- Reference issues using `Closes #123` when applicable.
- PRs must include: summary of changes, rationale, run command(s) used, sample logs/metrics path (e.g., `.../GradDotProd_pile_*`), and any config updates.
- For SLURM: include partition/account changes and expected GPU/precision (e.g., `bfloat16`).

## Security & Configuration Tips
- Review `Scripts/train.sh` SBATCH fields (email, partition, account, output paths) for your cluster.
- Ensure GPUs support BF16 if using `--model_dtype/--train_dtype bfloat16`.
- Large preprocessing can run for many hours; use screened/queued jobs and quota-aware `RESULTS_DIR`.

