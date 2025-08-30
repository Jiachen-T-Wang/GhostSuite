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
- Quick smoke eval: `Examples/ghost_mlp.py`, `Examples/ghost_gradproj_mlp.py` and `Examples/ghost_gradproj_lm.py` provide minimal examples for using `GradDotProdEngine` and `GradProjLoraEngine`. 


## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; wrap lines ~100 chars.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Add docstrings for public functions; prefer type hints for new/edited APIs.
- Keep imports ordered: stdlib, third-party, local.


## Testing Guidelines

The following are the command lines for testing the codebase. 

**Important: (Let's only use GPT2-Small trained on Pile now, don't need to worry about LLAVA)**
**Important: when running evaluation, always use --batch_size 2 due to our small GPU memory**

#### GradDotProd Language Model Training
```bash
cd Examples/GradDotProd_LM
./train.sh --batch_size 2  # Run with default GradDotProd method
./train.sh --batch_size 2 --method Regular  # Run without gradient computation
./train.sh --batch_size 2 --learning_rate 1e-4 --max_steps 100000  # Custom parameters
```

#### Gradient Projection Language Model
```bash
cd Examples/GradProj_LM
./train.sh  # Run gradient projection computation
```

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
   - Create minimal, isolated test cases for specific functions or modules in `Test/`. 
   - Focus on unit tests that validate individual pieces of functionality
   - Avoid running the full application unless explicitly necessary

3. **Documentation Phase**
   - Summarize findings in a markdown file within `Development/CodeReview/`
   - Use descriptive filenames that indicate the review scope (e.g., `dataloader-review-2025-08-27.md`)
   - **Important**: Never overwrite existing review documents; always create new files with unique names
