**Codebase Reorganization Request: Consolidating Training Examples**

**Current Structure:**
- The main workspace contains a language model training example demonstrating the GradDotProd engine
- `Examples/` directory contains three minimal examples:
  - `ghost_mlp.py` - demonstrates GradDotProd engine usage
  - `ghost_gradproj_mlp.py` - demonstrates GradProjLoraEngine usage
  - `ghost_gradproj_lm.py` - demonstrates GradProjLoraEngine usage on language model. 
- `Examples/GradProj_GPT2/` contains a complete GPT2 implementation using GradProjLoraEngine

**Objective:**
Refactor the codebase to achieve a cleaner structure where:
1. The main workspace contains no `.py` files
2. All model training code is consolidated under `Examples/`
3. The current main workspace language model example is moved to `Examples/GradDotProd_GPT2/`

**Key Consideration:**
There are shared dependencies between `Examples/GradProj_GPT2/` and the main workspace (e.g., `dataloader.py`), which need to be handled carefully during reorganization.

**Request:**
Please provide a detailed plan for this codebase reorganization that addresses:
- How to handle shared code components
- Directory structure recommendations
- Step-by-step migration approach
- Any potential risks or dependencies to consider

---

## Proposed Reorganization Plan

### Goals & Scope
- Remove all top-level `.py` files from the repo root.
- Consolidate all runnable training examples under `Examples/` with a consistent layout.
- Factor shared utilities into a single importable library package to avoid duplication.
- Move the current root language model example to `Examples/GradDotProd_GPT2/`.
- Avoid silent fallbacks; add explicit errors where necessary during transition.

### Target Directory Structure (post-reorg)
```
.
├── src/
│   └── ghostsuite/
│       ├── __init__.py
│       ├── config/               # config and paths
│       │   └── config_file.py
│       ├── data/                 # data processing utilities
│       │   ├── __init__.py
│       │   └── tokenize_pile_by_domain.py
│       ├── dataloaders/
│       │   ├── __init__.py
│       │   ├── dataloader.py
│       │   └── llava_dataloader.py   # kept but not prioritized for testing
│       ├── engines/
│       │   ├── __init__.py
│       │   └── ghostEngines/     # existing engines moved under package
│       ├── training/
│       │   ├── __init__.py
│       │   ├── training_loop.py
│       │   └── training_utils.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── model_setup.py
│       └── utils.py
├── Examples/
│   ├── GradDotProd_GPT2/
│   │   ├── train.py              # replaces root example entry
│   │   ├── README.md
│   │   └── configs/
│   │       └── gpt2_small_pile.yaml (or .json)
│   ├── GradProj_GPT2/            # unchanged layout; update imports
│   ├── ghost_mlp/
│   │   └── train.py              # from ghost_mlp.py
│   ├── ghost_gradproj_mlp/
│   │   └── train.py              # from ghost_gradproj_mlp.py
│   └── ghost_gradproj_lm/
│       └── train.py              # from ghost_gradproj_lm.py
├── Scripts/
│   └── train.sh                  # updated to call example entrypoints
├── Demo/
│   └── (notebooks use `import ghostsuite`)
├── Development/
│   └── Plan/
│       └── code_reorg.md
├── Test/
│   └── (unit tests import ghostsuite)
├── pyproject.toml                # editable install for `ghostsuite`
├── requirements.txt
└── README.md
```

Notes:
- The `src/` layout enables `pip install -e .` for clean imports (`from ghostsuite...`).
- `ghostEngines/` content is nested under `src/ghostsuite/engines/ghostEngines` to preserve current module names while allowing future cleanup.
- LLaVA parts remain but are deprioritized per testing guidance.

### Handling Shared Code Components
- Promote shared modules (`dataloader.py`, `training_loop.py`, `training_utils.py`, `model_setup.py`, `config_file.py`, `utils.py`, and `data_processing/*`) into `src/ghostsuite` as shown above.
- Convert all Examples to import from the package (e.g., `from ghostsuite.training import training_loop`).
- Eliminate any code duplication by centralizing logic in `ghostsuite` and keeping Examples thin (config + entrypoint only).
- For any legacy root-level imports during transition, provide stub modules that immediately raise a descriptive `ImportError` with the new path (no silent fallback).

### Step-by-Step Migration Approach
1) Inventory and Dependency Map
   - Use repo-wide search to list all imports touching `main.py`, `training_*`, `dataloader.py`, `ghostEngines/`.
   - Document any relative imports that will break when files move.

2) Create Library Package Skeleton
   - Add `src/ghostsuite/` with `__init__.py` and subpackages as in the target structure.
   - Move shared modules into the package, preserving filenames initially to minimize churn.
   - Add `pyproject.toml` for editable install (`pip install -e .`).

3) Update Imports Internally
   - Within moved modules, convert relative or root-level imports to package-style imports (e.g., `from ghostsuite.training import training_utils`).
   - Ensure no circular imports; if present, split utilities by concern.

4) Normalize Examples Layout
   - Create `Examples/GradDotProd_GPT2/train.py` that wires CLI/args to package calls.
   - Convert `Examples/*` single-file scripts into directories with `train.py` and a minimal `README.md` describing usage.
   - Move the current root language model example logic from `main.py` into `Examples/GradDotProd_GPT2/train.py`.

5) Update Scripts and Docs
   - Modify `Scripts/train.sh` to call `python Examples/GradDotProd_GPT2/train.py` (or accept `--example` to target different subfolders).
   - Update `README.md` with new quickstart commands and `pip install -e .` step.

6) Add Transitional Guardrails (No Fallbacks)
   - Replace root-level `main.py` and other formerly top-level entry modules with 5–10 line stubs that raise `ImportError` and point to the new path. Remove these stubs only after one release.

7) Remove Top-Level `.py`
   - After Examples and Scripts are verified, delete root `.py` files so the root has no Python files.

8) Validation & Testing
   - Run smoke eval and training with the new example entry:
     - `./Scripts/train.sh --batch_size 2` (GradDotProd default)
     - `./Scripts/train.sh --batch_size 2 --method Regular`
   - Confirm Pile tokenization pathing still works (`python -m ghostsuite.data.tokenize_pile_by_domain ...`).
   - Ensure `RESULTS_DIR`, `PILE_DATA_DIR`, and `LLAVA_DATASET_DIR` resolve correctly after package move (no hard-coded root-relative paths).

9) Documentation & Examples Consistency
   - Each example folder includes: `README.md`, minimal config file, and exact command lines matching Testing Guidelines (batch size 2 for eval/training examples).
   - Add an import map in `README.md` documenting old→new import paths.

### Risks & Mitigations
- Import breakage: Mitigate with exhaustive search/replace and stubs that raise explicit errors with migration hints.
- Path assumptions: Audit `config_file.py` and any `os.path` usage for root-relative assumptions; switch to env vars or explicit args. Reject missing config with clear errors.
- Packaging pitfalls: Use `src/` layout with `pyproject.toml` to avoid namespace confusion and ensure `pip install -e .` works.
- CI/user scripts drift: Update `Scripts/train.sh` and documentation together; add a quick `--example` flag to keep workflows simple.
- LLaVA side-effects: Keep `llava_dataloader.py` packaged but do not block GPT2-Small validation on it.

### Acceptance Criteria
- No `.py` files in repo root.
- All training runs start from `Examples/*/train.py` via `Scripts/train.sh`.
- Examples import only from `ghostsuite` package; no direct cross-example imports.
- Smoke eval and training commands succeed with `--batch_size 2` on GPT2-Small (Pile).
- Documentation updated to reflect new structure with no silent fallbacks.

### Rollback Plan
- Keep a branch/tag before the move.
- Transitional stubs enable quick redirection or rollback if any downstream scripts rely on old paths.

### Open Questions
- Should we introduce a unified CLI (e.g., `python -m ghostsuite.cli train --example grad_dotprod_gpt2`), or keep per-example entrypoints only?
- Preferred config format for examples (`.yaml` vs `.json`) given current parser utilities?
