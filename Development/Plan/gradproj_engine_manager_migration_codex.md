# GradProj → GhostEngineManager Migration (Codex Plan)

## Goals
- Unify all GradProjection usage under `GhostEngineManager` with strict, explicit configuration.
- Remove direct `GradProjLoraEngine` instantiation from examples.
- Preserve GradDotProd behavior; no regressions.

## Scope
- Code: `ghostEngines/engine_manager.py`, new `ghostEngines/utils.py` helper, example scripts in `Examples/`.
- Tests: new `Test/test_engine_manager_gradproj.py` focused unit tests.
- Docs: update relevant READMEs and example snippets.

## Preconditions
- Adopt “no fallback anywhere”: missing/ambiguous config must raise errors; never silently default.
- Use batch size 2 for any eval/integration sanity checks.

## Step‑By‑Step Plan

1) Decide and Enforce Directory Semantics
- Rule: Exactly one of `config.result_dir` or `config.proj_dir` must be provided for GradProj usage via the manager.
- Behavior:
  - If `proj_dir` given: use it as-is for saving projections.
  - Else if `result_dir` given: use `os.path.join(result_dir, "projections")`.
  - If both or neither are provided: raise `ValueError`.
- Deliverables: documented rule in code comments and README snippets.

2) Update `GhostEngineManager` for Strict GradProj Support
- Unknown method: raise `ValueError` instead of warning/continuing.
- Parameter validation (GradProj required): `proj_layers`, `proj_rank_total`, `proj_rank_min`, `proj_seed`, `proj_dtype`, and either `proj_dir` or `result_dir` per rule above.
- Pass‑through options: include `proj_row_orthonormal`, `include_embeddings`, `include_conv2d` if present.
- Attach/Reattach:
  - On init: call `engine.attach()` (no optimizer) for GradProj.
  - For `reattach_after_evaluation()`: if engine exposes `attach_with_optimizer`, call it; else call `attach()` with no args for GradProj to avoid TypeError.
- Metrics saving: gate `should_save_metrics()` and `save_metrics()` to GradDotProd only (GradProj saves from `collect_batch()`).
- Acceptance: unit tests pass; no duplicate `.../projections/projections` paths; no TypeError on reattach.

3) Add Helper `ghostEngines/utils.py`
- Function: `create_gradproj_manager(model, engine_config, optimizer=None, ddp_info=None)`
  - Builds a minimal config object with attributes required by the manager.
  - Validates required fields strictly (no defaults). Enforce directory rule above.
  - Requires explicit `ddp_info` (no implicit defaults) with at least `{'master_process': bool, 'ddp': bool, 'device': <device or str>}`.
  - Returns an instance of `GhostEngineManager`.
- Rationale: centralizes mapping from existing example dicts to manager config without duplicating ad‑hoc classes.

4) Migrate Examples to Manager
- `Examples/ghost_gradproj_mlp.py`:
  - Replace `from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine` with `from ghostEngines import GhostEngineManager` (or helper if used).
  - Build `engine_config` dict (same keys as before); set `method='GradProjLora'` and provide `proj_dir` (or `result_dir` per the rule).
  - Create manager with `optimizer=None` and `ddp_info={'master_process': True, 'ddp': False, 'device': device}`.
  - Where direct `engine.collect_batch()` is needed, use `manager.engine.collect_batch(...)`.
  - Use `manager.cleanup()` instead of `engine.detach()` at end.
- `Examples/ghost_gradproj_lm.py`:
  - Same migration pattern as MLP; ensure large model uses manager; set method; update calls.
- `Examples/GradProj_LM/main.py`:
  - Set `config.method = 'GradProjLora'`.
  - Provide `proj_dir` (preferred) to avoid ambiguity.
  - Replace direct engine creation with manager; update projection loop to use `manager.engine` where needed.
- Acceptance: example scripts run their minimal flows and write projections to the expected directory.

5) Tests: `Test/test_engine_manager_gradproj.py`
- Initialization:
  - Manager raises on unknown `method`.
  - Manager raises when both/neither of `proj_dir` and `result_dir` provided.
  - Manager constructs `GradProjLoraEngine` with correct pass‑through (`proj_row_orthonormal=True` → engine.proj_method == 'orthonormal').
- Attach/Reattach:
  - `detach_for_evaluation()` + `reattach_after_evaluation()` does not pass optimizer to GradProj and does not raise.
- Path Semantics:
  - When only `result_dir` provided, projections saved under `<result_dir>/projections`.
  - When `proj_dir` provided, no extra `projections/` suffix is added.
- Forward Prep:
  - `prepare_forward_input()` returns original `(X, Y)` for GradProj.

6) Documentation Updates
- Update examples’ READMEs to show manager‑based usage for GradProj.
- Add a short note in `CLAUDE.md` or central README about strict configuration for `GhostEngineManager` (no silent defaults).
- Mention batch size guidance (`--batch_size 2`) for memory.

7) Validation & Non‑Regression
- Run a tiny synthetic pass on MLP example (CPU acceptable) to assert one projection file is produced and metadata created.
- Run GradDotProd quick smoke per repo guidelines to ensure unchanged behavior:
  - `cd Examples/GradDotProd_LM && ./train.sh --batch_size 2`.

8) Rollback Plan
- If issues arise, examples can be temporarily pinned to direct `GradProjLoraEngine` usage behind a feature flag in the example files while fixes land. Keep new manager code in place.

## Definition of Done
- All example scripts use `GhostEngineManager` for GradProj flows.
- New tests pass locally; GradDotProd smoke still passes.
- No ambiguous config fallbacks; explicit errors for misconfiguration.
- Projections and metadata saved to correct, non‑duplicated paths.

## Open Questions
- Should `GhostEngineManager` accept a generic registry for future engines to avoid method branching? Out of scope but worth tracking.
- Do we want a forced flush mechanism for GradProj `save_metrics()` for symmetry? Optional; can be a follow‑up if buffering is introduced.
