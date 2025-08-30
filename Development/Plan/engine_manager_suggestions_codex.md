# GhostSuite Engine Interface Suggestions (Codex)

## Goals
- Provide a clean, stable user interface for all Ghost engines (current and future).
- Remove method-specific logic from the manager; avoid branching and implicit defaults.
- Enforce explicit configuration (no silent fallbacks) per repo guidelines.

## Current Pain Points
- Manager contains method-specific behavior (e.g., GradDotProd-only logic for validation concatenation, save intervals), making it harder to add new engines.
- Mixed responsibilities: initialization, per-iteration policy, and IO decisions live in one class.
- Inconsistent attach/reattach behavior (optimizer vs no-optimizer engines).

## Proposed Architecture

- EngineProtocol (single, minimal interface)
  - attach(context): Required. Context may include `optimizer`, `ddp_info`, `device`, and paths.
  - detach(): Required.
  - on_train_batch_start(X, Y, iter_num, batch_idx): Optional.
  - prepare_forward_input(X, Y): Required. Returns `(X_forward, Y_forward)`; engines that don’t modify input return original.
  - on_after_backward(): Optional.
  - on_before_optim_step(): Optional.
  - on_after_optim_step(): Optional.
  - should_save(iter_num): Optional; engine decides its own cadence.
  - save(iter_num): Optional; engine writes its outputs.
  - on_eval_start()/on_eval_end(): Optional.
  - cleanup(): Required; flush and free resources.

- EngineRegistry (extensible)
  - Registry maps `method` (str) → `factory(config) -> EngineProtocol`.
  - Engines register themselves (module-level function) or via a central `register_engine(method, factory)` API. Optionally allow setuptools entry points in the future.

- GhostEngineManager (thin orchestrator)
  - Responsibilities:
    - Validate config strictly and resolve required paths.
    - Instantiate engine via registry and call `engine.attach(context)`.
    - Provide narrow loop hooks that simply delegate to the engine:
      - `prepare_forward_input`, `on_train_batch_start`, `on_after_backward`, `on_before_optim_step`, `on_after_optim_step`, `save_if_needed(iter_num)`, `detach_for_evaluation`, `reattach_after_evaluation`, `cleanup`.
    - No method-specific conditionals. Unknown methods raise `ValueError`.
  - Path policy (explicit):
    - If `proj_dir` set: use as-is.
    - Else if `result_dir` set: use `os.path.join(result_dir, '<engine_default_subdir>')`.
    - If both or neither provided: raise `ValueError`.
    - Each engine defines its default subdir name, e.g., `grad_dotprods`, `projections`.

- Configuration Model (strict, typed)
  - `TrainingConfig`: global training fields (batch sizes, LR, dtypes, eval cadence, result_dir, device, ddp_info).
  - `EngineConfig` base: includes `method` and method-agnostic common fields.
  - Per-engine config classes: `GradDotProdConfig`, `GradProjConfig`, etc., with explicit required fields. No defaults inside manager.
  - A `validate_config(config)` step runs before engine creation and raises on any missing/invalid field.

- Lifecycle & Hooks
  - Training loop calls manager methods only; manager delegates to the engine:
    - before forward: `prepare_forward_input()`
    - after backward: `on_after_backward()`
    - before optimizer step: `on_before_optim_step()`
    - after optimizer step: `on_after_optim_step()`
    - per-iteration save: `save_if_needed(iter_num)` → engine.should_save + engine.save
  - Evaluation phase: `detach_for_evaluation()` / `reattach_after_evaluation()` use engine’s attach/detach consistently (no optimizer assumptions).

- Error Handling
  - No fallbacks. All missing fields raise `ValueError` with clear messages.
  - Unknown `method` raises `ValueError`.
  - Incompatible combinations (e.g., GradDotProd without val data) raise early.

## Migration Strategy
- Introduce `EngineProtocol` and refactor existing engines to conform via small adapters if necessary.
- Add a `registry.py` with simple register/get APIs; register `GradDotProdEngine` and `GradProjLoraEngine`.
- Slim `GhostEngineManager` to delegation-only logic; move method-specific code (e.g., save cadence, input concatenation, directory naming) into engines.
- Update training loop to call manager’s narrow interface (it largely already does), remove manager’s method branching.
- Provide a helper to build per-engine config objects from CLI args (strict validation).

## Testing Strategy
- Protocol conformance tests: verify each engine implements required methods and behaviors.
- Manager delegation tests: ensure manager calls are forwarded and no method-specific branches exist.
- Path policy tests: `proj_dir` vs `result_dir` behavior per engine default subdir name.
- Backward-compat mini-integration tests:
  - GradDotProd: dot products saved at cadence; val concatenation happens via engine.
  - GradProj: projections saved via engine, no dependence on manager’s save.

## Example Usage (conceptual)
```python
# Build configs explicitly
train_cfg = TrainingConfig(...)
engine_cfg = GradProjConfig(method='GradProjLora', proj_layers='mlp,attn', proj_rank_total=256, ...)

# Create manager (method resolved via registry)
manager = GhostEngineManager(train_cfg, engine_cfg, model, optimizer, ddp_info)

# Training loop (delegated)
manager.on_train_batch_start(X, Y, iter_num, batch_idx)
Xf, Yf = manager.prepare_forward_input(X, Y)
loss = model(Xf, labels=Yf).loss
loss.backward()
manager.on_after_backward()
manager.on_before_optim_step()
optimizer.step(); optimizer.zero_grad()
manager.on_after_optim_step()
manager.save_if_needed(iter_num)
```

## Additional Suggestions
- Capabilities discovery: allow engines to expose a `capabilities` dict (e.g., requires_val_batch, needs_optimizer, modifies_forward_input) to enable early validation and better error messages.
- Metrics/Logging: engines own their metric saving; manager only coordinates timing.
- Typed hints and docstrings across interfaces to improve IDE support and reduce misuse.

## Trade-offs
- Slightly more boilerplate when adding a new engine (define protocol methods and register), but payoff is a cleaner, stable surface and zero branching in the manager.
- Per-engine config classes add clarity at the cost of more initial code, aligned with “no fallback anywhere”.
