# GREATS pretrain: `prepare_gradients()` runs (and is discarded) in the scoring pass

- **Status:** open
- **Severity:** low (wasted work / clarity, not a correctness bug)
- **Area:** `examples/greats/pretrain/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
`_ghost_pass` calls `self.ghost.prepare_gradients()` unconditionally, including on the
scoring pass (`do_step=False`):

- `examples/greats/pretrain/training_loop.py:194` — `self.ghost.prepare_gradients()` (no `do_step` guard).

`prepare_gradients()` → `_prepare_and_apply_train_grad()` runs the full subtract-val
recovery over every parameter (`scale * (grad - grad_val)`, deletes `_ghost_grad_val`,
locks grad creation; `ghostEngines/graddotprod_engine.py:172-209`). On a scoring pass the
result is immediately discarded (`optimizer.zero_grad` at `training_loop.py:210`), so the
only effect is wasted per-parameter work and a lock/unlock cycle. Correctness is unaffected
(the dot-product log, the only thing scoring needs, is produced independently in the
backward hooks).

## Proposed fix
Guard the recovery on `do_step`:

```python
if do_step:
    self.ghost.prepare_gradients()
    self.scaler.unscale_(self.optimizer)
    ...
```

The scoring pass then only does `aggregate_and_log()` + read + clear, which is all it
needs. Cheaper and clearer about intent.
