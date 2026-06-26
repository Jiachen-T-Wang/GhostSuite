# GREATS pretrain: norm-layer transient cleanup is replicated in the example, should live in the engine

- **Status:** open
- **Severity:** medium (brittle coupling; silent corruption if it regresses)
- **Area:** `examples/greats/pretrain/`, `ghostEngines/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
The pretrain trainer runs two engine passes per step with **different batch sizes**
(scoring over `N + m`, update over `k + m`). In subtract-val mode the LayerNorm/RMSNorm
backward path caches `activations` / `_ghost_saved_activation` on the module and never
clears it (its cleanup `norm_backward_hook` is only registered when subtract-val is OFF).
Without intervention, the stale activation from the scoring pass leaks into the update pass
and is reused at the wrong batch size — silent corruption.

The example works around this by deleting the engine's internal per-module attributes
itself, after every pass:

- `examples/greats/pretrain/training_loop.py:221-225` — `_clear_transient_layer_state()`
  deletes `activations`, `backprops`, `_ghost_saved_activation` across all modules.
- Engine side that creates the leak: `ghostEngines/autograd_grad_sample_dotprod.py:486-489`
  (norm branch caches and returns without cleanup) and `:507` (cleanup hook only registered
  when `not _SUBTRACT_VAL`).

## Why this is a problem
An *example* should not need to know the engine's internal attribute names to be correct.
This couples `examples/greats/pretrain/` to `ghostEngines` internals: if those attr names
change, this example silently regresses to the stale-activation bug with no error. Any
future multi-pass caller would have to re-discover and re-implement the same cleanup.

## Proposed fix
Move the cleanup into the engine so multi-pass use is correct by default: clear norm-layer
transient state (`activations` / `_ghost_saved_activation` / `backprops`) at the end of
`GradDotProdEngine.aggregate_and_log()` (or the equivalent end-of-pass boundary) when in
subtract-val mode. Then `_clear_transient_layer_state()` can be removed from the example.

## References
- `docs/investigations/norm_layer_stale_activation_2026-06-25.md` (the root-cause note).
