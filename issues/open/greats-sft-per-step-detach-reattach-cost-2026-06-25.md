# GREATS SFT: per-step engine detach/reattach re-registers hooks every step

- **Status:** open
- **Severity:** low (performance; correct as written)
- **Area:** `examples/greats/sft/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
Each GREATS step detaches the engine for the plain LoRA update and reattaches afterward:

- `examples/greats/sft/training_loop.py:113-116` — `self.engine.detach(); ... ; self.engine.attach(self.optimizer)`.

`detach()` → `remove_hooks()` and `attach()` → `add_hooks()` remove and re-register
forward-pre / forward / output hooks across **every** hooked LoRA `nn.Linear` on each step
(`ghostEngines/autograd_grad_sample_dotprod.py:388-572`, `:575-598`). On a 7B model with
LoRA on `q,k,v,o_proj` this is hundreds of hook (de)registrations per step.

This is correct (the ungated output hook would otherwise fire in the plain backward), but
it is real per-step overhead.

## Proposed fix
Two options, in increasing effort:
1. Add a one-line comment at the detach/reattach site noting this is a known per-step cost
   and why the detach is required (so it isn't mistaken for accidental churn).
2. Replace detach/reattach with a cheap enable/disable gate on the engine's output hook, so
   the plain update skips dot-product work without tearing down and rebuilding all hooks.

Option (2) also benefits any future multi-pass caller.
