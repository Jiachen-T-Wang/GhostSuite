# GREATS pretrain (Phase A): no committed validation artifact / selection-benefit result

- **Status:** open
- **Severity:** medium (validation gap, not a code bug)
- **Area:** `examples/greats/pretrain/`, `docs/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
Phase B (sft) ships a committed reproduction with both a writeup and a raw log
(`docs/analysis/greats_sft_repro_2026-06-25.md`, `docs/logs/greats_sft_repro/compare_2026-06-25.log`).
Phase A (pretrain) ships only a smoke-test command in its README and **no committed run
artifact**. The pretrain README even advertises a falsifiable check that is never
demonstrated:

> "Selection benefit: top-`k` should reach lower val loss than bottom-`k` (rank by `-s_i`)
> on the same stream" — `examples/greats/pretrain/README.md`

## Impact
There is no committed evidence that the pretrain example actually *selects* (i.e. that
top-k beats bottom-k / no-selection on the same stream), only that it runs. This is the
asymmetry to close before the pretrain example is treated as a validated reference.

## Proposed fix
Run and commit a small Phase-A result, following the AGENTS.md GPU/Slurm method (H200,
drop warmup, bench OFF):
- the smoke run (synthetic, GPT2-Tiny) to confirm finite/non-diverging eval loss; and
- a top-k vs bottom-k vs Regular selection-benefit comparison on the same stream (mirrors
  the existing `sel50` experiments), with a short writeup under `docs/analysis/` and the
  raw log under `docs/logs/`.
