# GREATS SFT repro doc under-specifies the actual run (percentage / epochs)

- **Status:** open
- **Severity:** low (doc accuracy)
- **Area:** `docs/analysis/`, `docs/logs/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
`docs/analysis/greats_sft_repro_2026-06-25.md` describes the upstream mirror with
`percentage=0.05`, and its Setup section lists `max_train_samples=3000`. The committed log,
however, shows the run actually used `percentage=1.0` with the 3000-sample cap, and 5
epochs (capped at 120 steps):

- `docs/logs/greats_sft_repro/compare_2026-06-25.log:2` —
  `Loaded 3000 instruction samples (from 3000 raw, percentage=1.0)`.
- `:4` — `total optimizer steps: 120 (375/epoch x 5 epochs, consume=8)`.

The numbers are internally consistent for a time-boxed run, but the doc's framing
(`percentage=0.05`, no mention of `--num_train_epochs 5`) doesn't match what was executed.

## Proposed fix
Add the exact command/flags actually used to the doc's Setup section, e.g.
`--percentage 1.0 --max_train_samples 3000 --num_train_epochs 5 --max_steps 120`, and
distinguish "upstream config being mirrored" (`percentage=0.05`, 3 epochs) from "this
time-boxed run's overrides". Per AGENTS.md, the analysis doc should record the exact
command/config/seed/GPU for the committed log.
