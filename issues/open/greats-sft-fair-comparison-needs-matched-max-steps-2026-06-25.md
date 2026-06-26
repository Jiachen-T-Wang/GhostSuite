# GREATS SFT: epoch-based runs put GREATS and Regular on different schedules

- **Status:** open
- **Severity:** low (fairness caveat / footgun, not a bug)
- **Area:** `examples/greats/sft/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Problem
`total_steps` is derived from how many samples each method *consumes* per step:

- `examples/greats/sft/main.py:46-48` —
  `consume = candidate_batch_size if GREATS else batch_size`;
  `steps_per_epoch = len(train_samples) // consume`.

So for the same `--num_train_epochs`, GREATS (consume = `N`) runs **fewer** optimizer steps
than Regular (consume = `k`), and the linear LR schedule (`num_training_steps = total_steps`,
warmup = `warmup_ratio * total_steps`) is correspondingly different length. The committed
repro avoids this only because it pins `--max_steps 120` for both arms
(`docs/logs/greats_sft_repro/compare_2026-06-25.log` shows `375/epoch` for GREATS vs
`750/epoch` for Regular, both capped at 120).

## Impact
Anyone running the documented epoch-based `train.sh` without a matched `--max_steps` gets
two arms on different step counts and different schedules — not a fair GREATS-vs-Regular
comparison.

## Proposed fix
Document (in `examples/greats/sft/README.md`) that fair comparison requires a matched
`--max_steps` for both arms, and/or have `train.sh` set an explicit shared `--max_steps`
for the comparison recipe. Optionally, define an epoch in terms of optimizer *steps* (not
consumed samples) so both arms run the same number of updates per epoch.
