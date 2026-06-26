# GREATS SFT: `n_val` > available MMLU dev rows silently corrupts scoring

- **Status:** closed (fixed 2026-06-25, branch `greats-example`)
- **Severity:** high (silent dot-product corruption, no error raised)
- **Area:** `examples/greats/sft/`
- **Found:** 2026-06-25 (review of `greats-example`)

## Resolution
Applied both proposed fixes (combined, as recommended):
1. `data_utils.build_mmlu_val` now **raises `ValueError`** when the subject's dev set has
   fewer than `n_val` rows, instead of silently truncating to `min(n_val, len(df))`.
2. `training_loop.py` constructs the engine with `val_batch_size=len(self.val_samples)`
   (the actually-built target) rather than `config.n_val`, so the `[candidate ++ val]`
   split stays aligned regardless of how many rows were loaded.

Verified: `world_religions` (5 dev rows) builds fine at `n_val<=5`; `n_val=8` now fails fast
with a clear message instead of corrupting the scoring split. (Note: the dev CSV has 5 rows,
not 4 as stated below — the default `n_val=4` was safe, but the bug was real for any
`n_val` above the dev size.) A larger validation target should instead draw from the MMLU
**test** set (separate enhancement), not an oversized `n_val`.

## Problem
The validation target is built with `min(n_val, len(df))` rows, but the engine is
constructed with `val_batch_size = config.n_val`, and the per-sample train/val split is
inferred as `total_bs - val_batch_size`. If a subject's MMLU dev CSV has fewer rows than
`n_val`, `val_samples` is silently short while the engine still treats the last `n_val`
rows of the `[candidate ++ val]` batch as validation. The split is then misaligned: the
last real candidate(s) are counted as validation, so `entry["dot_product"]` is computed on
the wrong partition and the selected top-k is corrupted — **with no error**.

## Evidence
- `examples/greats/sft/data_utils.py:127` — `df = df[: min(n_val, len(df))]` (silent truncation).
- `examples/greats/sft/training_loop.py:54-55` — `GradDotProdEngine(... val_batch_size=config.n_val ...)`.
- `examples/greats/sft/training_loop.py:123-138` — scoring forward over `collate(candidates + self.val_samples)`; the split relies on `val_batch_size`.
- `ghostEngines/autograd_grad_sample_dotprod.py:448-450` — `_ghost_train_bs = total_bs - val_batch_size`.

This is not hypothetical: `world_religions_dev.csv` has **exactly 4 rows**, so the default
`n_val=4` works only by coincidence. The analysis doc's own recommended follow-up
("a larger `n_val` scoring target", `docs/analysis/greats_sft_repro_2026-06-25.md`) would
directly trigger the corruption.

## Impact
Any run with `n_val` greater than the subject's dev-set size silently selects on a
misaligned split. Default config is safe; the bug surfaces exactly when a user follows the
documented next step of increasing `n_val`.

## Proposed fix
Make the engine's val size track the actually-built target. Either:
1. In `training_loop.py`, set `val_batch_size = len(self.val_samples)` instead of
   `config.n_val`; **or**
2. In `data_utils.build_mmlu_val` / `main.py`, assert `len(val_samples) == config.n_val`
   and fail explicitly when the dev set is smaller than requested.

Option (1) is the more robust default (fail-safe to whatever was loaded); option (2)
follows the repo's "fail explicitly on missing state" guideline. Either is acceptable;
combining them (assert + use actual length) is best.
