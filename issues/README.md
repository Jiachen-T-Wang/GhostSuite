# Issues

Durable follow-ups, tracked as one markdown file per issue.

- `open/` — active issues.
- `closed/` — resolved issues (move the file here, add a short resolution note + commit/date).

Filenames are descriptive kebab-case with a date suffix, matching the `docs/` convention.

## Open

### GREATS example (`greats-example` branch review, 2026-06-25)
- [sft: `n_val` > MMLU dev rows silently corrupts scoring](open/greats-sft-nval-dev-rows-misalignment-2026-06-25.md) — **high**
- [pretrain: norm-layer transient cleanup belongs in the engine, not the example](open/greats-pretrain-norm-cleanup-belongs-in-engine-2026-06-25.md) — medium
- [pretrain: no committed Phase-A validation / selection-benefit artifact](open/greats-pretrain-missing-phaseA-validation-artifact-2026-06-25.md) — medium
- [sft: epoch-based runs put GREATS and Regular on different schedules](open/greats-sft-fair-comparison-needs-matched-max-steps-2026-06-25.md) — low
- [sft: repro doc under-specifies the actual run (percentage/epochs)](open/greats-sft-repro-doc-vs-log-mismatch-2026-06-25.md) — low
- [pretrain: `prepare_gradients()` runs and is discarded in the scoring pass](open/greats-pretrain-prepare-gradients-runs-in-scoring-pass-2026-06-25.md) — low
- [sft: per-step engine detach/reattach re-registers hooks every step](open/greats-sft-per-step-detach-reattach-cost-2026-06-25.md) — low
