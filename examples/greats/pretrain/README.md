# GREATS Pretraining — Online Batch Selection (first-order)

Online batch selection during GPT-2 pretraining, built on the `GradDotProd` ghost engine
and the shared model/data utilities under `examples/lm/`.

## Algorithm (per training step)
Let `N` = candidate pool size (`--candidate_batch_size`), `k` = trained subset size
(`--batch_size`, `N >= k`), `m` = scoring target size (`--val_batch_size`).

1. **Draw** a candidate pool of `N` train samples and a fresh scoring val batch of `m`
   samples (from the fixed eval window pool by default, so selection targets exactly the
   eval population).
2. **Scoring pass** — one fused `GradDotProd` forward/backward over `[candidate ++ val]`
   yields per-candidate `s_i = <g_i, g_val>` (or cosine with `--select_metric cosine`).
   No optimizer step.
3. **Select** the top-`k` candidates by `s_i`.
4. **Update pass** — a normal `GradDotProd` step over `[selected ++ val]`; subtract-val
   recovers the selected-subset mean gradient into `.grad` and the optimizer steps.

Steps 2 and 4 are **two engine passes per step** (matching GREATS' two-pass note). The
scoring pass cannot be reused for the update because subtract-val recovers the mean
gradient over the *whole* candidate pool, not the selected subset.

This is first-order selection: there is **no** pairwise train–train Gram matrix and **no**
greedy second-order redundancy term. Adding them is future work (see the plan).

## Quick start

### Smoke test (synthetic data, 1 GPU, no corpus)
```bash
python examples/greats/pretrain/main.py --method GREATS --train_set synthetic \
    --architecture GPT2-Tiny --candidate_batch_size 16 --batch_size 8 \
    --val_batch_size 4 --max_steps 12 --eval_interval 4 \
    --model_dtype float32 --train_dtype float32
```
A pass prints, per step, the loss and selection stats (mean selected score, fraction of
positive-scoring candidates) and finite, non-diverging eval losses.

### Real training (tokenized Pile)
```bash
cd examples/greats/pretrain
./train.sh --architecture GPT2-Small --candidate_batch_size 32 --batch_size 16
# No-selection baseline (trains on a normal batch of --batch_size):
./train.sh --method Regular --batch_size 16
```
See `examples/lm/graddotprod_lm/README.md` for how to tokenize the Pile.

## Key flags
- `--method {GREATS,Regular}` — online selection vs. no-selection baseline.
- `--candidate_batch_size N` — candidate pool scored each step (`N >= --batch_size`).
- `--batch_size k` — trained subset size (samples kept per step).
- `--val_batch_size m` — scoring target (validation) batch size.
- `--select_metric {dot,cosine}` — ranking score; `cosine` forces per-sample grad norms.
- `--score_val_from_eval_pool / --no-score_val_from_eval_pool` — draw the scoring val
  batch from the fixed eval window pool (default on).
- `--score_exclude_params wte,lm_head` — exclude params from the *score* only (training
  is unaffected); useful to stop large tied layers dominating the raw score.

## Sanity checks
- **Reduction:** `--candidate_batch_size == --batch_size` selects all candidates, i.e.
  trains on the full pool (a useful no-selection control).
- **Selection benefit:** top-`k` should reach lower val loss than bottom-`k` (rank by
  `-s_i`) on the same stream — mirroring the `examples/lm/graddotprod_lm` `sel50`
  experiments.

## Cost & measurement
GREATS does `~(N+m)` scoring + `~(k+m)` update sample-forwards per step vs. `k` for the
baseline, so step time is genuinely higher — the scoring pass is real work. Report `tps`
honestly and follow the GPU/Slurm method in `AGENTS.md` (H200, drop warmup, bench OFF).
