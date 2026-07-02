# GREATS Pretraining — Online Batch Selection (first-order)

Online batch selection during GPT-2 pretraining, built on the `GradDotProd` ghost engine
and the shared model/data utilities under `examples/lm/`.

## Algorithm (per training step)
Let `N` = candidate pool size (`--candidate_batch_size`), `k` = trained subset size
(`--batch_size`, `N >= k`), `m` = scoring target size (`--val_batch_size`).

1. **Draw** a candidate pool of `N` train samples and a fresh scoring val batch of `m`
   samples (from the fixed eval window pool by default, so selection targets exactly the
   eval population).
2. **Scoring pass** — one `GradDotProd` forward/backward over `[candidate ++ val]` yields
   per-candidate `s_i = <g_i, g_val>` (or cosine with `--eager --select_metric cosine`).
   No optimizer step. This is the **only** ghost pass; by default it runs the **decoupled
   in-graph + `torch.compile` fast path**.
3. **Select** the top-`k` candidates by `s_i`.
4. **Update** — a **plain** forward/backward + optimizer step on the selected `k` only (no
   val, no ghost). This is exact: subtract-val over `[selected ++ val]` recovers the mean
   train gradient over the selected `k`, which equals a plain mean-loss backward over those
   `k`. Dropping the val (`m`) forwards + ghost overhead from the update is a **~25% per-step
   speedup** vs a second ghost pass.

This is first-order selection: there is **no** pairwise train–train Gram matrix and **no**
greedy second-order redundancy term. Adding them is future work (see the plan).

### Engine fast path (default) and flags
The scoring pass defaults to `--decoupled_fn --decoupled_compile` (the compile-clean in-graph
path, same as `examples/lm/graddotprod_lm`). Notes:
- `--eager` uses the per-layer-hook engine instead. Needed for **`--select_metric cosine`**
  (the fast path has no grad norms yet — future work) and for **large models** (e.g.
  GPT2-Large) where compile's extra activation memory OOMs but eager fits. At the plain-update
  working point eager is only ~4% slower than compile, so `--eager` is a safe choice.
- `--no_decoupled_compile` keeps the in-graph path without compile (slower than eager — not
  recommended); `--decoupled_mem_budget`, `--decoupled_compile_toplevel` are the compile levers
  (note: top-level compile regresses GPT-2; `mem_budget` does not lower the fp32-logits memory
  floor).
- The fast path requires bf16 training (`--train_dtype bfloat16`, GradScaler disabled).

### Update rule = a pluggable policy
The per-step "which samples to update on" logic is a `ghostEngines.SelectionPolicy` fed to the
shared `examples/lm/shared/selection_trainer.online_selection_step` driver. GREATS uses
`TopK(batch_size)`; the Regular baseline uses `NoSelection` (no scoring, plain step). The driver
also backs `examples/lm/graddotprod_lm` (`UpdateAll`), and `BottomK`/`Threshold` are available for
rejected-arm / online-sel50 ablations — so a new selection experiment is a one-line policy swap.

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
GREATS does `~(N+m)` scoring (ghost) + `~k` update (plain) sample-forwards per step vs. `k`
for the baseline, so step time is genuinely higher — the scoring pass is real work. (The
update is a plain step on the selected `k`, so it no longer pays the val `m` forwards or any
ghost overhead — a ~25% step-time saving vs a second ghost pass.) Report `tps` honestly:
measure steady-state (drop the first compile/warmup steps, benchmarking hooks off). Measured H200
steady-state (GREATS `N=32/k=16/m=16`, bf16): GPT2-Small **0.199 s** (compile) / 0.207 s
(eager); GPT2-Medium 0.463 s / 0.486 s.

## Experiment & results
A committed GREATS-vs-Regular comparison on Pile — with the val/test loss figure, raw run logs,
the comparison launcher (`run_compare.sbatch`), and the plot script — lives in
[`experiments/`](experiments/README.md).
