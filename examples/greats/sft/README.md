# GREATS SFT — Online Selection for LoRA Instruction Tuning

Self-contained **online batch selection** during LoRA instruction tuning, mirroring the
upstream [GREATS](https://github.com/Jiachen-T-Wang/GREATS) / LESS `warmup_train.sh` setup
but running in the GhostSuite `.venv` on the `GradDotProd` engine. Supports both first-order
and the default second-order (Gram-based greedy) selection — see **Selection modes** below.

## What it does (per optimizer step)
1. Draw a candidate pool of `N = round(fracinv * batch_size)` instruction samples.
2. **Scoring pass** (GREATS): one `GradDotProd` forward/backward over `[candidate ++ val]`
   gives each candidate `s_i = <g_i, g_val>` over the **LoRA adapters** (`g_val` = gradient
   on the MMLU validation target). `cosine` ranking is also available.
3. Select the top-`k = batch_size` candidates.
4. **Detach the engine** and take a plain LoRA step on the selected subset, then reattach.
   (Plain update, not subtract-val, because instruction masking + variable lengths make the
   subtract-val sample-count scaling inexact — this matches upstream's normal training step.)

Only the LoRA `nn.Linear` adapters are trainable, so the engine scores exactly those; no
norm layers are scored.

## Setup
- Model: a Llama-2-7b-hf checkpoint. Pass `--model_path` (a HF hub id like
  `meta-llama/Llama-2-7b-hf`, or a local snapshot dir), or set `GREATS_SFT_MODEL_PATH`.
  A local snapshot needs no HF token.
- Data: instruction jsonl + MMLU under `--data_dir` (or set `GREATS_SFT_DATA_DIR`).
- Env: the GhostSuite `.venv` (`uv sync && source .venv/bin/activate`); `peft` is included.

## Quick start

### Plumbing smoke (tiny model, fast)
Build a tiny Llama that shares the real tokenizer, then run a few steps on a small data
slice (CPU is slow on shared login nodes — use a GPU node):
```bash
# (one-time) save a tiny Llama to a local path, then:
python main.py --method GREATS --model_path <tiny-llama-dir> \
    --data_dir <greats-data-dir> \
    --batch_size 4 --fracinv 2.0 --read_limit_per_file 60 --percentage 1.0 \
    --max_train_samples 48 --num_train_epochs 1 --max_steps 6 \
    --model_dtype float32 --device cuda
```

### Llama-2-7b LoRA (mirrors warmup_train.sh)
```bash
cd examples/greats/sft
./train.sh --method GREATS  --batch_size 4 --fracinv 2.0 --subject world_religions --n_val 4
./train.sh --method Regular --batch_size 4                --subject world_religions --n_val 4
```
`train.sh` sets LoRA r=128 / alpha=1 / dropout=0.1 on `q,k,v,o_proj`, lr 2e-5, bf16,
max_seq 512 — matching upstream `base_training_args.sh`.

## Key flags
`--method {GREATS,Regular}`, `--selection {first_order,second_order}`, `--batch_size` (k),
`--fracinv` (candidate pool `N = fracinv*k`), `--n_val` / `--subject` (MMLU target),
`--percentage` (fraction of the corpus), `--lora_*`, `--learning_rate`, `--max_seq_length`,
`--num_train_epochs` / `--max_steps`, `--model_path`, `--data_dir`.

## Selection modes
- `--selection second_order` (**default, true GREATS**): in one forward/backward over
  `[candidate ++ val]`, `gram_scorer.py` builds both the first-order TracIN scores
  `<g_i, g_val>` **and** the candidate-candidate Gram `<g_i, g_j>` over the LoRA params,
  then runs the redundancy-aware greedy selection weighted by `(lr, lr^2)` (port of
  upstream `greedy_selection`). The per-sample LoRA gradient is materialized directly
  (LoRA factors are small), and the Gram math is verified against per-sample autograd to
  ~1e-7.
- `--selection first_order`: top-k by `<g_i, g_val>` via the `GradDotProd` engine (the
  weaker ablation; what the `greats-example` branch shipped).

## Scope
Single-GPU; MMLU few-shot test accuracy as the headline metric (not the full LESS eval
harness).

## Experiment & results
A committed GREATS-vs-Regular MMLU answer-perplexity comparison — with the figure, raw run logs,
the comparison launcher (`run_compare.sbatch`), and the plot script — lives in
[`experiments/`](experiments/README.md).
