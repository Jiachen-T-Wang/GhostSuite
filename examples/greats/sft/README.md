# GREATS SFT — Online Selection for LoRA Instruction Tuning

Self-contained **first-order online batch selection** during LoRA instruction tuning,
mirroring the upstream [GREATS](https://github.com/Jiachen-T-Wang/GREATS) / LESS
`warmup_train.sh` setup but running in the GhostSuite `.venv` on the `GradDotProd` engine.
See `docs/plans/greats_sft_phaseB_2026-06-25.md`.

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

## Setup (this cluster)
- Model: Llama-2-7b-hf is cached at a local path (default in `config_file.py`); no HF token
  needed. Override with `--model_path`.
- Data: instruction jsonl + MMLU under `--data_dir`
  (default `/scratch/gpfs/PMITTAL/tongwu/other/GREATS/data`).
- Env: the GhostSuite `.venv` (`source init.sh`); `peft` is included.

## Quick start

### Plumbing smoke (tiny model, fast)
Build a tiny Llama that shares the real tokenizer, then run a few steps on a small data
slice (CPU is slow on shared login nodes — use a GPU node):
```bash
# (one-time) save a tiny Llama to a shared path, then:
python main.py --method GREATS --model_path <tiny-llama-dir> \
    --data_dir /scratch/gpfs/PMITTAL/tongwu/other/GREATS/data \
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
`train.sh` sets LoRA r=128 / alpha=1 / dropout=0.1 on `q,k,v,o_proj`, lr 1e-5, bf16,
max_seq 512 — matching upstream `base_training_args.sh`.

## Key flags
`--method {GREATS,Regular}`, `--select_metric {dot,cosine}`, `--batch_size` (k),
`--fracinv` (candidate pool `N = fracinv*k`), `--n_val` / `--subject` (MMLU target),
`--percentage` (fraction of the corpus), `--lora_*`, `--learning_rate`, `--max_seq_length`,
`--num_train_epochs` / `--max_steps`, `--model_path`, `--data_dir`.

## Scope (v1)
First-order selection only (top-k by `<g_i,g_val>`); the pairwise Gram + greedy
second-order term (true GREATS) is a planned follow-up. Single-GPU; MMLU val-loss logging
(not the full LESS eval harness).
