# Training Launcher Usage

This repo provides a convenience wrapper script to start training and set common
runtime options.

## Basic usage

```bash
./run_train.sh
```

You can pass any training arguments through to the Python module:

```bash
./run_train.sh --training.steps=1000 --optimizer.lr=1e-4
```

## Choosing a dataset (including DCLM curated recipes)

Use `--training.dataset` to pick a dataset by name and optionally `--training.dataset_path` to override its location:

```bash
# Train on curated recipe 18 (default path is inferred)
./run_train.sh --training.dataset=recipe_18

# Train on a custom location (e.g., Recipe_07 shards)
./run_train.sh --training.dataset=recipe_7 \
  --training.dataset_path=/scratch/gpfs/PMITTAL/tianhao/DCLM-Pool/data/from-2t/curated/Recipe_7/processed_data_curated/processed_data
```

Available curated recipe names: `recipe_1`–`recipe_37` (zero-padded variants like `recipe_07` also work). Each points to `shard_*_processed.jsonl` under `processed_data_curated/processed_data`.

## Weights & Biases defaults and overrides

By default, `run_train.sh` sets:

- `WANDB_PROJECT=torchtitan-debug`
- `WANDB_RUN_NAME=<timestamp>` (format: `YYYYMMDD_HHMMSS`)

You can override these either with environment variables or CLI flags:

```bash
# Environment variables
WANDB_PROJECT=my_project WANDB_RUN_NAME=my_run ./run_train.sh

# CLI flags
./run_train.sh --wandb-project my_project --wandb-run-name my_run
```

## Other common overrides

```bash
NGPU=4 LOG_RANK=0,1 ./run_train.sh
CONFIG_FILE=./torchtitan/models/pythia/pythia_70m.toml ./run_train.sh

# Sweep with a specific dataset
./run_train_sweep.sh --dataset recipe_18 --dataset-path /scratch/gpfs/PMITTAL/tianhao/DCLM-Pool/data/from-2t/curated/Recipe_18/processed_data_curated/processed_data

# Flexible sweep grid
./run_train_sweep_full.sh \
  --grid optimizer.lr=3e-4,1e-3 \
  --grid training.local_batch_size=4,8 \
  --dataset recipe_18 \
  --dataset-path /scratch/gpfs/PMITTAL/tianhao/DCLM-Pool/data/from-2t/curated/Recipe_18/processed_data_curated/processed_data \
  --wandb-project myproj --wandb-group ablation
```
