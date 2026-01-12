#!/bin/bash

#SBATCH --job-name=torchtitan       # Job name
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err
#SBATCH --time=13:59:59             
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem-per-cpu=16G            

#SBATCH --gres=gpu:1
#SBATCH --partition=ailab

set -euo pipefail

BASE_DIR="/scratch/gpfs/PMITTAL/tianhao/GhostSuite/examples/torchtitan"
RUN_SCRIPT="${BASE_DIR}/run_train.sh"
DUMP_ROOT="${BASE_DIR}/results"
TRAIN_DATASET="${TRAIN_DATASET:-}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-}"

# Sweep configuration ---------------------------------------------------------
# Set your Weights & Biases project for all runs in this sweep.
export WANDB_PROJECT="torchtitan-sweep-c4"

# Learning rates to sweep over.
LRS=(1e-6 1e-5 1e-4 7e-4 1e-3 3e-3 5e-3 7e-3 8e-3 9e-3 1e-2)

# Warmup steps to sweep over. Set to an empty array (WARMUP_STEPS=()) to keep the default.
WARMUP_STEPS=(100)

# Extra arguments shared by all runs (e.g., limit training steps for quick sweeps).
# Example: EXTRA_ARGS=(--training.steps=500 --job.description=\"sweep-test\")
EXTRA_ARGS=()

# CLI overrides for dataset selection and additional passthrough args.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            TRAIN_DATASET="$2"
            shift 2
            ;;
        --dataset=*)
            TRAIN_DATASET="${1#*=}"
            shift
            ;;
        --dataset-path)
            TRAIN_DATASET_PATH="$2"
            shift 2
            ;;
        --dataset-path=*)
            TRAIN_DATASET_PATH="${1#*=}"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Optional: override the default config file or training module.
# CONFIG_FILE="${BASE_DIR}/torchtitan/models/pythia/pythia_70m.toml"
# TRAIN_FILE="torchtitan.train"
# NGPU=1
# LOG_RANK=0

cd "${BASE_DIR}"

warmup_values=("${WARMUP_STEPS[@]}")
if [[ ${#warmup_values[@]} -eq 0 ]]; then
    warmup_values=("")  # empty string means "use default warmup"
fi

mkdir -p "${DUMP_ROOT}/${WANDB_PROJECT}"

for lr in "${LRS[@]}"; do
    for warmup in "${warmup_values[@]}"; do
        args=(--optimizer.lr="${lr}")
        run_suffix="lr${lr}"

        if [[ -n "${warmup}" ]]; then
            args+=(--lr_scheduler.warmup_steps="${warmup}")
            run_suffix+="_wu${warmup}"
        else
            run_suffix+="_wuDefault"
        fi

        if [[ -n "${TRAIN_DATASET}" ]]; then
            args+=(--training.dataset="${TRAIN_DATASET}")
            run_suffix+="_${TRAIN_DATASET}"
        fi

        if [[ -n "${TRAIN_DATASET_PATH}" ]]; then
            args+=(--training.dataset_path="${TRAIN_DATASET_PATH}")
        fi

        export WANDB_RUN_NAME="${run_suffix}"
        dump_folder="${DUMP_ROOT}/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
        mkdir -p "${dump_folder}"

        echo "============================================================"
        echo "Starting sweep run: ${WANDB_RUN_NAME}"
        echo "Args: ${args[*]} --job.dump_folder=${dump_folder} ${EXTRA_ARGS[*]}"
        echo "============================================================"

        "${RUN_SCRIPT}" --job.dump_folder="${dump_folder}" "${args[@]}" "${EXTRA_ARGS[@]}"
    done
done
