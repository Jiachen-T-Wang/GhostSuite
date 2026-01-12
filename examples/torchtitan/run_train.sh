#!/usr/bin/bash

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ensure paths resolve relative to this script, not the caller's CWD.
cd "${SCRIPT_DIR}"

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
# COMM_MODE="fake_backend" ./run_train.sh  # for config validation without GPU
# COMM_MODE="local_tensor" ./run_train.sh  # for local tensor debugging mode

# Load proxy only when running on compute nodes (i.e., inside a Slurm job)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load proxy/default
else
    echo "Skipping proxy/default module load (not on compute node)"
fi

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan-debug"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

# Skip code/file uploads while keeping live metrics
export WANDB_DISABLE_CODE=${WANDB_DISABLE_CODE:-true}
export WANDB_IGNORE_GLOBS=${WANDB_IGNORE_GLOBS:-"output.log,wandb-metadata.json,w*.json,requirements.txt,config.yaml"}

# COMM_MODE options: "fake_backend" (dry run), "local_tensor" (debug mode), or empty for normal training
COMM_MODE=${COMM_MODE:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# CLI overrides (kept out of python args)
PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-project=*)
            WANDB_PROJECT="${1#*=}"
            shift
            ;;
        --wandb-run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --wandb-run-name=*)
            WANDB_RUN_NAME="${1#*=}"
            shift
            ;;
        --)
            shift
            PASSTHRU_ARGS+=("$@")
            break
            ;;
        *)
            PASSTHRU_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME="$(date +%Y%m%d_%H%M%S)"
fi

export WANDB_PROJECT
export WANDB_RUN_NAME

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" --job.config_file "${CONFIG_FILE}" "${PASSTHRU_ARGS[@]}" --comm.mode=${COMM_MODE} --training.steps=1
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "${PASSTHRU_ARGS[@]}"
fi
