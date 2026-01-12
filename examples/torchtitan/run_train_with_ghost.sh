#!/usr/bin/bash

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Ensure repo root (for ghostEngines) is on PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH}"

# Load proxy only when running on compute nodes (i.e., inside a Slurm job)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load proxy/default
else
    echo "Skipping proxy/default module load (not on compute node)"
fi

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train_with_ghost"}

WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan-ghost"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

export WANDB_DISABLE_CODE=${WANDB_DISABLE_CODE:-true}
export WANDB_IGNORE_GLOBS=${WANDB_IGNORE_GLOBS:-"output.log,wandb-metadata.json,w*.json,requirements.txt,config.yaml"}

COMM_MODE=${COMM_MODE:-""}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

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
    WANDB_RUN_NAME="$(date +%Y%m%d_%H%M%S)_ghost"
fi

export WANDB_PROJECT
export WANDB_RUN_NAME

if [ -n "$COMM_MODE" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" --job.config_file "${CONFIG_FILE}" "${PASSTHRU_ARGS[@]}" --comm.mode=${COMM_MODE} --training.steps=1
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "${PASSTHRU_ARGS[@]}"
fi
