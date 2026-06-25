#!/bin/bash
#SBATCH --job-name=greats-pretrain
#SBATCH --mail-type=fail
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab

# Load the proxy module only on compute nodes (inside a Slurm job).
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load proxy/default
else
    echo "Skipping proxy/default module load (not on a compute node)"
fi

# Defaults (override any via --flag value).
METHOD="GREATS"
ARCHITECTURE="GPT2-Small"
BATCH_SIZE=16              # k: trained subset size
CANDIDATE_BATCH_SIZE=32    # N: candidate pool scored each step
VAL_BATCH_SIZE=16          # m: scoring target size
SELECT_METRIC="dot"
WARMUP_STEP=2000
LEARNING_RATE=6e-4
OPTIMIZER="adamw"
MAX_STEPS=20000
SEED=42
TRAIN_SET="pile"
VAL_SET="pile"
EVAL_INTERVAL=200
EVAL_ITER=20
EVAL_BS=16
MODEL_DTYPE="float32"
TRAIN_DTYPE="bfloat16"
USE_WANDB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2;;
        --architecture) ARCHITECTURE="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --candidate_batch_size) CANDIDATE_BATCH_SIZE="$2"; shift 2;;
        --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2;;
        --select_metric) SELECT_METRIC="$2"; shift 2;;
        --warmup_step) WARMUP_STEP="$2"; shift 2;;
        --learning_rate) LEARNING_RATE="$2"; shift 2;;
        --optimizer) OPTIMIZER="$2"; shift 2;;
        --max_steps) MAX_STEPS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --train_set) TRAIN_SET="$2"; shift 2;;
        --val_set) VAL_SET="$2"; shift 2;;
        --eval_interval) EVAL_INTERVAL="$2"; shift 2;;
        --eval_iter) EVAL_ITER="$2"; shift 2;;
        --eval_bs) EVAL_BS="$2"; shift 2;;
        --model_dtype) MODEL_DTYPE="$2"; shift 2;;
        --train_dtype) TRAIN_DTYPE="$2"; shift 2;;
        --wandb) USE_WANDB=true; shift 1;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --method {GREATS,Regular}        Selection method (default: GREATS)"
            echo "  --architecture ARCH              GPT2-{Tiny,Small,Medium,Large}"
            echo "  --batch_size k                   Trained subset size (default: 16)"
            echo "  --candidate_batch_size N         Candidate pool size, N>=k (default: 32)"
            echo "  --val_batch_size m               Scoring target size (default: 16)"
            echo "  --select_metric {dot,cosine}     Candidate ranking score (default: dot)"
            echo "  --learning_rate / --max_steps / --seed / --train_set / dtypes ..."
            echo "  --wandb                          Enable Weights & Biases logging"
            exit 0;;
        *) echo "Unknown option: $1"; echo "Use --help for usage"; exit 1;;
    esac
done

# Run from this script's directory so `python main.py` resolves local modules.
cd "$(dirname "$0")" || exit 1

CMD="python main.py --method \"$METHOD\" --architecture \"$ARCHITECTURE\""
CMD="$CMD --batch_size \"$BATCH_SIZE\" --candidate_batch_size \"$CANDIDATE_BATCH_SIZE\""
CMD="$CMD --val_batch_size \"$VAL_BATCH_SIZE\" --select_metric \"$SELECT_METRIC\""
CMD="$CMD --warmup_step \"$WARMUP_STEP\" --learning_rate \"$LEARNING_RATE\""
CMD="$CMD --optimizer \"$OPTIMIZER\" --max_steps \"$MAX_STEPS\" --seed \"$SEED\""
CMD="$CMD --train_set \"$TRAIN_SET\" --val_set \"$VAL_SET\""
CMD="$CMD --eval_interval \"$EVAL_INTERVAL\" --eval_iter \"$EVAL_ITER\" --eval_bs \"$EVAL_BS\""
CMD="$CMD --model_dtype \"$MODEL_DTYPE\" --train_dtype \"$TRAIN_DTYPE\""
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --wandb"
fi

echo "Executing: $CMD"
eval $CMD
