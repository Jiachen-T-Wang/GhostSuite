#!/bin/bash
#SBATCH --job-name=opus-pretrain
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
fi

# Defaults (override any via --flag value).
METHOD="OPUS"
ARCHITECTURE="GPT2-Small"
BATCH_SIZE=16              # k: trained subset size
CANDIDATE_BATCH_SIZE=32    # N: candidate pool scored each step
VAL_BATCH_SIZE=16          # m: proxy (scoring target) size
SELECTION_METHOD="stochastic"
TEMPERATURE=0.9
PRECONDITIONER="adamw_scalar"
PROJ_DIM=8192
SCORE_SEQ_LEN=""
WARMUP_STEP=2000
LEARNING_RATE=6e-4
MAX_STEPS=20000
SEED=42
TRAIN_SET="pile"
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
        --opus_selection_method) SELECTION_METHOD="$2"; shift 2;;
        --opus_temperature) TEMPERATURE="$2"; shift 2;;
        --opus_preconditioner) PRECONDITIONER="$2"; shift 2;;
        --proj_dim) PROJ_DIM="$2"; shift 2;;
        --score_seq_len) SCORE_SEQ_LEN="$2"; shift 2;;
        --warmup_step) WARMUP_STEP="$2"; shift 2;;
        --learning_rate) LEARNING_RATE="$2"; shift 2;;
        --max_steps) MAX_STEPS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --train_set) TRAIN_SET="$2"; shift 2;;
        --eval_interval) EVAL_INTERVAL="$2"; shift 2;;
        --eval_iter) EVAL_ITER="$2"; shift 2;;
        --eval_bs) EVAL_BS="$2"; shift 2;;
        --model_dtype) MODEL_DTYPE="$2"; shift 2;;
        --train_dtype) TRAIN_DTYPE="$2"; shift 2;;
        --wandb) USE_WANDB=true; shift 1;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] — see examples/opus/README.md"
            exit 0;;
        *) echo "Unknown option: $1"; echo "Use --help for usage"; exit 1;;
    esac
done

cd "$(dirname "$0")" || exit 1

CMD="python main.py --method \"$METHOD\" --architecture \"$ARCHITECTURE\""
CMD="$CMD --batch_size \"$BATCH_SIZE\" --candidate_batch_size \"$CANDIDATE_BATCH_SIZE\""
CMD="$CMD --val_batch_size \"$VAL_BATCH_SIZE\""
CMD="$CMD --opus_selection_method \"$SELECTION_METHOD\" --opus_temperature \"$TEMPERATURE\""
CMD="$CMD --opus_preconditioner \"$PRECONDITIONER\" --proj_dim \"$PROJ_DIM\""
if [ -n "$SCORE_SEQ_LEN" ]; then
    CMD="$CMD --score_seq_len \"$SCORE_SEQ_LEN\""
fi
CMD="$CMD --warmup_step \"$WARMUP_STEP\" --learning_rate \"$LEARNING_RATE\""
CMD="$CMD --max_steps \"$MAX_STEPS\" --seed \"$SEED\" --train_set \"$TRAIN_SET\""
CMD="$CMD --eval_interval \"$EVAL_INTERVAL\" --eval_iter \"$EVAL_ITER\" --eval_bs \"$EVAL_BS\""
CMD="$CMD --model_dtype \"$MODEL_DTYPE\" --train_dtype \"$TRAIN_DTYPE\""
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --wandb"
fi

echo "Executing: $CMD"
eval $CMD
