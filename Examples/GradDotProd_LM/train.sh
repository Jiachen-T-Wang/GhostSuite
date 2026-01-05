#!/bin/bash

#SBATCH --job-name=train-with-ghost       # Job name
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err
#SBATCH --time=5:59:59             
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # Memory per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --constraint="gpu80"

# Load proxy only when running on compute nodes (i.e., inside a Slurm job)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    module load proxy/default
else
    echo "Skipping proxy/default module load (not on compute node)"
fi



# Default values
METHOD="GradDotProd"
ARCHITECTURE="GPT2-Small"
BATCH_SIZE=16
VAL_BATCH_SIZE=16
WARMUP_STEP=2000
LEARNING_RATE=6e-4
OPTIMIZER="adamw"
MAX_STEPS=20000
SEED=42
TRAIN_SET="pile"
VAL_SET="pile"
EVAL_ONLY=false
EVAL_INTERVAL=200
EVAL_ITER=20
EVAL_BS=16
DOT_PROD_SAVE_INTERVAL=10
MODEL_DTYPE="float32"
TRAIN_DTYPE="bfloat16"
DYNAMIC_VAL_BATCH=true
LOG_GRAD_NORMS=false
REPLAY_RUN_DIR=""
REPLAY_FILTER_METRIC="dot_product"
REPLAY_FILTER_THRESHOLD=0.0
REPLAY_REBATCH_SIZE=""
REPLAY_DROP_LAST=false
REPLAY_SHUFFLE=false
REPLAY_SHUFFLE_SEED=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --val_batch_size)
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --warmup_step)
            WARMUP_STEP="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --train_set)
            TRAIN_SET="$2"
            shift 2
            ;;
        --val_set)
            VAL_SET="$2"
            shift 2
            ;;
        --eval_only)
            EVAL_ONLY=true
            shift 1
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --eval_iter)
            EVAL_ITER="$2"
            shift 2
            ;;
        --eval_bs)
            EVAL_BS="$2"
            shift 2
            ;;
        --dot_prod_save_interval)
            DOT_PROD_SAVE_INTERVAL="$2"
            shift 2
            ;;
        --model_dtype)
            MODEL_DTYPE="$2"
            shift 2
            ;;
        --train_dtype)
            TRAIN_DTYPE="$2"
            shift 2
            ;;
        --dynamic_val_batch)
            DYNAMIC_VAL_BATCH=true
            shift 1
            ;;
        --log_grad_norms)
            LOG_GRAD_NORMS=true
            shift 1
            ;;
        --replay_run_dir)
            REPLAY_RUN_DIR="$2"
            shift 2
            ;;
        --replay_filter_metric)
            REPLAY_FILTER_METRIC="$2"
            shift 2
            ;;
        --replay_filter_threshold)
            REPLAY_FILTER_THRESHOLD="$2"
            shift 2
            ;;
        --replay_rebatch_size)
            REPLAY_REBATCH_SIZE="$2"
            shift 2
            ;;
        --replay_drop_last)
            REPLAY_DROP_LAST=true
            shift 1
            ;;
        --replay_shuffle)
            REPLAY_SHUFFLE=true
            shift 1
            ;;
        --replay_shuffle_seed)
            REPLAY_SHUFFLE_SEED="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --wandb_mode)
            WANDB_MODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --method METHOD               Training method (default: Regular)"
            echo "  --architecture ARCH           Model architecture (default: GPT2-Small)"
            echo "  --batch_size SIZE             Batch size (default: 16)"
            echo "  --val_batch_size SIZE         Validation batch size (default: 1)"
            echo "  --warmup_step STEPS           Warmup steps (default: 2000)"
            echo "  --learning_rate RATE          Learning rate (default: 3e-4)"
            echo "  --optimizer OPT               Optimizer (default: adamw)"
            echo "  --max_steps STEPS             Maximum training steps (default: 50000)"
            echo "  --seed SEED                   Random seed (default: 42)"
            echo "  --train_set DATASET           Training dataset (default: pile)"
            echo "  --val_set DATASET             Validation dataset (default: pile)"
            echo "  --eval_only                   Evaluation only mode (default: false)"
            echo "  --eval_interval INTERVAL      Evaluation interval (default: 10)"
            echo "  --eval_iter ITER              Evaluation iterations (default: 20)"
            echo "  --eval_bs SIZE                Evaluation batch size (default: 16)"
            echo "  --dot_prod_save_interval INT  Dot product save interval (default: 10)"
            echo "  --model_dtype DTYPE           Model data type (default: bfloat16)"
            echo "  --train_dtype DTYPE           Training data type (default: bfloat16)"
            echo "  --dynamic_val_batch           Refresh validation batch every training step for GradDotProd"
            echo "  --log_grad_norms              Record per-sample train grad norm and aggregated val grad norm"
            echo "  --replay_run_dir PATH         Use stored GradDotProd logs as training data (filtered replay)"
            echo "  --replay_filter_metric NAME   dot_product or cosine (default: dot_product)"
            echo "  --replay_filter_threshold X   Drop samples below threshold (default: 0.0)"
            echo "  --replay_rebatch_size SIZE    Rebatch filtered samples to this size (default: batch_size)"
            echo "  --replay_drop_last            Drop final incomplete batch from replay data"
            echo "  --replay_shuffle              Shuffle filtered replay samples (loads all samples into memory)"
            echo "  --replay_shuffle_seed SEED    Seed for replay shuffle (defaults to --seed)"
            echo "  --wandb_project NAME          Weights & Biases project (default: GhostSuite)"
            echo "  --wandb_run_name NAME         Optional Weights & Biases run name"
            echo "  --wandb_mode MODE             Weights & Biases mode (online/offline/disabled, default: online)"
            echo "  -h, --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

WANDB_PROJECT="${WANDB_PROJECT:-GhostSuite}"
WANDB_MODE="${WANDB_MODE:-online}"
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    WANDB_RUN_NAME="${METHOD}_${ARCHITECTURE}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_${TIMESTAMP}"
fi

echo "Running with parameters:"
echo "Method: $METHOD"
echo "Architecture: $ARCHITECTURE"
echo "Batch size: $BATCH_SIZE"
echo "Val batch size: $VAL_BATCH_SIZE"
echo "Warmup step: $WARMUP_STEP"
echo "Learning rate: $LEARNING_RATE"
echo "Optimizer: $OPTIMIZER"
echo "Max steps: $MAX_STEPS"
echo "Seed: $SEED"
echo "Train set: $TRAIN_SET"
echo "Val set: $VAL_SET"
echo "Eval only: $EVAL_ONLY"
echo "Eval interval: $EVAL_INTERVAL"
echo "Eval iter: $EVAL_ITER"
echo "Eval batch size: $EVAL_BS"
echo "Dot prod save interval: $DOT_PROD_SAVE_INTERVAL"
echo "Model dtype: $MODEL_DTYPE"
echo "Train dtype: $TRAIN_DTYPE"
echo "Dynamic val batch: $DYNAMIC_VAL_BATCH"
echo "Log grad norms: $LOG_GRAD_NORMS"
echo "Replay run dir: ${REPLAY_RUN_DIR:-none}"
echo "Replay filter metric: $REPLAY_FILTER_METRIC"
echo "Replay filter threshold: $REPLAY_FILTER_THRESHOLD"
echo "Replay rebatch size: ${REPLAY_REBATCH_SIZE:-default}"
echo "Replay drop last: $REPLAY_DROP_LAST"
echo "Replay shuffle: $REPLAY_SHUFFLE"
echo "Replay shuffle seed: ${REPLAY_SHUFFLE_SEED:-default}"
echo "WandB project: $WANDB_PROJECT"
echo "WandB run name: $WANDB_RUN_NAME"
echo "WandB mode: $WANDB_MODE"

# Build the command with all parameters
CMD="python main.py --method \"$METHOD\" --architecture \"$ARCHITECTURE\" --batch_size \"$BATCH_SIZE\" --val_batch_size \"$VAL_BATCH_SIZE\" --warmup_step \"$WARMUP_STEP\" --learning_rate \"$LEARNING_RATE\" --optimizer \"$OPTIMIZER\" --max_steps \"$MAX_STEPS\" --seed \"$SEED\" --train_set \"$TRAIN_SET\" --val_set \"$VAL_SET\" --eval_interval \"$EVAL_INTERVAL\" --eval_iter \"$EVAL_ITER\" --eval_bs \"$EVAL_BS\" --dot_prod_save_interval \"$DOT_PROD_SAVE_INTERVAL\" --model_dtype \"$MODEL_DTYPE\" --train_dtype \"$TRAIN_DTYPE\" --wandb --wandb_project \"$WANDB_PROJECT\" --wandb_run_name \"$WANDB_RUN_NAME\" --wandb_mode \"$WANDB_MODE\""
if [ "$DYNAMIC_VAL_BATCH" = true ]; then
    CMD="$CMD --dynamic_val_batch"
fi
if [ "$LOG_GRAD_NORMS" = true ]; then
    CMD="$CMD --log_grad_norms"
fi
if [ -n "$REPLAY_RUN_DIR" ]; then
    CMD="$CMD --replay_run_dir \"$REPLAY_RUN_DIR\" --replay_filter_metric \"$REPLAY_FILTER_METRIC\" --replay_filter_threshold \"$REPLAY_FILTER_THRESHOLD\""
    if [ -n "$REPLAY_REBATCH_SIZE" ]; then
        CMD="$CMD --replay_rebatch_size \"$REPLAY_REBATCH_SIZE\""
    fi
    if [ "$REPLAY_DROP_LAST" = true ]; then
        CMD="$CMD --replay_drop_last"
    fi
    if [ "$REPLAY_SHUFFLE" = true ]; then
        CMD="$CMD --replay_shuffle"
    fi
    if [ -n "$REPLAY_SHUFFLE_SEED" ]; then
        CMD="$CMD --replay_shuffle_seed \"$REPLAY_SHUFFLE_SEED\""
    fi
fi

# Add eval_only flag if set
if [ "$EVAL_ONLY" = true ]; then
    CMD="$CMD --eval_only"
fi

echo "Executing command: $CMD"
eval $CMD
