#!/bin/bash

#SBATCH --job-name=ghost-test       # Job name
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --output=/scratch/gpfs/tw8948/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm_output/slurm-%j.err
#SBATCH --time=7:59:59             
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=32G                    # Memory per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --partition=pli-lc
#SBATCH --account=ai2_data



# Default values
METHOD="GradDotProd"
ARCHITECTURE="GPT2-Small"
BATCH_SIZE=16
VAL_BATCH_SIZE=1
WARMUP_STEP=2000
LEARNING_RATE=3e-4
OPTIMIZER="adamw"
MAX_STEPS=50000
SEED=42
TRAIN_SET="pile"
VAL_SET="pile"
EVAL_ONLY=false
EVAL_INTERVAL=200
EVAL_ITER=20
EVAL_BS=16
DOT_PROD_SAVE_INTERVAL=10
MODEL_DTYPE="bfloat16"
TRAIN_DTYPE="bfloat16"

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

# Build the command with all parameters
CMD="python main.py --method \"$METHOD\" --architecture \"$ARCHITECTURE\" --batch_size \"$BATCH_SIZE\" --val_batch_size \"$VAL_BATCH_SIZE\" --warmup_step \"$WARMUP_STEP\" --learning_rate \"$LEARNING_RATE\" --optimizer \"$OPTIMIZER\" --max_steps \"$MAX_STEPS\" --seed \"$SEED\" --train_set \"$TRAIN_SET\" --val_set \"$VAL_SET\" --eval_interval \"$EVAL_INTERVAL\" --eval_iter \"$EVAL_ITER\" --eval_bs \"$EVAL_BS\" --dot_prod_save_interval \"$DOT_PROD_SAVE_INTERVAL\" --model_dtype \"$MODEL_DTYPE\" --train_dtype \"$TRAIN_DTYPE\""

# Add eval_only flag if set
if [ "$EVAL_ONLY" = true ]; then
    CMD="$CMD --eval_only"
fi

echo "Executing command: $CMD"
eval $CMD




