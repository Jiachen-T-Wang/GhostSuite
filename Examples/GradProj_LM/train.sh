#!/bin/bash

#SBATCH --job-name=ghost-test       # Job name
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --output=/scratch/gpfs/tw8948/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/tw8948/slurm_output/slurm-%j.err
#SBATCH --time=2:59:59             
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem-per-cpu=16G            
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --partition=pli-lc
#SBATCH --account=ai2_data

# Default values
ARCHITECTURE="GPT2-Small"
BATCH_SIZE=2
MAX_SAMPLES=1000
PROJ_LAYERS="mlp,attn"
PROJ_RANK_TOTAL=256
PROJ_RANK_MIN=4
PROJ_SEED=42
PROJ_DTYPE="bfloat16"
MODEL_DTYPE="bfloat16"
TRAIN_DTYPE="bfloat16"
BLOCK_SIZE=1024
SEED=42
OUTPUT_DIR="./Results"
DEVICE="cuda"
PROJ_SAVE_INTERVAL=1
VERBOSE=false
PROJ_ROW_ORTHONORMAL=false
INCLUDE_EMBEDDINGS=false

# Function to show usage
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Gradient Projection Computation for GPT-2 on Pile Dataset"
    echo ""
    echo "Options:"
    echo "  --architecture ARCH        Model architecture: GPT2-Small, GPT2-Medium, GPT2-Large (default: $ARCHITECTURE)"
    echo "  --batch_size SIZE          Batch size for processing (default: $BATCH_SIZE)"
    echo "  --max_samples NUM          Maximum samples to process, 'all' for entire dataset (default: $MAX_SAMPLES)"
    echo "  --proj_layers LAYERS       Comma-separated layer patterns: mlp, attn, mlp,attn (default: $PROJ_LAYERS)"
    echo "  --proj_rank_total RANK     Target total projection dimension per layer (default: $PROJ_RANK_TOTAL)"
    echo "  --proj_rank_min MIN        Minimum dimension for k_i and k_o (default: $PROJ_RANK_MIN)"
    echo "  --proj_seed SEED           Random seed for projection matrices (default: $PROJ_SEED)"
    echo "  --proj_dtype DTYPE         Data type for projections: float16, bfloat16, float32 (default: $PROJ_DTYPE)"
    echo "  --model_dtype DTYPE        Model data type: float16, bfloat16, float32 (default: $MODEL_DTYPE)"
    echo "  --train_dtype DTYPE        Training data type: float16, bfloat16, float32 (default: $TRAIN_DTYPE)"
    echo "  --block_size SIZE          Sequence length for GPT2 (default: $BLOCK_SIZE)"
    echo "  --seed SEED                Random seed for data sampling (default: $SEED)"
    echo "  --output_dir DIR           Directory to save projections (default: $OUTPUT_DIR)"
    echo "  --device DEVICE            Device to use: cuda, cpu (default: $DEVICE)"
    echo "  --proj_save_interval INT   Save projections every N iterations (default: $PROJ_SAVE_INTERVAL)"
    echo "  --verbose                  Enable verbose output (default: disabled)"
    echo "  --proj_row_orthonormal     Use row-orthonormal projections (default: disabled)"
    echo "  --include_embeddings       Include embedding layers (default: disabled)"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Use all defaults"
    echo "  $0 --batch_size 4 --max_samples 5000                # Larger batch and more samples"
    echo "  $0 --architecture GPT2-Medium --proj_layers mlp     # Medium model, MLP layers only"
    echo "  $0 --max_samples all --verbose                       # Process entire dataset with verbose output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --proj_layers)
            PROJ_LAYERS="$2"
            shift 2
            ;;
        --proj_rank_total)
            PROJ_RANK_TOTAL="$2"
            shift 2
            ;;
        --proj_rank_min)
            PROJ_RANK_MIN="$2"
            shift 2
            ;;
        --proj_seed)
            PROJ_SEED="$2"
            shift 2
            ;;
        --proj_dtype)
            PROJ_DTYPE="$2"
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
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --proj_save_interval)
            PROJ_SAVE_INTERVAL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --proj_row_orthonormal)
            PROJ_ROW_ORTHONORMAL=true
            shift
            ;;
        --include_embeddings)
            INCLUDE_EMBEDDINGS=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Basic validation
if [[ "$ARCHITECTURE" != "GPT2-Small" && "$ARCHITECTURE" != "GPT2-Medium" && "$ARCHITECTURE" != "GPT2-Large" ]]; then
    echo "Error: Invalid architecture '$ARCHITECTURE'. Must be GPT2-Small, GPT2-Medium, or GPT2-Large"
    exit 1
fi

if [[ "$PROJ_DTYPE" != "float16" && "$PROJ_DTYPE" != "bfloat16" && "$PROJ_DTYPE" != "float32" ]]; then
    echo "Error: Invalid proj_dtype '$PROJ_DTYPE'. Must be float16, bfloat16, or float32"
    exit 1
fi

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
    echo "Error: Invalid device '$DEVICE'. Must be cuda or cpu"
    exit 1
fi

# Convert max_samples to appropriate format
if [[ "$MAX_SAMPLES" == "all" ]]; then
    MAX_SAMPLES_ARG=""
else
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

# Show configuration
echo "==================================="
echo "Gradient Projection Configuration"
echo "==================================="
echo "Architecture: $ARCHITECTURE"
echo "Batch size: $BATCH_SIZE"
echo "Max samples: $MAX_SAMPLES"
echo "Projection layers: $PROJ_LAYERS"
echo "Projection rank total: $PROJ_RANK_TOTAL"
echo "Projection rank min: $PROJ_RANK_MIN"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Verbose: $VERBOSE"
echo "==================================="

# Build Python command with parsed arguments
PYTHON_CMD="python main.py"
PYTHON_CMD="$PYTHON_CMD --architecture $ARCHITECTURE"
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --proj_layers \"$PROJ_LAYERS\""
PYTHON_CMD="$PYTHON_CMD --proj_rank_total $PROJ_RANK_TOTAL"
PYTHON_CMD="$PYTHON_CMD --proj_rank_min $PROJ_RANK_MIN"
PYTHON_CMD="$PYTHON_CMD --proj_seed $PROJ_SEED"
PYTHON_CMD="$PYTHON_CMD --proj_dtype $PROJ_DTYPE"
PYTHON_CMD="$PYTHON_CMD --model_dtype $MODEL_DTYPE"
PYTHON_CMD="$PYTHON_CMD --train_dtype $TRAIN_DTYPE"
PYTHON_CMD="$PYTHON_CMD --block_size $BLOCK_SIZE"
PYTHON_CMD="$PYTHON_CMD --seed $SEED"
PYTHON_CMD="$PYTHON_CMD --output_dir \"$OUTPUT_DIR\""
PYTHON_CMD="$PYTHON_CMD --device $DEVICE"
PYTHON_CMD="$PYTHON_CMD --proj_save_interval $PROJ_SAVE_INTERVAL"

# Add max_samples if specified
if [[ -n "$MAX_SAMPLES_ARG" ]]; then
    PYTHON_CMD="$PYTHON_CMD $MAX_SAMPLES_ARG"
fi

# Add boolean flags if enabled
if [[ "$VERBOSE" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

if [[ "$PROJ_ROW_ORTHONORMAL" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --proj_row_orthonormal"
fi

if [[ "$INCLUDE_EMBEDDINGS" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --include_embeddings"
fi

echo "Running command: $PYTHON_CMD"
echo ""

# Run the gradient projection computation
eval $PYTHON_CMD

echo ""
echo "Gradient projection computation completed!"
echo "Results saved to: $OUTPUT_DIR"
