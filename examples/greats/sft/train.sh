#!/bin/bash
#SBATCH --job-name=greats-sft
#SBATCH --mail-type=fail
#SBATCH --time=1:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab

# Mirrors the upstream GREATS/LESS warmup_train.sh config (LoRA r=128 on
# q/k/v/o_proj, lr 1e-5, bf16, max_seq 512), as a self-contained GhostSuite example.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then module load proxy/default; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

METHOD="GREATS"        # GREATS | Regular
SUBJECT="world_religions"
N_VAL=4
BATCH_SIZE=4
FRACINV=2.0
PERCENTAGE=0.05
EPOCHS=3
SEED=42
MAX_STEPS=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2;;
        --subject) SUBJECT="$2"; shift 2;;
        --n_val) N_VAL="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --fracinv) FRACINV="$2"; shift 2;;
        --percentage) PERCENTAGE="$2"; shift 2;;
        --num_train_epochs) EPOCHS="$2"; shift 2;;
        --max_steps) MAX_STEPS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 [--method GREATS|Regular] [--subject S] [--n_val N]"
            echo "          [--batch_size k] [--fracinv F] [--percentage P]"
            echo "          [--num_train_epochs E] [--max_steps M] [--seed S]"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

cd "$(dirname "$0")" || exit 1
python -u main.py \
    --method "$METHOD" --subject "$SUBJECT" --n_val "$N_VAL" \
    --batch_size "$BATCH_SIZE" --fracinv "$FRACINV" --percentage "$PERCENTAGE" \
    --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" --seed "$SEED" \
    --lora_r 128 --lora_alpha 1 --lora_dropout 0.1 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --learning_rate 1e-5 --max_seq_length 512 --model_dtype bfloat16
