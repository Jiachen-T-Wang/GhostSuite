#!/bin/bash
#SBATCH --job-name=greats-sft-cmp
#SBATCH --time=1:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/GhostSuite-wt-greats/outputs/sft_cmp_%x_%j.out

# Reruns the GREATS-vs-Regular SFT comparison with the up-to-date upstream config
# (lr 2e-5, label masking, shuffled val mini-batch, MMLU sociology test accuracy).
# Usage: sbatch run_compare.sbatch <METHOD> <BATCH_SIZE> <FRACINV>
#   GREATS arm:  sbatch run_compare.sbatch GREATS 2 2.0   (pool 4, select 2)
#   Regular arm: sbatch run_compare.sbatch Regular 4 1.0  (train 4, no selection)
set -e
METHOD=${1:-GREATS}
BATCH_SIZE=${2:-2}
FRACINV=${3:-2.0}

module load proxy/default 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite-wt-greats
cd "$REPO"
source init.sh >/dev/null 2>&1
cd "$REPO/examples/greats/sft"

python -u main.py \
    --method "$METHOD" --batch_size "$BATCH_SIZE" --fracinv "$FRACINV" \
    --subject sociology --n_val 5 --val_batchsize 2 --n_test 500 \
    --learning_rate 2e-5 --lora_r 128 --lora_alpha 1 --lora_dropout 0.1 \
    --max_seq_length 512 --model_dtype bfloat16 \
    --percentage 1.0 --read_limit_per_file 2000 --max_train_samples 5000 \
    --num_train_epochs 1 --max_steps 1200 --eval_interval 200 --logging_steps 20 \
    --seed 11
