#!/bin/bash
#SBATCH --job-name=greats-sft-cmp
#SBATCH --time=1:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=slurm-%x-%j.out

# Reruns the SFT comparison with the up-to-date upstream config (lr 2e-5, label
# masking, shuffled val mini-batch), reporting MMLU eval/test answer perplexity (the
# trialrun.png metric) plus accuracy.
# Usage: sbatch run_compare.sbatch <METHOD> <BATCH_SIZE> <FRACINV> [SELECTION]
#   Regular:           sbatch run_compare.sbatch Regular 4 1.0
#   GREATS 1st-order:  sbatch run_compare.sbatch GREATS 2 2.0 first_order
#   GREATS 2nd-order:  sbatch run_compare.sbatch GREATS 2 2.0 second_order
set -e
METHOD=${1:-GREATS}
BATCH_SIZE=${2:-2}
FRACINV=${3:-2.0}
SELECTION=${4:-second_order}

module load proxy/default 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Resolve the repo root (dir containing init.sh) from the submit directory, so this
# works from any checkout/worktree without a hardcoded path.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$REPO" != "/" ] && [ ! -f "$REPO/init.sh" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/init.sh" ] || { echo "ERROR: could not locate repo root (init.sh) from ${SLURM_SUBMIT_DIR:-$(pwd)}"; exit 1; }
cd "$REPO"
source init.sh >/dev/null 2>&1
cd "$REPO/examples/greats/sft"

python -u main.py \
    --method "$METHOD" --selection "$SELECTION" --batch_size "$BATCH_SIZE" --fracinv "$FRACINV" \
    --subject sociology --n_val 5 --val_batchsize 2 --n_test 500 \
    --learning_rate 2e-5 --lora_r 128 --lora_alpha 1 --lora_dropout 0.1 \
    --max_seq_length 512 --model_dtype bfloat16 \
    --percentage 1.0 --read_limit_per_file 2000 --max_train_samples 5000 \
    --num_train_epochs 1 --max_steps 1200 --eval_interval 100 --logging_steps 50 \
    --seed 11
