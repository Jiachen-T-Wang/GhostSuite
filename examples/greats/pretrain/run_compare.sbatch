#!/bin/bash
#SBATCH --job-name=greats-pre-cmp
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/GhostSuite-wt-greats/examples/greats/pretrain/results/pre_cmp_%x_%j.out

# GREATS pretraining comparison: GPT2-Small on Pile, online batch selection.
# Equal update size: Regular trains on k=16 random; GREATS selects the best 16 of a
# 32-candidate pool. Three arms via args:
#   sbatch run_compare.sbatch Regular  no    <MAX_STEPS>
#   sbatch run_compare.sbatch GREATS   excl  <MAX_STEPS>   # --score_exclude_params wte,lm_head
#   sbatch run_compare.sbatch GREATS   incl  <MAX_STEPS>   # full score (tied wte/lm_head included)
set -e
METHOD=${1:-GREATS}
EXCL=${2:-excl}
MAX_STEPS=${3:-10000}

module load proxy/default 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PILE_DATA_DIR_TRAIN=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-train
export PILE_DATA_DIR_VAL=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-val-gpt2
export PILE_DATA_DIR_TEST=/scratch/gpfs/PMITTAL/tianhao/PretrainData/pile/pile-test-gpt2

REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite-wt-greats
cd "$REPO"
source init.sh >/dev/null 2>&1
cd "$REPO/examples/greats/pretrain"

ARGS="--architecture GPT2-Small --train_set pile --val_set pile \
  --batch_size 16 --val_batch_size 16 \
  --learning_rate 6e-4 --warmup_step 2000 --max_steps $MAX_STEPS \
  --eval_interval 500 --eval_iter 200 --eval_bs 16 \
  --model_dtype float32 --train_dtype bfloat16 --seed 42 \
  --wandb_run_name ${METHOD}_${EXCL}_pre"

if [ "$METHOD" = "GREATS" ]; then
    ARGS="$ARGS --method GREATS --candidate_batch_size 32"
    if [ "$EXCL" = "excl" ]; then
        ARGS="$ARGS --score_exclude_params wte,lm_head"
    fi
else
    ARGS="$ARGS --method Regular"
fi

echo "Running: python main.py $ARGS"
python -u main.py $ARGS
