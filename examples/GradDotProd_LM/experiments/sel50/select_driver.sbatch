#!/bin/bash
#SBATCH --job-name=ghost-select
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:59
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err

# Selection driver: trains two Regular arms on a scoring run's dot-product
# scores -- top-50% (>= global median) and positive (>= 0). Args:
#   $1 = scoring run name (under results/)
#   $2 = arm prefix (e.g. sel20full / sel20excl)
#   $3 = max_steps for the selection runs (e.g. 10000)
set -euo pipefail
REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite
SCORE_NAME="$1"; PREFIX="$2"; MAX_STEPS="$3"; SEED="${4:-42}"
RUN_DIR="$REPO/examples/GradDotProd_LM/results/$SCORE_NAME"

module load proxy/default
cd "$REPO"; source init.sh
cd "$REPO/examples/GradDotProd_LM"

python experiments/sel50/compute_medians.py --run_dir "$RUN_DIR"
# shellcheck disable=SC1090
source "$RUN_DIR/sel50_medians.env"
echo "[select:$PREFIX] DOT_MEDIAN=$DOT_MEDIAN N=$N_SAMPLES MAX_STEPS=$MAX_STEPS"

run_arm () {
  local name="$1" threshold="$2"
  echo "=== [arm] $name  metric=dot_product threshold=$threshold steps=$MAX_STEPS ==="
  python main.py \
    --method Regular --architecture GPT2-Small \
    --batch_size 16 --val_batch_size 16 \
    --learning_rate 0.0006 --max_steps "$MAX_STEPS" --warmup_step 2000 \
    --optimizer adamw --seed "$SEED" \
    --model_dtype float32 --train_dtype bfloat16 \
    --eval_interval "${EVAL_INTERVAL:-200}" --eval_iter "${EVAL_ITER:-128}" --eval_bs 16 \
    --replay_run_dir "$RUN_DIR" \
    --replay_filter_metric dot_product --replay_filter_threshold "$threshold" \
    --wandb_run_name "$name"
}

# ARMS selects which arms to run (default both). e.g. ARMS="positive".
for arm in ${ARMS:-top50 positive}; do
  case "$arm" in
    top50)    run_arm "${PREFIX}_top50"    "$DOT_MEDIAN" ;;  # top-50% by dot-product
    positive) run_arm "${PREFIX}_positive" 0.0 ;;            # all positively-scored (dot >= 0)
    *) echo "[select:$PREFIX] unknown arm '$arm'" ;;
  esac
done
echo "[select:$PREFIX] ALL ARMS DONE"
