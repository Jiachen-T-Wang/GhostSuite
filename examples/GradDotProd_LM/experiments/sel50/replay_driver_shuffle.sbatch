#!/bin/bash
#SBATCH --job-name=ghost-sel50-shuf
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:59
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err

# Shuffled rerun of the four selection arms: identical to replay_driver.sbatch
# but with --replay_shuffle --replay_shuffle_seed 42, so the selected half is
# trained in random order instead of original iteration order. This removes the
# replay-ordering confound, isolating the effect of WHICH data is kept. Reuses
# the existing scoring run + its precomputed medians (no rescoring).
set -euo pipefail

REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite
SCORE_NAME=score10k_sel50_seed42
RUN_DIR="$REPO/examples/GradDotProd_LM/results/$SCORE_NAME"
SHUF_SEED=42

module load proxy/default
cd "$REPO"
source init.sh
cd "$REPO/examples/GradDotProd_LM"

# Reuse precomputed medians if present, else compute them.
[ -f "$RUN_DIR/sel50_medians.env" ] || python experiments/sel50/compute_medians.py --run_dir "$RUN_DIR"
# shellcheck disable=SC1090
source "$RUN_DIR/sel50_medians.env"
echo "[driver-shuf] DOT_MEDIAN=$DOT_MEDIAN  COS_MEDIAN=${COS_MEDIAN:-NA}  HAVE_COSINE=${HAVE_COSINE}  SHUF_SEED=$SHUF_SEED"

run_arm () {
  local name="$1" metric="$2" threshold="$3" invert="$4"
  local extra=""
  [ "$invert" = "1" ] && extra="--replay_filter_invert"
  echo "=== [arm] $name  metric=$metric  threshold=$threshold  invert=$invert  shuffle=1 ==="
  python main.py \
    --method Regular --architecture GPT2-Small \
    --batch_size 16 --val_batch_size 16 \
    --learning_rate 0.0006 --max_steps 5000 --warmup_step 2000 \
    --optimizer adamw --seed 42 \
    --model_dtype float32 --train_dtype bfloat16 \
    --eval_interval 200 --eval_iter 20 --eval_bs 16 \
    --replay_run_dir "$RUN_DIR" \
    --replay_filter_metric "$metric" --replay_filter_threshold "$threshold" $extra \
    --replay_shuffle --replay_shuffle_seed "$SHUF_SEED" \
    --wandb_run_name "$name"
}

run_arm sel_dot_top50_shuf dot_product "$DOT_MEDIAN" 0
run_arm sel_dot_bot50_shuf dot_product "$DOT_MEDIAN" 1
if [ "${HAVE_COSINE}" = "1" ]; then
  run_arm sel_cos_top50_shuf cosine "$COS_MEDIAN" 0
  run_arm sel_cos_bot50_shuf cosine "$COS_MEDIAN" 1
else
  echo "[driver-shuf] HAVE_COSINE=0 -> skipping cosine arms"
fi

echo "[driver-shuf] ALL ARMS DONE"
