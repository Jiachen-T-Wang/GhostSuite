#!/bin/bash
#SBATCH --job-name=ghost-sel50-replay2
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:59
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err

# Replay driver for the corrected scores. Args:
#   $1 = scoring run name (under results/)
#   $2 = arm prefix (e.g. fixfull / fixexcl) used in the arm run names
# Computes dot & cosine global medians, then trains four matched 5k-step Regular
# arms: {dot,cosine} x {top-50%, bottom-50%}. Same training configs as scoring.
set -euo pipefail
REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite
SCORE_NAME="$1"
PREFIX="$2"
RUN_DIR="$REPO/examples/GradDotProd_LM/results/$SCORE_NAME"

module load proxy/default
cd "$REPO"
source init.sh
cd "$REPO/examples/GradDotProd_LM"

python experiments/sel50/compute_medians.py --run_dir "$RUN_DIR"
# shellcheck disable=SC1090
source "$RUN_DIR/sel50_medians.env"
echo "[driver:$PREFIX] DOT_MEDIAN=$DOT_MEDIAN COS_MEDIAN=${COS_MEDIAN:-NA} HAVE_COSINE=${HAVE_COSINE}"

run_arm () {
  local name="$1" metric="$2" threshold="$3" invert="$4"
  local extra=""
  [ "$invert" = "1" ] && extra="--replay_filter_invert"
  echo "=== [arm] $name metric=$metric threshold=$threshold invert=$invert ==="
  python main.py \
    --method Regular --architecture GPT2-Small \
    --batch_size 16 --val_batch_size 16 \
    --learning_rate 0.0006 --max_steps 5000 --warmup_step 2000 \
    --optimizer adamw --seed 42 \
    --model_dtype float32 --train_dtype bfloat16 \
    --eval_interval 200 --eval_iter 20 --eval_bs 16 \
    --replay_run_dir "$RUN_DIR" \
    --replay_filter_metric "$metric" --replay_filter_threshold "$threshold" $extra \
    --wandb_run_name "$name"
}

run_arm "${PREFIX}_dot_top50" dot_product "$DOT_MEDIAN" 0
run_arm "${PREFIX}_dot_bot50" dot_product "$DOT_MEDIAN" 1
if [ "${HAVE_COSINE}" = "1" ]; then
  run_arm "${PREFIX}_cos_top50" cosine "$COS_MEDIAN" 0
  run_arm "${PREFIX}_cos_bot50" cosine "$COS_MEDIAN" 1
fi
echo "[driver:$PREFIX] ALL ARMS DONE"
