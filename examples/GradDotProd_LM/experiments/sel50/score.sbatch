#!/bin/bash
#SBATCH --job-name=ghost-score10k
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:59
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err

# Job A: GradDotProd scoring run, 10,000 steps, with per-sample grad norms logged
# (so both dot-product and cosine selection are available downstream). The
# train-only gradient is recovered before the optimizer step, so this run's
# first 5,000 steps double as the matched random-data control.
set -euo pipefail

REPO=/scratch/gpfs/PMITTAL/tianhao/GhostSuite
RUN_NAME=score10k_sel50_seed42

module load proxy/default
cd "$REPO"
source init.sh
cd "$REPO/examples/GradDotProd_LM"

python main.py \
  --method GradDotProd --architecture GPT2-Small \
  --batch_size 16 --val_batch_size 16 \
  --learning_rate 0.0006 --max_steps 10000 --warmup_step 2000 \
  --optimizer adamw --seed 42 \
  --model_dtype float32 --train_dtype bfloat16 \
  --dynamic_val_batch --log_grad_norms \
  --eval_interval 200 --eval_iter 20 --eval_bs 16 \
  --dot_prod_save_interval 10 \
  --wandb_run_name "$RUN_NAME"

echo "[score.sbatch] DONE run_dir=$REPO/examples/GradDotProd_LM/results/$RUN_NAME"
