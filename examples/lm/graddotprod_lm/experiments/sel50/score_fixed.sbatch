#!/bin/bash
#SBATCH --job-name=ghost-score-fixed
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:59
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# GradDotProd scoring run on the CORRECTED engine (tied wte/lm_head + Linear
# grad-norms fixed). Args:  $1 = wandb run name,  $2 = exclude patterns
# (comma-separated, e.g. "wte,lm_head"; empty for the full score),
# $3 = max_steps (default 10000).
set -euo pipefail
# Repo root: override with GHOST_REPO, else derive from the submit dir (sbatch from inside the repo).
REPO="${GHOST_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)}"
: "${REPO:?set GHOST_REPO to the GhostSuite repo root, or submit from inside the repo}"
RUN_NAME="$1"
EXCLUDE="${2:-}"
MAX_STEPS="${3:-10000}"
SEED="${4:-42}"

module load proxy/default
cd "$REPO"
uv sync && source .venv/bin/activate
cd "$REPO/examples/lm/graddotprod_lm"

EXCL_ARG=()
[ -n "$EXCLUDE" ] && EXCL_ARG=(--score_exclude_params "$EXCLUDE")
echo "[score_fixed] RUN_NAME=$RUN_NAME EXCLUDE='${EXCLUDE:-<none>}' MAX_STEPS=$MAX_STEPS SEED=$SEED"

python main.py \
  --method GradDotProd --architecture GPT2-Small \
  --batch_size 16 --val_batch_size 16 \
  --learning_rate 0.0006 --max_steps "$MAX_STEPS" --warmup_step 2000 \
  --optimizer adamw --seed "$SEED" \
  --model_dtype float32 --train_dtype bfloat16 \
  --dynamic_val_batch --log_grad_norms --score_val_from_eval_pool \
  --eval_interval "${EVAL_INTERVAL:-200}" --eval_iter "${EVAL_ITER:-256}" --eval_bs 16 \
  --dot_prod_save_interval 10 \
  "${EXCL_ARG[@]}" \
  --wandb_run_name "$RUN_NAME"

echo "[score_fixed] DONE run_dir=$REPO/examples/lm/graddotprod_lm/results/$RUN_NAME"
