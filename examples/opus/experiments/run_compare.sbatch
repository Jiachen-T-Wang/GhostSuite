#!/bin/bash
#SBATCH --job-name=opus-pre-cmp
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output=slurm-%x-%j.out

# OPUS pretraining comparison arms: GPT2-Small on Pile, the EXACT protocol of the committed
# GREATS experiment (examples/greats/pretrain/experiments/run_compare.sbatch) — equal update
# size k=16 selected from a 32-candidate pool, m=16 proxy from the eval window pool, lr 6e-4
# cosine / warmup 2000, bf16 train, seed 42, eval every 500 (200 iters) — so the resulting
# curves are directly comparable against the committed Regular / GREATS logs. Arms:
#   sbatch experiments/run_compare.sbatch stochastic <MAX_STEPS>
#       OPUS default: Boltzmann stochastic-greedy, adamw_scalar preconditioner,
#       temperature 1e-9 — calibrated to the observed score scale at this setup
#       (pilot: score std ~1.1e-9 under adamw_scalar; upstream's T=0.9 is tuned to
#       their GPT2-XL/Muon score scale and would sample ~uniformly here).
#   sbatch experiments/run_compare.sbatch greedy     <MAX_STEPS>
#       Deterministic greedy with the Gram redundancy penalty in RAW gradient units
#       (preconditioner none; pilot: score ~2e-2, Gram diag ~0.3 — the penalty is
#       meaningfully scaled). This is GREATS + the second-order diversity term.
set -e
ARM=${1:-stochastic}
MAX_STEPS=${2:-20000}

module load proxy/default 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Point these at your tokenized Pile splits (already-set env values are respected).
export PILE_DATA_DIR_TRAIN=${PILE_DATA_DIR_TRAIN:-/path/to/pile/pile-train}
export PILE_DATA_DIR_VAL=${PILE_DATA_DIR_VAL:-/path/to/pile/pile-val-gpt2}
export PILE_DATA_DIR_TEST=${PILE_DATA_DIR_TEST:-/path/to/pile/pile-test-gpt2}

# Resolve the repo root (dir containing init.sh) from the submit directory.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$REPO" != "/" ] && [ ! -f "$REPO/init.sh" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/init.sh" ] || { echo "ERROR: could not locate repo root (init.sh) from ${SLURM_SUBMIT_DIR:-$(pwd)}"; exit 1; }
cd "$REPO"
source init.sh >/dev/null 2>&1
cd "$REPO/examples/opus"

ARGS="--architecture GPT2-Small --train_set pile --val_set pile \
  --batch_size 16 --candidate_batch_size 32 --val_batch_size 16 \
  --learning_rate 6e-4 --warmup_step 2000 --max_steps $MAX_STEPS \
  --eval_interval 500 --eval_iter 200 --eval_bs 16 \
  --model_dtype float32 --train_dtype bfloat16 --seed 42 \
  --proj_dim 8192 \
  --wandb_run_name OPUS_${ARM}_pre"

case "$ARM" in
    stochastic)
        ARGS="$ARGS --method OPUS --opus_selection_method stochastic \
              --opus_preconditioner adamw_scalar --opus_temperature 1e-9";;
    greedy)
        ARGS="$ARGS --method OPUS --opus_selection_method greedy \
              --opus_preconditioner none";;
    *) echo "Unknown arm: $ARM (stochastic|greedy)"; exit 1;;
esac

echo "Running: python main.py $ARGS"
python -u main.py $ARGS
