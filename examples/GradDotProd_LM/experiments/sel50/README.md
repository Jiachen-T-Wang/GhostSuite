# sel50: gradient-dot-product data selection

Validation loss for training on **random** vs **gradient-dot-product–selected**
data (GPT2-Small, Pile; 20k-step scoring → 10k-step training). Curves below are
seed 42 with a 4096-window validation pool.

![Validation loss vs step (seed 42, 4096-window pool)](sel50_s42_4096_val_curves_2026-06-24.png)

## Launch

```bash
cd examples/GradDotProd_LM

# 1. Scoring runs (GradDotProd, 20k steps; 4096-window eval/scoring pool = EVAL_ITER*16)
full=$(sbatch --parsable --export=ALL,EVAL_ITER=256 \
  experiments/sel50/score_fixed.sbatch score20kpool4096_full_s42 ""            20000 42)
excl=$(sbatch --parsable --export=ALL,EVAL_ITER=256 \
  experiments/sel50/score_fixed.sbatch score20kpool4096_excl_s42 "wte,lm_head" 20000 42)

# 2. Selection arms (top-50% + positive, 10k steps) — run after each scoring job
sbatch --export=ALL,EVAL_ITER=256 --dependency=afterok:$full \
  experiments/sel50/select_driver.sbatch score20kpool4096_full_s42 sel4096_full_s42 10000 42
sbatch --export=ALL,EVAL_ITER=256 --dependency=afterok:$excl \
  experiments/sel50/select_driver.sbatch score20kpool4096_excl_s42 sel4096_excl_s42 10000 42
```
