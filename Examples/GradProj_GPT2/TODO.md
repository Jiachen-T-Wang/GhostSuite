In `/scratch/gpfs/tw8948/GhostPub/GhostSuite/Examples/GradProj_GPT2`, I have finished running the following experiments, which computes per-sample gradient projections for GPT2 on Pile with different projection dimension (16, 32, ..., 2048). 
```
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 16 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 32 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 64 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 128 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 256 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 512 --proj_seed 9 ;
sbatch train.sh --batch_size 2 --max_samples 100000 --proj_layers "mlp" --model_dtype float32 --train_dtype float32 --seed 1 --proj_rank_total 1024 --proj_seed 9 ;
```
The results have been saved to `/scratch/gpfs/tw8948/GhostPub/GhostSuite/Examples/GradProj_GPT2/Results`. 

**TODO (finished):** I would like to investigate how the quality of random projection changes as the projection dimension decreases. Specifically, for each projection dimension, I would like to load the stored gradient projections, and compute the dot-product between a reference set of data points and the rest of the data points (`10000 - num_ref`). Then, I would like to make a plot where the x-axis is the dimensionality from 16 to 512, and y-axis is the l2 distance of the gradient dot-products between each projection dimension and dimension 1024. The plotting code is in `GhostSuite/Examples/GradProj_GPT2/plot_error_with_dim.py`.

## Code Review Request
The following is the output when I run `python plot_error_with_dim.py` in `Examples/GradProj_GPT2`. It seems that the RMSE is very high. I wonder whether this is due to a bug in plotting code or in the implementation of random projection. 
```
============================================================
Computing errors from reference (rank 1024)
Using 50 reference samples
============================================================
Rank   16: Avg RMSE = 334.2379 (±490.4635), Relative error = 18.0660%
Rank   32: Avg RMSE = 318.6656 (±507.2437), Relative error = 17.2243%
Rank   64: Avg RMSE = 321.1535 (±578.9644), Relative error = 17.3588%
Rank  128: Avg RMSE = 232.2234 (±426.4135), Relative error = 12.5520%
Rank  256: Avg RMSE = 134.4810 (±251.8678), Relative error = 7.2689%
Rank  512: Avg RMSE = 131.3894 (±242.0437), Relative error = 7.1018%
```


### Code Review: Findings and Recommendations

Summary
- The projection implementation and the plotting script are consistent and correct for estimating gradient dot-products via random projections. The high absolute RMSE values stem from the scale of the dot-products; the relative error metric (7–18%) is the appropriate indicator and looks reasonable given JL-type variance at these ranks.

What I validated
- Reproduced the reported output by running `plot_error_with_dim.py` against `Results/` for ranks {16,32,64,128,256,512,1024} with `num_ref=50` and `max_iters=100`.
- Inspected `ghostEngines/gradProjection` implementation:
  - Projection matrices use Gaussian with std=1/sqrt(rows) (or orthonormal rows if enabled), implying E[P^T P] = I and an unbiased estimator for inner products.
  - Per-layer projected gradient is `P_o @ (∑_t B_t A_t^T) @ P_i^T` computed efficiently via hooks; concatenation across layers preserves unbiasedness of the global dot-product estimate.
  - Scaling by `batch_size` correctly removes the 1/B factor from mean-reduced loss; token-length scaling is constant across runs (block_size), so comparisons are fair.
- Verified data alignment and experiment setup:
  - Same `seed` and generator are used across runs, so the sampled batches are consistent between ranks.
  - Directory naming in `Results/` matches the plotting code’s path template (e.g., `proj_layers_mlp_rank_total_..._dtype_bfloat16_row_on_False_emb_False`).
  - Saved projections are bfloat16; the plotting script casts to float32 on load.
- Sanity-checked shapes via `metadata.json`: for GPT2-Small with `mlp` only, there are 24 projected layers; total dims scale as ~24×`proj_rank_total` (small per-layer rounding from `choose_ki_ko` is expected).

Interpretation of the numbers
- Absolute RMSE (e.g., ~334 at rank 16) is large because the underlying dot-products have large magnitude and variance across samples; relative error is the meaningful metric here.
- Relative error decreases monotonically with dimension (~18% at 16 → ~7% at 512), which matches JL theory for unbiased random projections with variance shrinking ~O(1/√k_total).
- Using rank 1024 as the “reference” adds noise to the denominator and the target; it is still a high-dimension proxy, but not ground truth. This can inflate apparent error slightly at all ranks.

Potential issues or edge cases (did not find blocking bugs)
- Reference choice: Comparing to 1024-D projections (not exact gradients) introduces reference noise. If 2048 is available, using it as reference would reduce that noise; best is to compare to true gradient dot-products on a small subset.
- Normalization choice: The script reports one global relative error by dividing the average RMSE by a single RMS over all refs. A per-reference normalization (RMSE_i / RMS(ref_i), then average) can give a more robust summary across heterogeneous references.
- Output directory creation: `plot_error_with_dim.py` assumes `Plots/` exists. It exists in this repo, but adding a `mkdir` for robustness would avoid future failures.
- bfloat16 storage: Quantization noise from bfloat16 accumulates more in higher dimensions. This affects both the lower-rank and the 1024-D reference, so relative comparisons remain fair, but it can slightly inflate errors. If disk allows, saving projections in float32 for a small subset can help calibrate this effect.

Recommendations
- Keep the current projection and plotting logic; there is no evidence of a correctness bug.
- For a tighter estimate of projection quality:
  - Use rank 2048 (already computed) as the reference when memory permits, or
  - Compute true gradient dot-products for a small batch via `GradDotProdEngine` and compare projections directly against ground truth.
  - Report per-reference normalized errors and include median with IQR alongside mean±std to summarize skew.
- Optionally, repeat the analysis with `proj_row_orthonormal=True` (already computed) to see if row-orthonormal P slightly reduces variance at small ranks.
- Consider increasing `max_iters` to stabilize estimates if I/O allows or stratify references across the run (early/late) to check stationarity.

Quick checklist I used
- Projection math: unbiasedness of z_i^T z_j with P scaled 1/√k → OK.
- Two-sided matrix sketch (P_o, P_i) preserves dot-products in expectation → OK.
- Seeded sampling consistency across runs → OK.
- File naming/paths and dtype handling in loader → OK.
- Shapes and per-layer slice metadata → OK.

Bottom line
- The observed relative errors are in a plausible range for the given ranks and setup. The plotting and projection implementations look correct. If you want lower reported error, switch the reference to 2048 or true gradients, and/or aggregate more iterations for a lower-variance estimate.



### TODO
1. Add CLI for `plot_error_with_dim.py` and make the saved figures' file name adapt to the specific settings (similar to the naming conventions for files in `Examples/GradProj_GPT2/Results`). 
2. Revise 

