# Plan: Filtered Replay Training

Goal: add a training option that replays data from a previous GradDotProd run while dropping “low-value” samples (negative gradient dot product or cosine similarity) and preserving the original sample order.

1) Map previous-run artifacts and constraints
- Use `grad_dotprods/dot_prod_log_iter_*.pt` entries (`dot_product`, `train_grad_norm`, `val_grad_norm`, `X_train`, `Y_train`, `batch_idx`, `iter_num`) plus `grad_dotprods/valset.pt` as the source of replay data and validation batch.
- Confirm ordering by sorting files by iteration number and respecting each entry’s `iter_num` and in-batch order; record any assumptions (e.g., fixed val batch, consistent batch size).

2) Surface configuration to enable replay + filtering
- Add CLI flags in `Examples/GradDotProd_LM/config_file.py` (propagated into `TrainingConfig`) for `--replay_run_dir` (path to a past run), `--replay_filter_metric` (`dot_product`|`cosine`), `--replay_filter_threshold` (default 0.0, drop values below), and `--replay_drop_last`/`--replay_rebatch_size` if we need to control batching of filtered samples.
- Record these settings in the result directory for reproducibility.

3) Implement a log-backed replay loader
- New helper (e.g., `Examples/shared/replay_loader.py`) to stream dot-product logs in iteration order, compute the chosen metric, and filter out samples below the threshold.
- Preserve per-sample order: flatten batches in their original sequence after filtering while keeping associated metadata (`batch_idx`, `iter_num`) for debugging.
- Decide on data source: default to stored `X_train`/`Y_train` tensors to guarantee fidelity; optionally allow rehydrating from the raw dataset via `batch_idx` if desired for memory savings.
- Support lazy loading to avoid holding all logs in memory; expose an iterator that yields contiguous batches respecting `replay_rebatch_size`/`drop_last`.

4) Wire replay loader into training
- In `Examples/shared/training_utils.py` (or a thin wrapper), add a `get_batch` path that pulls from the replay loader when `config.replay_run_dir` is set; handle exhaustion by shortening `max_steps` or terminating training cleanly.
- Feed the saved `valset.pt` into `GhostEngineManager` so GradDotProd still has a consistent validation batch; fall back gracefully if the file is missing.
- Ensure DDP compatibility: shard the ordered sample stream by rank/world_size (e.g., stride-based slicing) without reordering within each shard.

5) Filtering semantics and validation
- Define cosine similarity as `dot_product / (train_grad_norm * val_grad_norm)`; error or warn if the required norms are absent when `cosine` is requested.
- Log counts of kept vs. dropped samples and examples of metric ranges to help users verify the filter effect.

6) Documentation and guardrails
- Update `Examples/GradDotProd_LM/README.md` (or a short HOWTO) to describe replay mode usage, required inputs, and limitations.
- Add lightweight checks/tests (e.g., a small fixture log file) to assert that filtering and ordering behave as expected and that training exits gracefully when the replay stream is exhausted.
