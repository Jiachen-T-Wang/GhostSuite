import os
from typing import Dict, Tuple

import torch

from ghostEngines import GradDotProdEngine
from ghostEngines.decoupled_capture_dotprod import GhostDecoupledManager
from torchtitan.hf_datasets.text_datasets import build_text_validation_dataloader
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


# Decoupled in-graph path (re-examination doc §3): a transparent identity Function keeps each
# layer's native backward, the model regional-compiles, and the per-sample dot-product is computed
# as a small in-backward transient. Uses the deferred-compile trainer wiring (attach -> warmup ->
# compile); train grads are recovered via subtract-val. When off, the eager GradDotProdEngine runs.
_DECOUPLED_FN = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"


class GhostDotProdHelper:
    """Utility wrapper to manage GradDotProdEngine lifecycle for TorchTitan runs."""

    def __init__(
        self,
        job_config: JobConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer,
        parallel_dims: ParallelDims,
        device: torch.device,
    ) -> None:
        self.job_config = job_config
        self.ghost_cfg = job_config.ghost
        self.device = device
        self.parallel_dims = parallel_dims

        self._assert_supported_parallelism()

        val_input_dict, val_labels = self._load_val_batch(tokenizer)
        self.val_input_dict = val_input_dict
        self.val_labels = val_labels
        self.val_batch_size = val_labels.shape[0]

        save_dir = self.ghost_cfg.save_dir or os.path.join(
            job_config.job.dump_folder, "ghost_dotprods"
        )

        self.use_decoupled_fn = _DECOUPLED_FN
        # The decoupled Function path uses the trainer's fn-path wiring (no saved_tensors_context;
        # dot-products collected after backward; subtract-val recovery before the optimizer step).
        self.use_fn_path = _DECOUPLED_FN
        if self.use_decoupled_fn:
            # Persistence is not wired on the fn-path (follow-up:
            # docs/issues/open/titan-fn-path-never-persists-dots_2026-07-02.md), so save_dir is
            # not created here — only the eager engine below writes to it.
            if self.ghost_cfg.save_interval > 0:
                logger.warning(
                    "Ghost fn-path (decoupled_fn=true): dot-product score PERSISTENCE IS NOT "
                    "WIRED in the torchtitan integration — scores are computed each step but "
                    "NOT saved (ghost.save_interval=%d and ghost.save_dir=%s are ignored). "
                    "Set GHOST_DUMP_DOTPROD=<dir> for debug dumps, or run with "
                    "--ghost.no-decoupled_fn to use the eager engine, which persists to "
                    "save_dir.",
                    self.ghost_cfg.save_interval,
                    save_dir,
                )
            self.fn_manager = GhostDecoupledManager(model, val_batch_size=self.val_batch_size)
            self.fn_manager.attach()
            self.engine = None
            self.dot_products = []
        else:
            os.makedirs(save_dir, exist_ok=True)
            self.fn_manager = None
            self.engine = GradDotProdEngine(
                module=model,
                val_batch_size=self.val_batch_size,
                loss_reduction="mean",
                use_dummy_bias=self.ghost_cfg.use_dummy_bias,
                dot_prod_save_path=save_dir,
                log_grad_norms=self.ghost_cfg.log_grad_norms,
            )
            self.engine.attach(optimizer)

    def _assert_supported_parallelism(self) -> None:
        if (
            getattr(self.parallel_dims, "pp", 1) > 1
            or getattr(self.parallel_dims, "tp", 1) > 1
            or getattr(self.parallel_dims, "cp", 1) > 1
        ):
            raise RuntimeError("Ghost GradDotProd is limited to single-stage, non-parallel training for now.")
        dp_world = getattr(self.parallel_dims, "dp_replicate", 1) * getattr(self.parallel_dims, "dp_shard", 1)
        if dp_world > 1:
            raise RuntimeError("Ghost GradDotProd currently supports only single-GPU runs.")

    def _load_val_batch(self, tokenizer) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Load a single validation batch and move it to the training device."""
        dataloader = build_text_validation_dataloader(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            job_config=self.job_config,
            infinite=False,
        )
        val_iter = iter(dataloader)
        try:
            input_dict, labels = next(val_iter)
        except StopIteration as ex:
            raise RuntimeError("Validation dataloader is empty; cannot build ghost val batch.") from ex

        for k, v in input_dict.items():
            input_dict[k] = v.to(self.device)
        labels = labels.to(self.device)

        if self.ghost_cfg.val_batch_size > 0 and labels.shape[0] != self.ghost_cfg.val_batch_size:
            # Align engine val_batch_size with actual batch for safety.
            self.ghost_cfg.val_batch_size = labels.shape[0]

        return input_dict, labels

    def attach_train_batch(
        self,
        train_input: torch.Tensor,
        train_labels: torch.Tensor,
        iter_num: int,
        batch_idx: int,
    ) -> None:
        if self.use_fn_path:
            return
        self.engine.attach_train_batch(train_input, train_labels, iter_num, batch_idx=batch_idx)

    # -- Decoupled Function path helpers -------------------------------------------------

    def warmup_for_compile(self, train_local_batch_size: int, seq_len: int) -> None:
        """Run one eager forward+backward to populate the per-layer dot/grad_val buffers BEFORE
        torch.compile traces the model (compile must not allocate buffers inside the graph).
        Uses a combined train+val token batch matching the real training shape."""
        combined_bs = train_local_batch_size + self.val_batch_size
        example = torch.randint(
            0, 1, (combined_bs, seq_len), device=self.device, dtype=torch.long
        )
        self.fn_manager.warmup(example)

    def begin_step(self) -> None:
        """Reset per-step dot/grad accumulation before the gradient-accumulation microbatch loop.

        fn-path: clears the decoupled manager's per-step accumulators. Eager: no-op (the engine's
        accumulator is cleared by its subtract-val recovery)."""
        if self.use_fn_path and self.fn_manager is not None:
            self.fn_manager.begin_step()

    def collect_step_dot(self) -> None:
        """After each microbatch's backward: read that microbatch's per-train-sample dot-product
        from the in-graph buffers and fold its val grad into the per-step accumulator (for the single
        subtract-val recovery in prepare_gradients_fn). Must run before the next backward overwrites
        the buffers."""
        dot = self.fn_manager.collect_microbatch_dot()
        if dot is not None:
            self.dot_products.append(dot.detach())

    def prepare_gradients_fn(self) -> None:
        """subtract-val recovery of train grads from autograd .grad + buffered grad_val."""
        self.fn_manager.recover_train_grads()

    def combine_with_val(
        self, train_input_dict: Dict[str, torch.Tensor], train_labels: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        combined_inputs: Dict[str, torch.Tensor] = {}
        for k, v in train_input_dict.items():
            if k not in self.val_input_dict:
                raise KeyError(f"Validation batch missing key '{k}' required by train batch.")
            combined_inputs[k] = torch.cat([v, self.val_input_dict[k]], dim=0)
        combined_labels = torch.cat([train_labels, self.val_labels], dim=0)
        return combined_inputs, combined_labels

    def aggregate_and_maybe_save(self, iter_num: int, skip_aggregation: bool = False) -> None:
        if self.use_fn_path:
            # Debug-only: dump the per-step aggregated dot-product so AC modes can be compared
            # against the no-AC compiled path (Phase 0 correctness guard for the AC-frontier
            # study). Gated by GHOST_DUMP_DOTPROD=<dir>; off by default.
            _dump = os.getenv("GHOST_DUMP_DOTPROD")
            if _dump and self.dot_products:
                os.makedirs(_dump, exist_ok=True)
                torch.save(
                    self.dot_products[-1].float().cpu(),
                    os.path.join(_dump, f"dot_iter_{iter_num}.pt"),
                )
            # Keep the dot-product log bounded; persistence is out of scope for the compile
            # benchmark path (correctness is validated by the equivalence test).
            if len(self.dot_products) > 8:
                self.dot_products = self.dot_products[-8:]
            return
        if not skip_aggregation:
            self.engine.aggregate_and_log()

        if not self.ghost_cfg.save_train_batch:
            for entry in self.engine.dot_product_log:
                entry.pop("X_train", None)
                entry.pop("Y_train", None)

        if self.ghost_cfg.save_interval > 0 and iter_num % self.ghost_cfg.save_interval == 0:
            self.engine.save_dot_product_log(iter_num=iter_num)

        # Avoid unbounded growth if user disables saving.
        if self.engine.dot_product_log:
            self.engine.dot_product_log.clear()

        self.engine.clear_gradients()

    def detach(self) -> None:
        if self.use_fn_path:
            if self.fn_manager is not None:
                self.fn_manager.detach()
            return
        if self.engine:
            self.engine.detach()
