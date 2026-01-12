import os
from typing import Dict, Tuple

import torch

from ghostEngines import GradDotProdEngine
from torchtitan.hf_datasets.text_datasets import build_text_validation_dataloader
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims


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
        os.makedirs(save_dir, exist_ok=True)

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
        self.engine.attach_train_batch(train_input, train_labels, iter_num, batch_idx=batch_idx)

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
        if self.engine:
            self.engine.detach()
