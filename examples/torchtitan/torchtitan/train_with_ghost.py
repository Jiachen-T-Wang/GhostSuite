import os
import sys
from typing import Iterable, Tuple

import torch

# Ensure repo root is on PYTHONPATH for ghostEngines import.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torchtitan.train import Trainer
from torchtitan.ghost.dotprod_helper import GhostDotProdHelper
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.tools.logging import init_logger, logger


class GhostTrainer(Trainer):
    """Trainer subclass that appends a fixed validation batch for ghost GradDotProd."""

    def __init__(self, job_config):
        if not job_config.ghost.enable:
            raise RuntimeError("GhostTrainer requires ghost.enable=true.")

        # Disable compile for compatibility with ghost hooks.
        job_config.compile.enable = False

        # Route validation loader to ghost config; optional validation loop stays controlled by ghost.enable_validation.
        job_config.validation.enable = job_config.ghost.enable_validation
        job_config.validation.dataset = job_config.ghost.val_dataset
        job_config.validation.dataset_path = job_config.ghost.val_dataset_path
        job_config.validation.local_batch_size = job_config.ghost.val_batch_size
        if job_config.validation.seq_len is None:
            job_config.validation.seq_len = job_config.training.seq_len

        super().__init__(job_config)

        if len(self.model_parts) != 1:
            raise RuntimeError("GhostTrainer currently supports single model part (no pipeline parallelism).")

        self.ghost_helper = GhostDotProdHelper(
            job_config=job_config,
            model=self.model_parts[0],
            optimizer=self.optimizers,
            tokenizer=self.tokenizer,
            parallel_dims=self.parallel_dims,
            device=self.device,
        )

        logger.info(
            "Ghost GradDotProd enabled | val_batch_size=%d | save_interval=%d | save_train_batch=%s",
            self.ghost_helper.val_batch_size,
            job_config.ghost.save_interval,
            job_config.ghost.save_train_batch,
        )
        logger.info(
            "Note: MFU will still include extra ghost dot-product compute as overhead "
            "since num_flops_per_token is model-only; expect MFU closer but not necessarily matching baseline."
        )

    def forward_backward_step(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        microbatch_idx: int = 0,
    ) -> torch.Tensor:
        """Override to append fixed validation batch and run ghost hooks."""
        # No parallel contexts supported in ghost mode.
        combined_input, combined_labels = self.ghost_helper.combine_with_val(input_dict, labels)
        # Include validation tokens in throughput/MFU metrics for ghost runs.
        self.metrics_processor.ntokens_since_last_log += (
            combined_labels.numel() - labels.numel()
        )

        self.ghost_helper.attach_train_batch(
            train_input=input_dict["input"],
            train_labels=labels,
            iter_num=self.step,
            batch_idx=microbatch_idx,
        )

        with self.ghost_helper.engine.saved_tensors_context():
            with self.train_context(None):
                with self.maybe_enable_amp:
                    pred = self.model_parts[0](combined_input["input"])
                    loss = self.loss_fn(pred, combined_labels)
                del pred
                loss.backward()

        # Aggregate dot products per microbatch before the next backward pass.
        self.ghost_helper.engine.aggregate_and_log()
        return loss

    def train_step(
        self, data_iterator: Iterable[Tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        accumulated_losses = []
        for microbatch_idx in range(self.gradient_accumulation_steps):
            try:
                input_dict, labels = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            loss = self.forward_backward_step(input_dict, labels, microbatch_idx=microbatch_idx)
            accumulated_losses.append(loss.detach())

        # Move accumulated train grads into .grad for optimizer step.
        self.ghost_helper.engine.prepare_gradients()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Persist ghost metrics and clear stored gradients.
        self.ghost_helper.aggregate_and_maybe_save(self.step, skip_aggregation=True)

        loss = torch.sum(torch.stack(accumulated_losses))

        if not self.metrics_processor.should_log(self.step):
            return

        global_avg_loss = global_max_loss = loss.detach().item()
        global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    def close(self) -> None:
        if hasattr(self, "ghost_helper"):
            self.ghost_helper.detach()
        super().close()


def main():
    from torchtitan.config import ConfigManager
    from torchtitan.train import run_with_config

    def _normalize_ghost_enable_args(args: list[str]) -> list[str]:
        normalized: list[str] = []
        true_values = {"1", "true", "yes", "y", "on"}
        false_values = {"0", "false", "no", "n", "off"}
        idx = 0
        while idx < len(args):
            arg = args[idx]
            if arg.startswith("--ghost.enable="):
                value = arg.split("=", 1)[1].strip().lower()
                if value in false_values:
                    normalized.append("--ghost.no-enable")
                elif value in true_values:
                    normalized.append("--ghost.enable")
                else:
                    normalized.append(arg)
                idx += 1
                continue
            if arg == "--ghost.enable" and idx + 1 < len(args):
                value = args[idx + 1].strip().lower()
                if value in false_values:
                    normalized.append("--ghost.no-enable")
                    idx += 2
                    continue
                if value in true_values:
                    normalized.append("--ghost.enable")
                    idx += 2
                    continue
            normalized.append(arg)
            idx += 1
        return normalized

    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    args = _normalize_ghost_enable_args(sys.argv[1:])
    config = ConfigManager().parse_args(args)

    trainer_class = GhostTrainer if config.ghost.enable else Trainer
    if not config.ghost.enable:
        logger.info("Ghost disabled; using standard Trainer.")

    run_with_config(trainer_class, config)


if __name__ == "__main__":
    main()
