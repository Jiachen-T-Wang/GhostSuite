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


# Compile-compatible Function paths. Both remove the eager backward hooks / lock / setattr, so
# torch.compile can regional-compile the model. GHOST_AUTOGRAD_FN fuses the dot-product into the
# joint graph (Phase 2); GHOST_DECOUPLED_FN keeps native backward + runs the dot-product in the
# decoupled 1b grouped post-backward pass (re-examination doc §3). GHOST_COMPILE_LOSS=1 supplies
# the compile-friendly cross-entropy for either.
_AUTOGRAD_FN = os.getenv("GHOST_AUTOGRAD_FN", "0") == "1"
_DECOUPLED_FN = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"
_FN_PATH = _AUTOGRAD_FN or _DECOUPLED_FN


class GhostTrainer(Trainer):
    """Trainer subclass that appends a fixed validation batch for ghost GradDotProd."""

    def __init__(self, job_config):
        if not job_config.ghost.enable:
            raise RuntimeError("GhostTrainer requires ghost.enable=true.")

        # Compile is only allowed on the graph-clean custom-Function path; the eager hook
        # engine still hard-disables it (its hooks force compiled autograd + graph breaks).
        # On the custom-Function path we DEFER compile: the model must be built uncompiled so
        # the ghost manager can monkeypatch the supported layers' forward FIRST, then we
        # regional-compile each block (attach-then-compile is the order Inductor functionalizes
        # the per-layer buffer mutations cleanly; compile-then-attach yields an invalid graph
        # output for the buffer copy_). See ghost_phase2_results.
        self._deferred_compile = False
        if not _FN_PATH:
            job_config.compile.enable = False
        elif job_config.compile.enable:
            logger.warning(
                "Ghost Function path with compile.enable=true: deferring regional compile "
                "until after the ghost manager is attached (and buffers warmed up)."
            )
            self._deferred_compile = True
            self._compile_config = job_config.compile
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

        # Now that the ghost manager has monkeypatched the supported layers' forward, apply
        # the deferred regional compile (per TransformerBlock) so each block's graph traces
        # the custom Functions and Inductor functionalizes the per-layer buffer writes. First
        # run an eager warmup forward+backward so all per-layer buffers are allocated OUTSIDE
        # the compiled graph (compile must not allocate/setattr inside the traced region).
        if self._deferred_compile:
            from torchtitan.models.llama3.infra.parallelize import apply_compile

            # Opt-in: trade recompute for activation memory in the min-cut partitioner, to cut
            # the in-graph-dot path's saved intermediates (the +38% peak-mem cost). 1.0 = save
            # everything (default); <1.0 = recompute more in backward. Set before compile.
            _mem_budget = os.getenv("GHOST_COMPILE_MEM_BUDGET")
            if _mem_budget:
                import torch._functorch.config as _fcfg
                _fcfg.activation_memory_budget = float(_mem_budget)
                logger.info("Set activation_memory_budget=%s", _mem_budget)

            self.ghost_helper.warmup_for_compile(
                train_local_batch_size=job_config.training.local_batch_size,
                seq_len=job_config.training.seq_len,
            )
            apply_compile(self.model_parts[0], self._compile_config)

            # Opt-in: also regional-compile the top-level layers (tok_embeddings / norm / output)
            # that apply_compile skips, so their ghost in-graph dot folds into a compiled region
            # instead of running eager. Each layer type is independently gated for the generality
            # study (a win on `output` does not imply a win on `tok_embeddings`).
            #   GHOST_COMPILE_TOPLEVEL=1 -> all three (unless a per-component flag overrides)
            #   GHOST_COMPILE_EMB / GHOST_COMPILE_NORM / GHOST_COMPILE_OUTPUT = 0/1 -> per layer
            _toplevel = os.getenv("GHOST_COMPILE_TOPLEVEL", "0") == "1"

            def _flag(name: str) -> bool:
                v = os.getenv(name)
                return _toplevel if v is None else (v == "1")

            _emb = _flag("GHOST_COMPILE_EMB")
            _norm = _flag("GHOST_COMPILE_NORM")
            _output = _flag("GHOST_COMPILE_OUTPUT")
            if _emb or _norm or _output:
                from torchtitan.models.llama3.infra.parallelize import (
                    apply_compile_top_level,
                )

                apply_compile_top_level(
                    self.model_parts[0],
                    self._compile_config,
                    compile_emb=_emb,
                    compile_norm=_norm,
                    compile_output=_output,
                )
            logger.info("Applied deferred regional compile after ghost attach + warmup.")

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

        if self.ghost_helper.use_fn_path:
            # Function paths: no saved_tensors_hooks; dot-products land in per-layer buffers
            # (autograd-fn) or are computed from captured (A, B) post-backward (decoupled).
            with self.train_context(None):
                with self.maybe_enable_amp:
                    pred = self.model_parts[0](combined_input["input"])
                    loss = self.loss_fn(pred, combined_labels)
                del pred
                loss.backward()
            self.ghost_helper.collect_step_dot()
            return loss

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
        if self.ghost_helper.use_fn_path:
            if self.gradient_accumulation_steps != 1:
                raise RuntimeError(
                    "Ghost Function paths currently require gradient_accumulation_steps == 1 "
                    "(buffers hold only the last microbatch's grad_val). "
                    f"Got {self.gradient_accumulation_steps}."
                )
            self.ghost_helper.prepare_gradients_fn()
        else:
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
