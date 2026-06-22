import os
import sys
from typing import Iterable, Tuple

import torch

# Ensure repo root is on PYTHONPATH for ghostEngines import.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torchtitan.train import Trainer
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.tools.logging import init_logger, logger

# NOTE: GhostDotProdHelper (and the ghostEngines it pulls in) is imported lazily inside
# GhostTrainer.__init__ — AFTER the ghost lever config is bridged to GHOST_* env vars — because
# ghostEngines reads those env vars at import time. Importing it at module top would freeze the
# lever flags before the config/CLI values are applied. For the same reason the Function-path
# selectors (GHOST_AUTOGRAD_FN / GHOST_DECOUPLED_FN) are read inside __init__ after the bridge,
# not at module top.


class GhostTrainer(Trainer):
    """Trainer subclass that appends a fixed validation batch for ghost GradDotProd."""

    def __init__(self, job_config):
        if not job_config.ghost.enable:
            raise RuntimeError("GhostTrainer requires ghost.enable=true.")

        # Bridge the dot-product lever config (--ghost.subtract_val / .batched_dotprod /
        # .batched_dotprod_compile / .decoupled_fn / .compile_toplevel / .regional_compile) to the
        # GHOST_* env vars the engine reads at import time. An explicitly-set env var wins (so
        # ad-hoc `GHOST_*=...` runs still work); otherwise the config value is applied. Must run
        # before the lazy GhostDotProdHelper import AND before the Function-path reads below.
        self._bridge_ghost_levers(job_config.ghost)

        # Read the compile-compatible Function-path selectors AFTER bridging. Both remove the
        # eager backward hooks so torch.compile can regional-compile the model. GHOST_AUTOGRAD_FN
        # fuses the dot-product into the joint graph (Phase 2); GHOST_DECOUPLED_FN keeps native
        # backward + the decoupled 1b grouped post-backward dot-product (the default fast path).
        autograd_fn = os.getenv("GHOST_AUTOGRAD_FN", "0") == "1"
        decoupled_fn = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"
        fn_path = autograd_fn or decoupled_fn

        # Compile is only allowed on the graph-clean custom-Function path; the eager hook
        # engine still hard-disables it (its hooks force compiled autograd + graph breaks).
        # On the custom-Function path we DEFER compile: the model must be built uncompiled so
        # the ghost manager can monkeypatch the supported layers' forward FIRST, then we
        # regional-compile each block (attach-then-compile is the order Inductor functionalizes
        # the per-layer buffer mutations cleanly; compile-then-attach yields an invalid graph
        # output for the buffer copy_). See ghost_phase2_results.
        self._deferred_compile = False
        if not fn_path:
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

        # Lazy import: ghostEngines reads GHOST_* at import time, so this must come after
        # _bridge_ghost_levers() (called at the top of __init__).
        from torchtitan.ghost.dotprod_helper import GhostDotProdHelper

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

            # Opt-in: also regional-compile the top-level layers that apply_compile skips, so their
            # ghost in-graph dot folds into a compiled region instead of running eager.
            # Per-layer attribution (docs/investigations/ghost_outemb_plus_ac_2026-06-19.md, Part 4)
            # found the entire gain comes from the `output` Linear (+1.9%); compiling `norm` /
            # `tok_embeddings` adds nothing (within noise) and only lengthens warmup. So
            # GHOST_COMPILE_TOPLEVEL=1 defaults to OUTPUT ONLY; emb/norm stay off unless explicitly
            # requested via their per-component flags (kept for the generality study).
            #   GHOST_COMPILE_TOPLEVEL=1 -> output only
            #   GHOST_COMPILE_EMB / GHOST_COMPILE_NORM / GHOST_COMPILE_OUTPUT = 0/1 -> per-layer override
            _toplevel = os.getenv("GHOST_COMPILE_TOPLEVEL", "0") == "1"

            def _flag(name: str, default: bool) -> bool:
                v = os.getenv(name)
                return default if v is None else (v == "1")

            _emb = _flag("GHOST_COMPILE_EMB", False)
            _norm = _flag("GHOST_COMPILE_NORM", False)
            _output = _flag("GHOST_COMPILE_OUTPUT", _toplevel)
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

    @staticmethod
    def _bridge_ghost_levers(ghost_cfg):
        """Apply the dot-product lever config to the GHOST_* env vars the engine reads at import.

        An already-set env var takes precedence (so ad-hoc ``GHOST_*=...`` runs win); otherwise
        the config value is written. Validates lever 1b's subtract-val dependency. Must run
        before GhostDotProdHelper / ghostEngines are imported.
        """
        mapping = [
            ("GHOST_SUBTRACT_VAL", ghost_cfg.subtract_val),
            ("GHOST_BATCHED_DOTPROD", ghost_cfg.batched_dotprod),
            ("GHOST_BATCHED_DOTPROD_COMPILE", ghost_cfg.batched_dotprod_compile),
            ("GHOST_DECOUPLED_FN", ghost_cfg.decoupled_fn),
            ("GHOST_COMPILE_TOPLEVEL", ghost_cfg.compile_toplevel),
            ("GHOST_REGIONAL_COMPILE", ghost_cfg.regional_compile),
        ]
        resolved = {}
        for name, cfg_val in mapping:
            if name in os.environ:
                resolved[name] = os.environ[name] == "1"
            else:
                os.environ[name] = "1" if cfg_val else "0"
                resolved[name] = bool(cfg_val)

        # op-SAC mm save-fraction (int, not bool). Read at apply_ac during super().__init__, so it
        # must be bridged here before model build. Env var still wins for ad-hoc overrides.
        if "GHOST_OPSAC_MM_EVERY" not in os.environ:
            os.environ["GHOST_OPSAC_MM_EVERY"] = str(ghost_cfg.opsac_mm_every)
        resolved["GHOST_OPSAC_MM_EVERY"] = os.environ["GHOST_OPSAC_MM_EVERY"]

        # Both grouped dot-product paths recover train grads via subtract-val after backward.
        for lever in ("GHOST_BATCHED_DOTPROD", "GHOST_DECOUPLED_FN"):
            if resolved[lever] and not resolved["GHOST_SUBTRACT_VAL"]:
                raise ValueError(
                    f"ghost lever {lever} requires ghost.subtract_val=true — it recovers train "
                    "gradients via subtract-val after backward. Enable subtract_val or disable "
                    "the lever."
                )

        # The decoupled-Function path selects its own manager (GhostDecoupledManager) instead of
        # the eager engine, so the eager-engine lever 1b is inert when decoupled_fn is on.
        if resolved["GHOST_DECOUPLED_FN"] and resolved["GHOST_BATCHED_DOTPROD"]:
            logger.warning(
                "ghost.decoupled_fn=true takes precedence; ghost.batched_dotprod (eager lever 1b) "
                "is ignored on the decoupled-Function path."
            )

        logger.info(
            "Ghost levers: subtract_val=%s batched_dotprod=%s batched_dotprod_compile=%s "
            "decoupled_fn=%s compile_toplevel=%s regional_compile=%s opsac_mm_every=%s",
            resolved["GHOST_SUBTRACT_VAL"], resolved["GHOST_BATCHED_DOTPROD"],
            resolved["GHOST_BATCHED_DOTPROD_COMPILE"], resolved["GHOST_DECOUPLED_FN"],
            resolved["GHOST_COMPILE_TOPLEVEL"], resolved["GHOST_REGIONAL_COMPILE"],
            resolved["GHOST_OPSAC_MM_EVERY"],
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
