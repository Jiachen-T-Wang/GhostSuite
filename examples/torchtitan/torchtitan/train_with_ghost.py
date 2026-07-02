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
# lever flags before the config/CLI values are applied. For the same reason the decoupled
# Function-path selector (GHOST_DECOUPLED_FN) is read inside __init__ after the bridge, not at
# module top.


class GhostTrainer(Trainer):
    """Trainer subclass that appends a fixed validation batch for ghost GradDotProd."""

    def __init__(self, job_config):
        if not job_config.ghost.enable:
            raise RuntimeError("GhostTrainer requires ghost.enable=true.")

        # Bridge the dot-product lever config (--ghost.subtract_val / .decoupled_fn /
        # .compile_toplevel / .regional_compile) to the GHOST_* env vars the engine reads at import
        # time. An explicitly-set env var wins (so ad-hoc `GHOST_*=...` runs still work); otherwise
        # the config value is applied. Must run before the lazy GhostDotProdHelper import AND before
        # the Function-path read below.
        self._bridge_ghost_levers(job_config.ghost)

        # Read the decoupled Function-path selector AFTER bridging. It removes the eager backward
        # hooks so torch.compile can regional-compile the model, keeps each layer's native backward,
        # and computes the decoupled dot-product as a small in-backward transient (the fast path).
        fn_path = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"

        # Compile is only allowed on the decoupled Function path; the eager hook engine still
        # hard-disables it (its hooks force compiled autograd + graph breaks). On the Function path
        # we DEFER compile: the model must be built uncompiled so the ghost manager can monkeypatch
        # the supported layers' forward FIRST, then we regional-compile each block (attach-then-
        # compile is the order Inductor functionalizes the per-layer buffer mutations cleanly;
        # compile-then-attach yields an invalid graph output for the buffer copy_).
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

            # The loss was built while compile.enable was force-disabled (the deferred-compile
            # trick), so it would run EAGER — an fp32 full-vocab cross-entropy the plain compiled
            # baseline does not pay. Recompile it here, swapping in the compile-friendly CE
            # (constant 1/N normalizer) because the default F.cross_entropy backward trips
            # Inductor's data-dependent-scalar codegen. Only the known CE variants are replaced;
            # anything else fails loud rather than silently changing the training objective.
            if job_config.ghost.compile_loss and "loss" in self._compile_config.components:
                from torchtitan.components import loss as loss_mod

                inner = getattr(self.loss_fn, "unwrapped_loss_fn", None)
                if inner is None:
                    raise RuntimeError(
                        "ghost.compile_loss: expected the trainer loss_fn to be the "
                        "RescaleAccumulatedLoss wrapper; got a bare callable."
                    )
                base = getattr(inner, "_torchdynamo_orig_callable", inner)
                if base not in (
                    loss_mod.cross_entropy_loss,
                    loss_mod.compile_friendly_cross_entropy_loss,
                ):
                    raise RuntimeError(
                        "ghost.compile_loss: loss_fn is not the known cross-entropy variant "
                        f"({base}); refusing to swap in the compile-friendly CE. Set "
                        "--ghost.no-compile_loss."
                    )
                # Native F.cross_entropy compiles to a ~2.4x faster fused kernel than the manual
                # log_softmax+gather CE (H200 trace: 1.26 vs 2.98 ms/pass) and is what the plain
                # baseline runs. The old data-dependent-scalar Inductor crash predates this
                # torch; GHOST_COMPILE_LOSS_FRIENDLY=1 restores the manual variant if it ever
                # resurfaces.
                friendly = os.getenv("GHOST_COMPILE_LOSS_FRIENDLY", "0") == "1"
                impl = (
                    loss_mod.compile_friendly_cross_entropy_loss
                    if friendly
                    else loss_mod.cross_entropy_loss
                )
                self.loss_fn.unwrapped_loss_fn = torch.compile(
                    impl, backend=self._compile_config.backend
                )
                logger.info(
                    "Ghost: compiled the loss (%s CE).",
                    "compile-friendly" if friendly else "native",
                )

            # Opt-in: also regional-compile the top-level layers that apply_compile skips, so their
            # ghost in-graph dot folds into a compiled region instead of running eager.
            # Per-layer attribution on H200/130M
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

        if self.ghost_helper.use_fn_path:
            # fn-path: score persistence is not wired (helper logs a loud warning when
            # save_interval > 0), so don't advertise save_interval/save_train_batch here.
            logger.info(
                "Ghost GradDotProd enabled | val_batch_size=%d | fn-path (scores computed "
                "in-graph, not persisted)",
                self.ghost_helper.val_batch_size,
            )
        else:
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
        the config value is written. Validates the decoupled path's subtract-val dependency. Must
        run before GhostDotProdHelper / ghostEngines are imported.
        """
        mapping = [
            ("GHOST_SUBTRACT_VAL", ghost_cfg.subtract_val),
            ("GHOST_DECOUPLED_FN", ghost_cfg.decoupled_fn),
            ("GHOST_COMPILE_TOPLEVEL", ghost_cfg.compile_toplevel),
            ("GHOST_REGIONAL_COMPILE", ghost_cfg.regional_compile),
            ("GHOST_SEPARATE_VAL", ghost_cfg.separate_val),
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

        # The decoupled path recovers train grads via subtract-val after backward.
        if resolved["GHOST_DECOUPLED_FN"] and not resolved["GHOST_SUBTRACT_VAL"]:
            raise ValueError(
                "ghost lever GHOST_DECOUPLED_FN requires ghost.subtract_val=true — it recovers "
                "train gradients via subtract-val after backward. Enable subtract_val or disable "
                "the lever."
            )
        if resolved["GHOST_SEPARATE_VAL"] and not resolved["GHOST_DECOUPLED_FN"]:
            raise ValueError(
                "ghost lever GHOST_SEPARATE_VAL is a mode of the decoupled-Function engine; "
                "enable ghost.decoupled_fn or disable separate_val."
            )

        logger.info(
            "Ghost levers: subtract_val=%s decoupled_fn=%s compile_toplevel=%s "
            "regional_compile=%s opsac_mm_every=%s",
            resolved["GHOST_SUBTRACT_VAL"], resolved["GHOST_DECOUPLED_FN"],
            resolved["GHOST_COMPILE_TOPLEVEL"], resolved["GHOST_REGIONAL_COMPILE"],
            resolved["GHOST_OPSAC_MM_EVERY"],
        )

    def _ghost_val_pass(self) -> None:
        """separate-val: plain fwd/bwd on the fixed val batch, then harvest .grad -> gval caches.

        Runs with the ghost wrappers disabled (native ops — the val rows need no dots) and with
        the accumulation-rescale off (the harvested gval should be the plain mean-over-val-tokens
        gradient, not divided by the train accumulation steps). ``harvest_val_grads`` clears the
        harvested ``.grad`` so the following train microbatches accumulate from zero."""
        mgr = self.ghost_helper.fn_manager
        helper = self.ghost_helper
        # Wrappers disabled: the val rows need no dots, and paying their projection GEMMs is
        # measurably worse than the guard-flip dispatch (H200 130M: running the val pass with
        # wrappers enabled cost +14.5 ms/step at N=1 vs the flip's ~0-3 ms).
        mgr.set_enabled(False)
        try:
            with self.train_context(None):
                with self.maybe_enable_amp:
                    pred = self.model_parts[0](helper.val_input_dict["input"])
                    with self.loss_fn.no_rescale():
                        loss = self.loss_fn(pred, helper.val_labels)
                del pred
                loss.backward()
        finally:
            mgr.set_enabled(True)
        mgr.harvest_val_grads()
        # The val pass is real per-step work: count its tokens once in throughput metrics
        # (the combined-batch path counts them once per microbatch instead).
        self.metrics_processor.ntokens_since_last_log += helper.val_labels.numel()

    def forward_backward_step(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        microbatch_idx: int = 0,
    ) -> torch.Tensor:
        """Override to append fixed validation batch and run ghost hooks."""
        # No parallel contexts supported in ghost mode.
        if self.ghost_helper.use_separate_val:
            # separate-val: the val gradient was harvested once at step start (_ghost_val_pass);
            # train microbatches run WITHOUT the appended val rows.
            combined_input, combined_labels = input_dict, labels
        else:
            combined_input, combined_labels = self.ghost_helper.combine_with_val(
                input_dict, labels
            )
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

        # Aggregate this microbatch's dot products, then fold its val grad into the per-step
        # accumulator (subtract-val) before the next backward overwrites the buffers.
        self.ghost_helper.engine.aggregate_and_log()
        self.ghost_helper.engine.accumulate_microbatch()
        return loss

    def train_step(
        self, data_iterator: Iterable[Tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Reset per-step dot/grad accumulation before the gradient-accumulation microbatch loop.
        self.ghost_helper.begin_step()

        # separate-val: one plain backward on the fixed val batch (wrappers disabled), harvest
        # autograd's .grad into the per-param gval caches the microbatch dots project against.
        # The val gradient is constant across a step's microbatches (weights don't change), so
        # this replaces carrying the val rows through EVERY microbatch's forward/backward.
        if self.ghost_helper.use_separate_val:
            self._ghost_val_pass()

        accumulated_losses = []
        for microbatch_idx in range(self.gradient_accumulation_steps):
            try:
                input_dict, labels = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            loss = self.forward_backward_step(input_dict, labels, microbatch_idx=microbatch_idx)
            accumulated_losses.append(loss.detach())

        # Recover train grads once via subtract-val from the accumulated per-microbatch val grads.
        if self.ghost_helper.use_fn_path:
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
