"""
Ghost Engine Manager - Unified interface for managing gradient computation engines.

This module provides a clean interface for initializing and managing ghost engines
based on configuration, removing the need for method-specific code in training loops.
"""

import os
import warnings
from contextlib import nullcontext
from typing import Optional, Union, Dict, Any

import torch
from torch import nn

from .graddotprod_engine import GradDotProdEngine
from .gradProjection.gradproj_engine import GradProjLoraEngine
from .decoupled_capture_dotprod import GhostDecoupledManager
from .decoupled_compile import attach_and_compile_decoupled


class GhostEngineManager:
    """
    A unified manager for ghost engines that provides a clean interface
    for training loops without method-specific initialization code.
    """
    
    def __init__(self, config, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 ddp_info: Dict[str, Any], val_data=None):
        """
        Initialize the ghost engine manager based on configuration.
        
        Args:
            config: Training configuration object with method and other settings
            model: The PyTorch model to attach engines to
            optimizer: The optimizer to integrate with
            ddp_info: Distributed training information
            val_data: Tuple of (X_val, Y_val) validation data (required for GradDotProd)
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.ddp_info = ddp_info
        
        # Initialize engine based on method
        self.engine = None
        # Decoupled in-graph + compile path (fn-path): hosts a GhostDecoupledManager instead of the
        # eager GradDotProdEngine; dot-products land in per-layer buffers during backward and train
        # grads are recovered via subtract-val before the optimizer step.
        self.decoupled_mgr = None
        self.is_fn_path = False
        self._last_dot = None
        if val_data is not None:
            self.X_val, self.Y_val = val_data
        else:
            self.X_val, self.Y_val = None, None
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the appropriate engine based on config.method."""
        if self.config.method == 'GradDotProd':
            self._initialize_graddotprod_engine()
        elif self.config.method == 'GradProjLora':
            self._initialize_gradproj_engine()
        elif self.config.method == 'Regular':
            # No engine needed for regular training
            print("[INFO] Regular training mode - no ghost engine required.")
        else:
            print(f"[WARNING] Unknown method '{self.config.method}' - no ghost engine initialized.")
    
    def _initialize_graddotprod_engine(self):
        """Initialize the GradDotProd engine — eager by default, or the decoupled fn-path."""
        if getattr(self.config, "decoupled_fn", False):
            self._initialize_decoupled_engine()
            return

        print("[INFO] Initializing GradDotProdEngine ...")
        
        # Prepare directory for saving dot products
        dot_prod_save_path = os.path.join(self.config.result_dir, "grad_dotprods")
        if self.ddp_info['master_process']:
            os.makedirs(dot_prod_save_path, exist_ok=True)
        
        # Initialize the engine
        self.engine = GradDotProdEngine(
            module=self.model,
            val_batch_size=self.config.val_batch_size,
            loss_reduction='mean',
            use_dummy_bias=True,
            dot_prod_save_path=dot_prod_save_path,
            log_grad_norms=getattr(self.config, "log_grad_norms", False),
            score_exclude_params=getattr(self.config, "score_exclude_params", None),
        )
        
        # Attach to optimizer
        self.engine.attach(self.optimizer)
        
        # Validate that validation data is provided
        if self.X_val is None or self.Y_val is None:
            raise ValueError("X_val and Y_val are required for GradDotProd method")
        
        # Attach validation data to the engine
        self.engine.attach_and_store_valset(self.X_val, self.Y_val)
        
        print("[INFO] GradDotProdEngine initialized successfully.")

    def _initialize_decoupled_engine(self):
        """Decoupled in-graph + (optional) regional-compile path for GradDotProd.

        Hosts a ``GhostDecoupledManager`` (compile-clean: native layer backward preserved, the
        dot-product computed as a small transient in-backward) instead of the eager engine, and
        regional-compiles the transformer blocks via the general ``attach_and_compile_decoupled``
        harness. Train grads are recovered via subtract-val in ``prepare_gradients``.
        """
        print("[INFO] Initializing ghost decoupled in-graph engine ...")
        if self.X_val is None or self.Y_val is None:
            raise ValueError("X_val and Y_val are required for GradDotProd method")
        if isinstance(self.X_val, dict):
            raise NotImplementedError(
                "Ghost decoupled fn-path supports tensor token inputs only (not LLaVA dict inputs)."
            )
        if self.ddp_info.get("ddp", False):
            raise RuntimeError("Ghost decoupled fn-path is single-GPU only (no DDP).")
        if getattr(self.config, "gradient_accumulation_steps", 1) != 1:
            raise RuntimeError(
                "Ghost decoupled fn-path requires gradient_accumulation_steps == 1 "
                "(per-layer buffers hold only the last microbatch's grad_val)."
            )

        device = self.ddp_info["device"]
        train_bs = self.config.batch_size
        val_bs = self.config.val_batch_size

        # GREATS' only ghost pass is scoring over candidate_batch_size + val (the update is a plain
        # step on the selected subset, taken with capture disabled — no second ghost shape). Prime
        # that one shape; single-pass callers (no candidate_batch_size) keep the update-size shape.
        cand = getattr(self.config, "candidate_batch_size", None)
        if cand is not None:
            warmup_shapes = [cand + val_bs]
        else:
            warmup_shapes = None
        default_total = train_bs + val_bs

        xv = self.X_val
        model = self.model

        def warmup_fn(total_bs=None):
            # Shape-faithful warmup batch: tile the stored validation tokens up to the combined
            # batch size (valid indices without needing the vocab size; seq matches the val batch).
            tb = default_total if total_bs is None else total_bs
            reps = (tb + xv.shape[0] - 1) // xv.shape[0]
            idx = xv.repeat(reps, 1)[:tb].to(device)
            was_training = model.training
            model.train()
            out = model(idx, idx)
            out.loss.backward()
            model.train(was_training)

        regions = None
        if getattr(self.config, "decoupled_compile", False):
            # Regional-compile the repeated transformer blocks (analogue of TorchTitan model.layers).
            transformer = getattr(model, "transformer", None)
            blocks = getattr(transformer, "h", None) if transformer is not None else None
            if blocks is None:
                raise RuntimeError(
                    "Ghost decoupled compile: could not find transformer blocks at "
                    "model.transformer.h to regional-compile."
                )
            regions = list(blocks)

        # Top-level compile lever (P3.2): candidate non-repeated layers (output + final norm).
        # The harness compiles only the in-graph (non-tied) ones; for the default tied GPT-2 the
        # lm_head is on the capture path and is skipped there.
        extra_regions = None
        if regions is not None and getattr(self.config, "decoupled_compile_toplevel", False):
            extra_regions = []
            if hasattr(model, "lm_head"):
                extra_regions.append(model.lm_head)
            transformer = getattr(model, "transformer", None)
            if transformer is not None and hasattr(transformer, "ln_f"):
                extra_regions.append(transformer.ln_f)

        # Activation-checkpointing lever (P3.1): min-cut partitioner recompute budget for the
        # compiled regions (cuts the in-graph path's pinned activations; matters at larger scale).
        mem_budget = getattr(self.config, "decoupled_mem_budget", None)

        self.decoupled_mgr = attach_and_compile_decoupled(
            model, val_batch_size=val_bs, warmup_fn=warmup_fn, compile_regions=regions,
            extra_regions=extra_regions, activation_memory_budget=mem_budget,
            score_exclude_params=getattr(self.config, "score_exclude_params", None),
            warmup_shapes=warmup_shapes,
        )
        self.is_fn_path = True
        self.engine = None

        # Persistence: mirror the eager engine's outputs (valset.pt + dot_prod_log_iter_*.pt) so
        # --decoupled_fn produces the same GradDotProd scores instead of silently discarding them.
        self._fn_save_dir = os.path.join(self.config.result_dir, "grad_dotprods")
        self.dot_product_log = []
        self._fn_train_batch = None  # (X_train, Y_train, iter_num, batch_idx) for the current step
        if self.ddp_info.get("master_process", True):
            os.makedirs(self._fn_save_dir, exist_ok=True)
            valset_path = os.path.join(self._fn_save_dir, "valset.pt")
            torch.save(
                {"X_val": self._to_cpu(self.X_val), "Y_val": self._to_cpu(self.Y_val)},
                valset_path,
            )
            print(f"[INFO] Saved validation set to {valset_path}")

        print(
            "[INFO] Ghost decoupled in-graph engine initialized "
            f"(compile={'on' if regions is not None else 'off'})."
        )

    @staticmethod
    def _to_cpu(t):
        return t.detach().to("cpu") if isinstance(t, torch.Tensor) else t

    def _fn_append_log(self, iter_num, batch_idx, X_train, Y_train):
        """Append this step's aggregated per-train-sample dot-product to the log (fn-path)."""
        if self._last_dot is None:
            warnings.warn("decoupled fn-path: no dot-product computed this step; nothing logged.")
            return
        entry = {
            "dot_product": self._to_cpu(self._last_dot),
            "X_train": self._to_cpu(X_train),
            "Y_train": self._to_cpu(Y_train),
            "iter_num": iter_num,
            "batch_idx": batch_idx,
        }
        self.dot_product_log.append(entry)

    def _fn_save_log(self, iter_num: int):
        """Write the accumulated fn-path dot-product log to disk and clear it."""
        if not self.dot_product_log:
            return
        os.makedirs(self._fn_save_dir, exist_ok=True)
        file_path = os.path.join(self._fn_save_dir, f"dot_prod_log_iter_{iter_num}.pt")
        torch.save(self.dot_product_log, file_path)
        print(f"[INFO] Saved decoupled dot-product log at iteration {iter_num} ...")
        self.dot_product_log.clear()

    def _initialize_gradproj_engine(self):
        """Initialize GradProjLoraEngine with projection setup."""
        print("[INFO] Initializing GradProjLoraEngine ...")
        
        # Prepare directory for saving projections
        proj_dir = os.path.join(self.config.result_dir, "projections")
        if self.ddp_info['master_process']:
            os.makedirs(proj_dir, exist_ok=True)
        
        # Get projection parameters from config or use defaults
        proj_layers = getattr(self.config, 'proj_layers', 'mlp,attn')
        proj_rank_total = getattr(self.config, 'proj_rank_total', 256)
        proj_rank_min = getattr(self.config, 'proj_rank_min', 8)
        proj_seed = getattr(self.config, 'proj_seed', 42)
        proj_dtype = getattr(self.config, 'proj_dtype', self.config.train_dtype)
        proj_save_interval = getattr(self.config, 'proj_save_interval', 
                                    self.config.dot_prod_save_interval)
        include_embeddings = getattr(self.config, 'include_embeddings', False)
        
        # Initialize the engine
        self.engine = GradProjLoraEngine(
            module=self.model,
            proj_layers=proj_layers,
            proj_rank_total=proj_rank_total,
            proj_rank_min=proj_rank_min,
            proj_seed=proj_seed,
            proj_dtype=proj_dtype,
            proj_dir=proj_dir,
            proj_save_interval=proj_save_interval,
            include_embeddings=include_embeddings
        )
        
        # Attach the engine
        self.engine.attach()
        
        print("[INFO] GradProjLoraEngine initialized successfully.")
    

    def is_active(self) -> bool:
        """Check if any ghost engine is active."""
        return self.engine is not None or self.is_fn_path
    
    def get_method(self) -> str:
        """Get the current method name."""
        return self.config.method
    
    def attach_train_batch(self, X_train, Y_train, iter_num, batch_idx=None):
        """Attach training batch information to the engine (if applicable)."""
        if self.is_fn_path:
            # Keep the current step's batch + iter so prepare_gradients can log this step's dot.
            self._fn_train_batch = (X_train, Y_train, iter_num, batch_idx)
            return
        if self.engine and hasattr(self.engine, 'attach_train_batch'):
            self.engine.attach_train_batch(X_train, Y_train, iter_num, batch_idx)

    def update_validation_batch(self, X_val, Y_val):
        """Update validation batch (useful when refreshing every step)."""
        self.X_val, self.Y_val = X_val, Y_val
    
    def set_decoupled_enabled(self, flag):
        """fn-path: toggle dot capture so a plain optimizer step (e.g. the GREATS update on the
        selected subset) can run with the manager still attached. No-op off the fn-path (the eager
        engine's output hook is already gated on the saved-tensors context)."""
        if self.is_fn_path and self.decoupled_mgr is not None:
            self.decoupled_mgr.set_enabled(flag)

    def prepare_decoupled_shape(self, total_bs):
        """fn-path multi-shape: point the in-graph buffers at this combined batch size before a
        forward (no-op off the fn-path). Required when one attached manager serves more than one
        batch shape per step, e.g. the GREATS scoring (N+m) and update (k+m) passes."""
        if self.is_fn_path and self.decoupled_mgr is not None:
            self.decoupled_mgr.prepare_shape(total_bs)

    def decoupled_run_step(self):
        """fn-path: aggregate the per-train-sample dot from the in-graph buffers and publish
        grad_val onto each param (needed before ``decoupled_recover``). Returns the dot tensor.
        Use directly (instead of ``prepare_gradients``) when a caller wants the score WITHOUT
        recovering train grads — e.g. a GREATS scoring pass that takes no optimizer step."""
        if not (self.is_fn_path and self.decoupled_mgr is not None):
            raise RuntimeError("decoupled_run_step is only valid on the decoupled fn-path.")
        return self.decoupled_mgr.run_step_dotprod()

    def decoupled_recover(self):
        """fn-path: recover train-only grads into ``.grad`` via subtract-val (call after
        ``decoupled_run_step`` has published grad_val)."""
        if not (self.is_fn_path and self.decoupled_mgr is not None):
            raise RuntimeError("decoupled_recover is only valid on the decoupled fn-path.")
        self.decoupled_mgr.recover_train_grads()

    # ------------------------------------------------------------------ #
    # Pluggable-policy primitives (used by examples/lm/shared.online_selection_step)
    # ------------------------------------------------------------------ #
    @property
    def val_batch_size(self):
        return self.config.val_batch_size

    def read_scores(self, metric="dot"):
        """One unified per-sample score read after a ghost scoring backward, hiding the eager-vs-fn
        split. Returns the per-train-sample score tensor on CPU (or None). Also appends the dot to
        the log so a caller can persist it (save_metrics). On the fn-path this publishes grad_val,
        so a subsequent ``recover_train_grads`` is valid; it does NOT recover or step.

        ``metric``: 'dot' (raw <g_i, g_val>) or 'cosine' (needs per-sample grad norms, eager only;
        the fast path has no norms and raises)."""
        if self.is_fn_path:
            self._last_dot = self.decoupled_run_step()  # also publishes grad_val for a later recover
            if self._fn_train_batch is not None:
                X_train, Y_train, iter_num, batch_idx = self._fn_train_batch
                self._fn_append_log(iter_num, batch_idx, X_train, Y_train)
            if metric != "dot":
                raise ValueError(
                    f"read_scores(metric={metric!r}) needs per-sample grad norms, which the "
                    "decoupled fast path does not produce. Use --eager for cosine ranking.")
            return None if self._last_dot is None else self._last_dot.detach().to("cpu")
        self.engine.aggregate_and_log()
        log = self.engine.dot_product_log
        entry = log[-1] if log else None
        if entry is None:
            return None
        scores = entry["dot_product"].float()
        if metric == "cosine":
            tn = entry["train_grad_norm"].float()
            vn = float(entry.get("val_grad_norm", 1.0)) or 1.0
            scores = scores / (tn * vn + 1e-12)
        return scores

    def recover_train_grads(self):
        """Reuse path: recover the train-only grad into ``.grad`` via subtract-val. Assumes
        ``read_scores`` already ran (fn-path needs grad_val published first)."""
        if self.is_fn_path:
            self.decoupled_recover()
        elif self.engine is not None and hasattr(self.engine, "prepare_gradients"):
            self.engine.prepare_gradients()

    def finish_step(self):
        """Post-step cleanup for the reuse path: the eager engine unlocks grad creation that
        ``recover_train_grads`` locked; the fn-path has nothing to do."""
        if self.is_fn_path:
            return
        if self.engine is not None and hasattr(self.engine, "clear_gradients"):
            self.engine.clear_gradients()

    def discard_scores(self):
        """Reselect path: drop the scoring pass's logged dots + per-param transient state WITHOUT
        recovering train grads (the caller then takes a fresh plain backward on the selection).

        Recovery is what normally deletes the per-param ``_ghost_grad_val`` (and tied stash); on a
        scoring-only step it is skipped, so clear those here to free the (val-grad-shaped) tensors
        and avoid stale state leaking into the next step."""
        if self.is_fn_path:
            self.dot_product_log.clear()
            self._last_dot = None
        elif self.engine is not None:
            self.engine.dot_product_log.clear()
            if hasattr(self.engine, "clear_gradients"):
                self.engine.clear_gradients()
        for p in self.model.parameters():
            for attr in ("_ghost_grad_val", "grad_dot_prod", "_ghost_tied_gval",
                         "_ghost_tied_stash", "_ghost_tied_train_bs", "_ghost_tied_log_norms"):
                if hasattr(p, attr):
                    delattr(p, attr)

    def set_capture_enabled(self, flag):
        """Toggle dot capture so a plain backward can run with the manager attached. Eager is
        gated on the saved-tensors context already, so this is a no-op there."""
        self.set_decoupled_enabled(flag)

    def prepare_gradients(self):
        """Prepare gradients after backward pass (if applicable)."""
        if self.is_fn_path and self.decoupled_mgr is not None:
            # fn-path: dot-products are already in per-layer buffers (computed in the in-graph
            # backward). Aggregate them (also publishes grad_val), then recover train grads via
            # subtract-val — all before the optimizer step. Log the dot for persistence.
            self._last_dot = self.decoupled_mgr.run_step_dotprod()
            self.decoupled_mgr.recover_train_grads()
            if self._fn_train_batch is not None:
                X_train, Y_train, iter_num, batch_idx = self._fn_train_batch
                self._fn_append_log(iter_num, batch_idx, X_train, Y_train)
            return
        if self.engine and hasattr(self.engine, 'prepare_gradients'):
            self.engine.prepare_gradients()

    def saved_tensors_context(self):
        """Context manager for saved tensor capture when supported."""
        if self.engine and hasattr(self.engine, "saved_tensors_context"):
            return self.engine.saved_tensors_context()
        return nullcontext()
    
    def aggregate_and_log(self):
        """Aggregate and log metrics after optimizer step (if applicable)."""
        if self.is_fn_path:
            # Intentional no-op: the fn-path already aggregated the dot and appended the log
            # entry in prepare_gradients(); nothing to do post-step.
            return
        if self.engine and hasattr(self.engine, 'aggregate_and_log'):
            self.engine.aggregate_and_log()

    def clear_gradients(self):
        """Clear gradients after optimizer step (if applicable)."""
        if self.is_fn_path:
            # Intentional no-op: the in-graph buffers are overwritten by the next backward, and
            # recover_train_grads() already deleted the per-param _ghost_grad_val/grad_dot_prod.
            return
        if self.engine and hasattr(self.engine, 'clear_gradients'):
            self.engine.clear_gradients()
    
    def should_save_metrics(self, iter_num: int) -> bool:
        """Check if metrics should be saved at this iteration."""
        if self.config.method == 'GradDotProd' and iter_num > 0:
            return iter_num % self.config.dot_prod_save_interval == 0
        elif self.config.method == 'GradProjLora' and iter_num > 0:
            save_interval = getattr(self.config, 'proj_save_interval', 
                                   self.config.dot_prod_save_interval)
            return iter_num % save_interval == 0
        # Add other engine-specific save intervals here
        return False
    
    def save_metrics(self, iter_num: int):
        """Save metrics to disk (if applicable)."""
        if self.is_fn_path:
            self._fn_save_log(iter_num)
            return
        if self.config.method == 'GradDotProd' and self.engine:
            self.engine.save_dot_product_log(iter_num=iter_num)
        elif self.config.method == 'GradProjLora' and self.engine:
            # For GradProjLora, collect_batch handles saving internally
            if hasattr(self.engine, 'save_projections'):
                self.engine.save_projections(iter_num=iter_num)
    
    def get_validation_data(self):
        """Get validation data for methods that need it."""
        return self.X_val, self.Y_val
    
    def prepare_forward_input(self, X_train, Y_train):
        """
        Prepare forward pass input by concatenating with validation data if needed.
        
        Returns:
            Tuple of (X, Y) ready for forward pass
        """
        if self.config.method == 'GradDotProd' and self.X_val is not None:
            # Concatenate train and val batches for GradDotProd method
            if isinstance(self.X_val, dict):
                # Handle dictionary inputs (e.g., LLaVA model)
                X_cat = {}
                X_cat["input_ids"] = torch.cat((X_train["input_ids"], self.X_val["input_ids"]), dim=0)
                X_cat["pixel_values"] = torch.cat((X_train["pixel_values"], self.X_val["pixel_values"]), dim=0)
                X_cat["attention_mask"] = torch.cat((X_train["attention_mask"], self.X_val["attention_mask"]), dim=0)
                Y_cat = torch.cat((Y_train, self.Y_val), dim=0)
                return X_cat, Y_cat
            else:
                # Handle tensor inputs
                X_cat = torch.cat((X_train, self.X_val), dim=0)
                Y_cat = torch.cat((Y_train, self.Y_val), dim=0)
                return X_cat, Y_cat
        else:
            # Regular training - return original inputs
            return X_train, Y_train
    
    def detach_for_evaluation(self):
        """Detach engines during evaluation to avoid interference."""
        if self.is_fn_path:
            # Intentionally stay attached. evaluation runs under torch.no_grad() (estimate_loss is
            # @torch.no_grad()), so the in-graph/capture Functions are forward-only identities — no
            # backward, hence no dot computation and no buffer writes — regardless of the eval batch
            # size. Detaching here would not help anyway: the regional-compiled blocks capture the
            # ghost-wrapped leaves at compile time and cannot be un-ghosted without recompiling.
            return
        if self.engine:
            if hasattr(self.engine, 'detach'):
                self.engine.detach()
            elif hasattr(self.engine, 'disable_hooks'):
                self.engine.disable_hooks()

    def reattach_after_evaluation(self):
        """Reattach engines after evaluation."""
        if self.is_fn_path:
            return  # fn-path never detached for eval (see detach_for_evaluation).
        if self.engine:
            if hasattr(self.engine, 'attach'):
                self.engine.attach(self.optimizer)
            elif hasattr(self.engine, 'enable_hooks'):
                self.engine.enable_hooks()
    
    def cleanup(self):
        """Cleanup and save any remaining data during training termination."""
        if self.is_fn_path:
            # Persist any remaining dot-products (matching the eager cleanup save at iter -1).
            if self.dot_product_log:
                try:
                    self._fn_save_log(-1)
                except Exception as e:
                    print(f"Error saving remaining decoupled dot-products during cleanup: {e}")
            if self.decoupled_mgr is not None:
                try:
                    self.decoupled_mgr.detach()
                except Exception as e:
                    print(f"Error detaching decoupled manager during cleanup: {e}")
            return
        if not self.engine:
            return
            
        # Save any remaining metrics
        if (self.config.method == 'GradDotProd' and 
            hasattr(self.engine, 'dot_product_log') and 
            self.engine.dot_product_log):
            try:
                # Save with a special cleanup iteration number
                self.engine.save_dot_product_log(iter_num=-1)
            except Exception as e:
                print(f"Error saving remaining dot products during cleanup: {e}")
        elif self.config.method == 'GradProjLora':
            # GradProjLora saves projections incrementally, just ensure cleanup
            if hasattr(self.engine, 'cleanup'):
                try:
                    self.engine.cleanup()
                except Exception as e:
                    print(f"Error during GradProjLora cleanup: {e}")
        
        # Detach the engine
        try:
            if hasattr(self.engine, 'detach'):
                self.engine.detach()
            elif hasattr(self.engine, 'disable_hooks'):
                self.engine.disable_hooks()
        except Exception as e:
            print(f"Error detaching ghost engine during cleanup: {e}")
