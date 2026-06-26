"""GREATS online batch selection — training loop.

First-order online selection (no pairwise Gram, no greedy second-order term). Each
step:

  1. Draw a candidate pool of N = ``candidate_batch_size`` train samples and a fresh
     scoring val batch of m = ``val_batch_size`` samples (from the eval window pool by
     default).
  2. *Scoring pass* — one fused GradDotProd forward/backward over ``[candidate ++ val]``
     giving per-candidate ``s_i = <g_i, g_val>`` (cosine optional). No optimizer step.
  3. Select the top-k candidates (k = ``batch_size``).
  4. *Update pass* — a normal GradDotProd step over ``[selected ++ val]``; subtract-val
     recovers the selected-subset mean gradient into ``.grad`` for the optimizer.

Steps 2 and 4 are two engine passes per step (the paper's two-pass note). Both reuse
the validated GradDotProd path, so selection never touches engine internals.

See docs/plans/greats_example_implementation_2026-06-25.md.
"""

import copy
import os
import sys
import time

import torch

# Reuse shared/ utilities under examples/lm/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)
for _p in (os.path.join(_EXAMPLES_DIR, "lm"), _REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.training_utils import (
    get_learning_rate,
    update_learning_rate,
    estimate_loss,
    save_training_results,
    to_device,
)
from ghostEngines import GhostEngineManager


class GreatsTrainer:
    """Online batch selection trainer (GREATS, first-order)."""

    def __init__(self, model, optimizer, scaler, config, ddp_info,
                 get_batch_fn, get_val_batch_fn, ctx):
        print("[INFO] Initializing GreatsTrainer ...")

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.ddp_info = ddp_info
        self.get_batch = get_batch_fn
        self.get_val_batch = get_val_batch_fn
        self.ctx = ctx
        self.wandb_run = None

        self.iter_num = 0
        self.is_greats = (config.method == "GREATS")

        self.ghost = None
        if self.is_greats:
            # The scoring/update engine is a GradDotProd engine. The manager keys on
            # config.method, so hand it a shallow copy with method='GradDotProd' while
            # the run itself is labelled 'GREATS' (result dirs, wandb, etc).
            score_cfg = copy.copy(config)
            score_cfg.method = "GradDotProd"

            X_val, Y_val = self.get_val_batch(config.val_batch_size, return_idx=False)
            X_val = to_device(X_val, ddp_info["device"])
            Y_val = to_device(Y_val, ddp_info["device"])

            self.ghost = GhostEngineManager(
                config=score_cfg,
                model=self.model,
                optimizer=self.optimizer,
                ddp_info=ddp_info,
                val_data=(X_val, Y_val),
            )

        self._init_wandb()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run_training(self):
        print("[INFO] Starting training...")
        result_file = self.config.get_result_file_path()
        try:
            while self.iter_num < self.config.max_steps:
                if self.iter_num % self.config.eval_interval == 0:
                    self._run_evaluation(result_file)
                if self.config.args.eval_only:
                    print("Eval only mode, exiting now")
                    break
                try:
                    if self.is_greats:
                        self._greats_step(self.iter_num)
                    else:
                        self._regular_step(self.iter_num)
                except StopIteration:
                    print("[INFO] Data exhausted; terminating training loop.")
                    break
                self.iter_num += 1
        except Exception as e:  # noqa: BLE001 - mirror graddotprod trainer
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup(result_file)

    # ------------------------------------------------------------------ #
    # GREATS step
    # ------------------------------------------------------------------ #
    def _greats_step(self, iter_num):
        t0 = time.time()

        # Fresh scoring val batch (online selection targets the current val draw).
        X_val, Y_val = self.get_val_batch(self.config.val_batch_size, return_idx=False)
        X_val = to_device(X_val, self.ddp_info["device"])
        Y_val = to_device(Y_val, self.ddp_info["device"])
        self.ghost.update_validation_batch(X_val, Y_val)

        # Candidate pool of N samples.
        X_cand, Y_cand, _ = self.get_batch(
            "train", batch_size=self.config.candidate_batch_size, return_idx=True
        )
        if isinstance(X_cand, dict):
            raise NotImplementedError(
                "The GREATS pretrain example supports tensor inputs (GPT-2) only."
            )

        # Scoring pass (no optimizer step).
        _, entry = self._ghost_pass(X_cand, Y_cand, iter_num, do_step=False)
        scores = self._scores_from_entry(entry)  # [N] cpu float

        # Select the top-k candidates.
        k = self.config.batch_size
        sel = torch.topk(scores, k).indices
        sel_dev = sel.to(X_cand.device)
        X_sel = X_cand.index_select(0, sel_dev)
        Y_sel = Y_cand.index_select(0, sel_dev)

        # LR schedule for the real update.
        lr = (get_learning_rate(iter_num, self.config)
              if self.config.decay_lr else self.config.learning_rate)
        update_learning_rate(self.optimizer, lr)

        # Update pass: real optimizer step on the selected subset.
        loss, _ = self._ghost_pass(X_sel, Y_sel, iter_num, do_step=True)

        torch.cuda.synchronize()
        dt = time.time() - t0

        sel_scores = scores.index_select(0, sel)
        frac_pos = (scores > 0).float().mean().item()
        print(f"Step {iter_num} | loss {loss.item():.4f} | lr {lr:.6f} | "
              f"selected {k}/{self.config.candidate_batch_size} | "
              f"mean sel score {sel_scores.mean().item():.3e} | "
              f"frac>0 {frac_pos:.2f} | {dt:.3f}s")

        metrics = {
            "train/lr": lr,
            "train/loss": loss.item(),
            "train/step_time": dt,
            "select/mean_selected_score": sel_scores.mean().item(),
            "select/mean_candidate_score": scores.mean().item(),
            "select/frac_positive": frac_pos,
        }
        self._log_metrics(metrics, step=iter_num)

    def _ghost_pass(self, X_train, Y_train, iter_num, do_step):
        """One GradDotProd forward/backward over [train ++ val].

        Returns ``(loss, log_entry)`` where ``log_entry`` is the per-sample score dict
        for ``X_train``. When ``do_step`` is True the optimizer steps on the subtract-val
        recovered (train-only) gradient; otherwise grads are discarded.
        """
        self.ghost.attach_train_batch(X_train, Y_train, iter_num)

        with self.ghost.saved_tensors_context():
            with self.ctx:
                X_fwd, Y_fwd = self.ghost.prepare_forward_input(X_train, Y_train)
                outputs = self.model(X_fwd, Y_fwd)
                loss = outputs.loss
            self.scaler.scale(loss).backward()

        if do_step:
            # Recover the train-only gradient into .grad (subtract-val) for the
            # optimizer. Skipped on the scoring pass: the per-candidate score log is
            # produced by the backward hooks and read via aggregate_and_log() below,
            # so the (per-parameter) subtract-val recovery would only be discarded.
            self.ghost.prepare_gradients()
            self.scaler.unscale_(self.optimizer)
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Aggregate per-layer scores into the engine log, then read + reset.
        self.ghost.aggregate_and_log()
        log = self.ghost.engine.dot_product_log
        entry = log[-1] if log else None
        self.ghost.clear_gradients()
        log.clear()
        self.optimizer.zero_grad(set_to_none=True)

        # Two ghost passes per step (scoring over N, update over k) have different
        # batch sizes. In subtract-val mode the norm layers cache `layer.activations`
        # and never clear it (their cleanup backward hook is not registered), so the
        # scoring pass's activation would leak into the update pass and split with the
        # wrong train size. Clear the transient per-module capture state so each pass
        # resolves its own freshly-saved activations.
        self._clear_transient_layer_state()
        return loss, entry

    def _clear_transient_layer_state(self):
        for module in self.model.modules():
            for attr in ("activations", "backprops", "_ghost_saved_activation"):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _scores_from_entry(self, entry):
        if entry is None:
            raise RuntimeError(
                "Scoring pass produced no gradient dot products; check that the model "
                "contains supported layers (nn.Linear / nn.Embedding / ...)."
            )
        scores = entry["dot_product"].float()
        if self.config.select_metric == "cosine":
            train_norm = entry["train_grad_norm"].float()
            val_norm = float(entry.get("val_grad_norm", 1.0)) or 1.0
            scores = scores / (train_norm * val_norm + 1e-12)
        return scores

    # ------------------------------------------------------------------ #
    # Regular (no-selection) baseline step
    # ------------------------------------------------------------------ #
    def _regular_step(self, iter_num):
        t0 = time.time()
        X, Y, _ = self.get_batch("train", batch_size=self.config.batch_size, return_idx=True)

        lr = (get_learning_rate(iter_num, self.config)
              if self.config.decay_lr else self.config.learning_rate)
        update_learning_rate(self.optimizer, lr)

        with self.ctx:
            outputs = self.model(X, Y)
            loss = outputs.loss
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"Step {iter_num} | loss {loss.item():.4f} | lr {lr:.6f} | {dt:.3f}s")
        self._log_metrics(
            {"train/lr": lr, "train/loss": loss.item(), "train/step_time": dt},
            step=iter_num,
        )

    # ------------------------------------------------------------------ #
    # Evaluation / cleanup / logging
    # ------------------------------------------------------------------ #
    def _run_evaluation(self, result_file):
        if self.ghost is not None:
            self.ghost.detach_for_evaluation()
        losses = estimate_loss(self.model, self.get_batch, self.config, self.ctx)
        if self.ghost is not None:
            self.ghost.reattach_after_evaluation()

        train_loss, val_loss, test_loss = losses["train"], losses["val"], losses["test"]
        print(f"step {self.iter_num}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, test loss {test_loss:.4f}")
        save_training_results(result_file, train_loss, val_loss, test_loss, self.iter_num)
        self._log_metrics({
            "eval/train_loss": float(train_loss),
            "eval/val_loss": float(val_loss),
            "eval/test_loss": float(test_loss),
        }, step=self.iter_num)

    def _cleanup(self, result_file):
        print("Running cleanup ...")
        self._run_evaluation(result_file)
        if self.ghost is not None:
            self.ghost.cleanup()
        if self.wandb_run is not None and self.ddp_info["master_process"]:
            try:
                self.wandb_run.finish()
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to finalize Weights & Biases run: {e}")

    def _init_wandb(self):
        if not getattr(self.config, "use_wandb", False):
            return
        if not self.ddp_info["master_process"]:
            return
        try:
            import wandb
        except ImportError:
            print("[WARN] Weights & Biases is not installed; skipping wandb logging.")
            return

        config_payload = {
            "method": self.config.method,
            "architecture": self.config.architecture,
            "train_set": self.config.args.train_set,
            "batch_size": self.config.batch_size,
            "candidate_batch_size": self.config.candidate_batch_size,
            "val_batch_size": self.config.val_batch_size,
            "select_metric": self.config.select_metric,
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "seed": self.config.seed,
        }
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                mode=self.config.wandb_mode,
                dir=self.config.wandb_dir,
                config=config_payload,
            )
            print(f"[INFO] Weights & Biases enabled (run: {self.config.wandb_run_name}).")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None

    def _log_metrics(self, metrics, step=None):
        if self.wandb_run is None or not self.ddp_info["master_process"]:
            return
        try:
            self.wandb_run.log(metrics, step=step if step is not None else self.iter_num)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to log metrics to Weights & Biases: {e}")
