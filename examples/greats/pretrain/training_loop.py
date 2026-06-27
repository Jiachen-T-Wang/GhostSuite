"""GREATS online batch selection — training loop.

First-order online selection (no pairwise Gram, no greedy second-order term). Each
step:

  1. Draw a candidate pool of N = ``candidate_batch_size`` train samples and a fresh
     scoring val batch of m = ``val_batch_size`` samples (from the eval window pool by
     default).
  2. *Scoring pass* — one GradDotProd forward/backward over ``[candidate ++ val]`` giving
     per-candidate ``s_i = <g_i, g_val>`` (cosine optional). No optimizer step. This is the
     only ghost pass; by default it runs the decoupled in-graph + ``torch.compile`` fast path.
  3. Select the top-k candidates (k = ``batch_size``).
  4. *Update* — a PLAIN forward/backward + optimizer step on the selected k only (no val, no
     ghost). This is exact: subtract-val over ``[selected ++ val]`` recovers the mean train
     gradient over the selected k, which equals a plain mean-loss backward over those k — so
     the update needs neither the val batch nor the dot-product machinery (its dots were
     discarded anyway). Dropping the val (m) forwards + ghost overhead from the update is the
     ~25% per-step speedup vs running a second ghost pass.
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
from ghostEngines import GhostEngineManager, TopK, NoSelection
from shared.selection_trainer import online_selection_step


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
            # The decoupled fast path reads grad_val from the UNSCALED backward, so it requires
            # the GradScaler disabled (true for bf16 training; fp16 is unsupported on this path).
            self.is_fn = self.ghost.is_fn_path
            if self.is_fn and self.scaler.is_enabled():
                raise RuntimeError(
                    "The decoupled fast path requires the GradScaler disabled. Use bf16 "
                    "(--train_dtype bfloat16) or select --eager."
                )
        else:
            self.is_fn = False

        # Update rule as a pluggable policy: GREATS keeps the top-k by score (fresh plain update on
        # them); Regular takes no scoring pass and a plain step on the whole batch.
        self.policy = TopK(config.batch_size) if self.is_greats else NoSelection()
        self.forward_fn = lambda m, X, Y: m(X, Y).loss

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
                    self._train_step(self.iter_num)
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
    # Training step (GREATS selection or the Regular baseline, via the shared driver)
    # ------------------------------------------------------------------ #
    def _train_step(self, iter_num):
        t0 = time.time()

        if self.is_greats:
            # Fresh scoring val batch (online selection targets the current val draw).
            X_val, Y_val = self.get_val_batch(self.config.val_batch_size, return_idx=False)
            X_val = to_device(X_val, self.ddp_info["device"])
            Y_val = to_device(Y_val, self.ddp_info["device"])
            self.ghost.update_validation_batch(X_val, Y_val)
            draw = self.config.candidate_batch_size   # score N candidates, keep top-k
        else:
            draw = self.config.batch_size             # Regular: a normal batch of k

        X, Y, _ = self.get_batch("train", batch_size=draw, return_idx=True)
        if isinstance(X, dict):
            raise NotImplementedError(
                "The GREATS pretrain example supports tensor inputs (GPT-2) only."
            )

        lr = (get_learning_rate(iter_num, self.config)
              if self.config.decay_lr else self.config.learning_rate)
        update_learning_rate(self.optimizer, lr)

        # The shared driver runs the scoring pass (if any), applies the policy, and does the update
        # (subtract-val recovery for UpdateAll, or a plain backward on the selected subset for TopK).
        scores, idx, loss = online_selection_step(
            manager=self.ghost, model=self.model, optimizer=self.optimizer,
            scaler=self.scaler, ctx=self.ctx, forward_fn=self.forward_fn, X=X, Y=Y,
            policy=self.policy, iter_num=iter_num, grad_clip=self.config.grad_clip,
            score_metric=self.config.select_metric,
        )

        torch.cuda.synchronize()
        dt = time.time() - t0
        self._log_step(iter_num, lr, loss, dt, scores, idx)

    def _log_step(self, iter_num, lr, loss, dt, scores, idx):
        if self.is_greats:
            sel_scores = scores.index_select(0, idx)
            frac_pos = (scores > 0).float().mean().item()
            k = self.config.batch_size
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
        else:
            print(f"Step {iter_num} | loss {loss.item():.4f} | lr {lr:.6f} | {dt:.3f}s")
            metrics = {"train/lr": lr, "train/loss": loss.item(), "train/step_time": dt}
        self._log_metrics(metrics, step=iter_num)

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
