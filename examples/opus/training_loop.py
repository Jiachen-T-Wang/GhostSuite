"""OPUS online data selection — training loop.

Each step (method=OPUS):

  1. Draw a candidate pool of N = ``candidate_batch_size`` train samples and a fresh proxy
     batch of m = ``val_batch_size`` samples (from the eval window pool by default — the
     same protocol as the GREATS pretrain experiment).
  2. *Scoring pass* — ONE projection forward/backward over ``[candidates ++ proxy]`` giving
     per-sample gradient sketches; scores and the candidate Gram are two small GEMMs
     (see opus_scorer.OpusProjScorer). No optimizer step.
  3. *Select* k = ``batch_size`` candidates: Boltzmann stochastic-greedy with the Gram
     redundancy penalty (OPUS default), deterministic greedy, or plain top-k (ablation).
  4. *Update* — a PLAIN forward/backward + optimizer step on the selected k only (identical
     to the GREATS example's subset update; scoring hooks are detached here).

Method=Regular is the no-selection baseline: a plain step on a batch of k.
"""

import os
import sys
import time

import numpy as np
import torch

# Reuse shared/ utilities under examples/lm/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_THIS_DIR)
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
from ghostEngines import greedy_selection, stochastic_greedy_selection
# The subset update is exactly the GREATS driver's plain update step (no val concat, no
# capture); reused so the OPUS and GREATS arms share one update implementation.
from ghostEngines.selection_driver import _plain_update

from opus_scorer import OpusProjScorer


class OpusTrainer:
    """Online data selection trainer (OPUS: sketched scores + diversity-aware selection)."""

    def __init__(self, model, optimizer, scaler, config, ddp_info,
                 get_batch_fn, get_val_batch_fn, ctx):
        print("[INFO] Initializing OpusTrainer ...")

        if ddp_info.get("ddp", False):
            raise RuntimeError("The OPUS example is single-GPU only (no DDP).")

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
        self.is_opus = (config.method == "OPUS")

        self.scorer = None
        if self.is_opus:
            self.scorer = OpusProjScorer(model, optimizer, config, ddp_info["device"])
        # Selection randomness (Boltzmann sampling) is decoupled from the data-order RNG so
        # the drawn batches stay comparable with the GREATS/Regular arms at the same seed.
        self.select_rng = np.random.default_rng(config.seed)

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
        except Exception as e:  # noqa: BLE001 - mirror the GREATS trainer
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup(result_file)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select(self, scores: np.ndarray, gram: np.ndarray) -> torch.Tensor:
        k = self.config.batch_size
        method = self.config.opus_selection_method
        if method == "stochastic":
            idx = stochastic_greedy_selection(
                scores, gram, k, temperature=self.config.opus_temperature,
                rng=self.select_rng)
        elif method == "greedy":
            idx = greedy_selection(scores, gram, k)
        elif method == "topk":
            idx = np.argsort(-scores)[:k].tolist()
        else:
            raise ValueError(f"Unknown opus_selection_method {method!r}.")
        return torch.tensor(idx, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #
    def _train_step(self, iter_num):
        t0 = time.time()
        device = self.ddp_info["device"]

        lr = (get_learning_rate(iter_num, self.config)
              if self.config.decay_lr else self.config.learning_rate)
        update_learning_rate(self.optimizer, lr)

        if self.is_opus:
            # Fresh proxy batch each step (online selection targets the current draw).
            X_val, Y_val = self.get_val_batch(self.config.val_batch_size, return_idx=False)
            X_val = to_device(X_val, device)
            Y_val = to_device(Y_val, device)

            X, Y, _ = self.get_batch("train", batch_size=self.config.candidate_batch_size,
                                     return_idx=True)
            if isinstance(X, dict):
                raise NotImplementedError(
                    "The OPUS example supports tensor inputs (GPT-2) only.")

            scores_t, gram_t = self.scorer.score(
                self.forward_fn, self.ctx, self.scaler, X, Y, X_val, Y_val)
            scores, gram = scores_t.numpy(), gram_t.numpy()
            idx = self._select(scores, gram)

            dev_idx = idx.to(X.device)
            Xs = X.index_select(0, dev_idx)
            Ys = Y.index_select(0, dev_idx)
            loss = _plain_update(self.model, self.optimizer, self.scaler, self.ctx,
                                 self.forward_fn, Xs, Ys, self.config.grad_clip)
        else:
            X, Y, _ = self.get_batch("train", batch_size=self.config.batch_size,
                                     return_idx=True)
            scores, idx = None, None
            loss = _plain_update(self.model, self.optimizer, self.scaler, self.ctx,
                                 self.forward_fn, X, Y, self.config.grad_clip)

        torch.cuda.synchronize()
        dt = time.time() - t0
        self._log_step(iter_num, lr, loss, dt, scores, idx)

    def _log_step(self, iter_num, lr, loss, dt, scores, idx):
        if self.is_opus:
            sel_scores = scores[idx.numpy()]
            frac_pos = float((scores > 0).mean())
            k = self.config.batch_size
            print(f"Step {iter_num} | loss {loss.item():.4f} | lr {lr:.6f} | "
                  f"selected {k}/{self.config.candidate_batch_size} | "
                  f"mean sel score {sel_scores.mean():.3e} | "
                  f"frac>0 {frac_pos:.2f} | {dt:.3f}s")
            metrics = {
                "train/lr": lr,
                "train/loss": loss.item(),
                "train/step_time": dt,
                "select/mean_selected_score": float(sel_scores.mean()),
                "select/mean_candidate_score": float(scores.mean()),
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
        # Scoring hooks only live inside score(), so evaluation needs no detach.
        losses = estimate_loss(self.model, self.get_batch, self.config, self.ctx)
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
            "opus_selection_method": self.config.opus_selection_method,
            "opus_temperature": self.config.opus_temperature,
            "opus_preconditioner": self.config.opus_preconditioner,
            "proj_dim": self.config.proj_dim,
            "score_seq_len": self.config.score_seq_len,
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
