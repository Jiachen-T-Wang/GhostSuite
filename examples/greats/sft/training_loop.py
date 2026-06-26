"""GREATS SFT trainer — first-order online batch selection for LoRA instruction tuning.

Per optimizer step:
  1. Draw a candidate pool of N = candidate_batch_size instruction samples.
  2. Scoring pass (GREATS only): one GradDotProd forward/backward over
     [candidate ++ val] gives per-candidate s_i = <g_i, g_val> over the LoRA params.
  3. Select the top-k = batch_size candidates.
  4. Detach the engine and take a plain LoRA step on the selected subset, then
     reattach for the next step. Plain update (not subtract-val) because with
     instruction masking + variable lengths the subtract-val sample-count scaling
     is not exact; this matches upstream's normal training step on selected inputs.

See docs/plans/greats_sft_phaseB_2026-06-25.md.
"""

import json
import os
import time

import torch

from data_utils import collate
from mmlu_accuracy import compute_mmlu_accuracy, compute_mmlu_perplexity
from gram_scorer import GramScorer, greedy_selection
from ghostEngines import GradDotProdEngine


class GreatsSFTTrainer:
    def __init__(self, model, optimizer, scheduler, tokenizer, config, device,
                 train_samples, val_samples, total_steps):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.total_steps = total_steps

        self.pad_id = tokenizer.pad_token_id
        self.is_greats = (config.method == "GREATS")
        self.global_step = 0
        self._results = []
        self._results_path = os.path.join(config.result_dir, f"{config.method}_test_acc.json")

        # Shuffled stream over the training samples.
        self._order = list(range(len(train_samples)))
        self._ptr = 0
        self._gen = torch.Generator().manual_seed(config.seed)
        self._reshuffle()

        # Per-step scoring val mini-batch is resampled from the n_val pool (upstream uses
        # val_batchsize=2, shuffle=True). The engine's val size is val_batchsize.
        self.val_bs = min(config.val_batchsize, len(self.val_samples))
        self._val_gen = torch.Generator().manual_seed(config.seed + 1)

        self.second_order = (config.selection == "second_order")
        self.engine = None
        self.gram_scorer = None
        if self.is_greats and self.second_order:
            # True GREATS: Gram-based greedy selection over the LoRA params.
            self.gram_scorer = GramScorer(self.model)
        elif self.is_greats:
            # First-order: top-k by <g_i, g_val> via the GradDotProd engine. Scores are
            # consumed in-memory each step (read + clear), so no on-disk score log is kept.
            self.engine = GradDotProdEngine(
                module=self.model,
                val_batch_size=self.val_bs,
                loss_reduction="mean",
                use_dummy_bias=False,
                dot_prod_save_path=None,
                log_grad_norms=config.log_grad_norms,
            )
            self.engine.attach(self.optimizer)

    def _sample_val(self):
        """Resample a fresh val mini-batch from the n_val pool each scoring step."""
        n = len(self.val_samples)
        idx = torch.randperm(n, generator=self._val_gen)[: self.val_bs].tolist()
        return [self.val_samples[i] for i in idx]

    # ------------------------------------------------------------------ #
    def _reshuffle(self):
        perm = torch.randperm(len(self.train_samples), generator=self._gen).tolist()
        self._order = perm
        self._ptr = 0

    def _next_samples(self, count):
        if self._ptr + count > len(self._order):
            self._reshuffle()
        idx = self._order[self._ptr:self._ptr + count]
        self._ptr += count
        return [self.train_samples[i] for i in idx]

    # ------------------------------------------------------------------ #
    def train(self):
        print(f"[INFO] Starting {self.config.method} SFT for {self.total_steps} steps "
              f"(k={self.config.batch_size}, N={self.config.candidate_batch_size}, "
              f"n_val={self.config.n_val}).")
        self.model.train()
        self._evaluate()  # baseline accuracy before any update
        while self.global_step < self.total_steps:
            t0 = time.time()
            if self.is_greats:
                loss = self._greats_step()
            else:
                loss = self._regular_step()
            dt = time.time() - t0

            if self.global_step % self.config.logging_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"step {self.global_step}/{self.total_steps} | "
                      f"loss {loss:.4f} | lr {lr:.2e} | {dt:.3f}s")

            if (self.config.eval_interval and self.global_step > 0
                    and self.global_step % self.config.eval_interval == 0):
                self._evaluate()

            self.global_step += 1

        self._evaluate()
        print("[INFO] Training completed.")

    # ------------------------------------------------------------------ #
    def _greats_step(self):
        candidates = self._next_samples(self.config.candidate_batch_size)
        k = self.config.batch_size

        if self.second_order:
            selected = self._select_second_order(candidates, k)
            # GramScorer capture is gated by a flag, so the update pass is already clean.
            return self._plain_update(selected)

        # First-order: top-k by <g_i, g_val> via the engine.
        scores = self._score_candidates(candidates)        # [N] cpu float
        topk = torch.topk(scores, k).indices.tolist()
        selected = [candidates[i] for i in topk]
        # No detach needed: the engine only installs its dot-product backward hook inside
        # saved_tensors_context() (used by the scoring pass), so the plain LoRA update below
        # — which runs outside that context — is already clean, with no per-step hook churn.
        return self._plain_update(selected)

    def _select_second_order(self, candidates, k):
        """True GREATS: Gram-based greedy selection weighted by (lr, lr^2)."""
        n = len(candidates)
        val_batch = self._sample_val()
        batch = collate(candidates + val_batch, self.pad_id, self.device)
        tracin, similarity = self.gram_scorer.score(
            batch["input_ids"], batch["attention_mask"], batch["labels"],
            n_train=n, n_val=len(val_batch),
        )
        lr = self.scheduler.get_last_lr()[0]
        selected_ind = greedy_selection(tracin * lr, similarity * (lr ** 2), k)
        return [candidates[i] for i in selected_ind]

    def _regular_step(self):
        batch_samples = self._next_samples(self.config.batch_size)
        return self._plain_update(batch_samples)

    def _score_candidates(self, candidates):
        n = len(candidates)
        val_batch = self._sample_val()
        batch = collate(candidates + val_batch, self.pad_id, self.device)
        # aggregate_and_log records X_train/Y_train in its log dict; set them.
        self.engine.attach_train_batch(
            batch["input_ids"][:n], batch["labels"][:n], self.global_step
        )
        self.optimizer.zero_grad(set_to_none=True)
        with self.engine.saved_tensors_context():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
        self.engine.aggregate_and_log()
        log = self.engine.dot_product_log
        entry = log[-1] if log else None
        log.clear()
        if entry is None:
            raise RuntimeError("Scoring produced no dot products; check trainable LoRA layers.")
        scores = entry["dot_product"].float()
        if self.config.select_metric == "cosine":
            tn = entry["train_grad_norm"].float()
            vn = float(entry.get("val_grad_norm", 1.0)) or 1.0
            scores = scores / (tn * vn + 1e-12)
        return scores

    def _plain_update(self, samples):
        batch = collate(samples, self.pad_id, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.config.grad_clip
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    @torch.no_grad()
    def _evaluate(self):
        """Headline metric: MMLU eval/test answer perplexity (as in upstream trialrun.png);
        also reports few-shot test accuracy."""
        eval_ppl, test_ppl = compute_mmlu_perplexity(
            self.model, self.tokenizer, self.config.data_dir, self.config.subject,
            n_val=self.config.n_val, n_test=self.config.n_test, device=self.device,
            max_seq_length=self.config.max_seq_length,
        )
        acc, n = compute_mmlu_accuracy(
            self.model, self.tokenizer, self.config.data_dir, self.config.subject,
            n_val=self.config.n_val, n_test=self.config.n_test, device=self.device,
        )
        print(f"  [eval] step {self.global_step} | '{self.config.subject}' "
              f"eval_ppl {eval_ppl:.4f} | test_ppl {test_ppl:.4f} | test_acc {acc:.4f} "
              f"(n={n})", flush=True)
        self._results.append({"step": self.global_step, "eval_ppl": eval_ppl,
                              "test_ppl": test_ppl, "test_acc": acc, "n_test": n})
        with open(self._results_path, "w") as f:
            json.dump(self._results, f, indent=2)
        return test_ppl
