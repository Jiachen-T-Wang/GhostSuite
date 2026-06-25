"""Second-order GREATS scoring: per-sample LoRA gradient inner products.

Computes, in one forward/backward over [candidate ++ val], the first-order TracIN
score `tracin_i = <g_i, g_val>` AND the candidate-candidate Gram
`similarity_ij = <g_i, g_j>` over the LoRA adapter parameters, then runs the greedy
redundancy-aware selection. This is the full GREATS algorithm (NeurIPS 2024), ported
from the up-to-date upstream `less/train/utils_ghost_dot_prod.compute_GradProd_GC_per_iter`
+ `greedy_selection`.

The per-sample LoRA gradient `G_i = sum_t b_{i,t} a_{i,t}^T` is materialized per layer
(cheap: LoRA factors are r x d_in / d_out x r), so the Gram is just `G @ G^T`. Capture
is gated by an `_enabled` flag, so the plain update pass (capture off) sees no hooks and
needs no detach.
"""

import numpy as np
import torch
from torch import nn


def greedy_selection(scores: np.ndarray, interaction: np.ndarray, K: int):
    """Greedily pick K indices, subtracting each pick's interaction row from the
    remaining scores (redundancy penalty). Port of upstream `greedy_selection`."""
    scores = scores.copy().astype(np.float64)
    selected = []
    for _ in range(K):
        i = int(np.argmax(scores))
        selected.append(i)
        scores = scores - interaction[i, :]
        scores[i] = -np.inf
    return selected


class GramScorer:
    """Captures per-LoRA-layer (input activation, output grad) and builds the GREATS
    first- and second-order scores."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._enabled = False
        self._handles = []
        self.layers = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and ("lora_A" in name or "lora_B" in name):
                if any(p.requires_grad for p in mod.parameters(recurse=False)):
                    self.layers.append(mod)
        if not self.layers:
            raise RuntimeError("GramScorer found no trainable LoRA Linear layers.")
        self._register()

    def _register(self):
        for layer in self.layers:
            def fwd_pre(mod, args):
                if self._enabled and args and isinstance(args[0], torch.Tensor):
                    mod._gram_A = args[0].detach()
            def fwd(mod, args, out):
                if self._enabled and isinstance(out, torch.Tensor) and out.requires_grad:
                    def _save(grad, m=mod):
                        m._gram_B = grad.detach()
                    out.register_hook(_save)
            self._handles.append(layer.register_forward_pre_hook(fwd_pre))
            self._handles.append(layer.register_forward_hook(fwd))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def score(self, input_ids, attention_mask, labels, n_train, n_val):
        """Return (tracin[n_train], similarity[n_train, n_train]) as numpy arrays."""
        self._enabled = True
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Per-sample loss = mean CE over each sample's own valid (non-masked) tokens,
        # so each sample is weighted equally (upstream uses loss.mean() of these).
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        pos_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_labels.size())
        mask = (shift_labels != -100).float()
        per_sample = (pos_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

        per_sample.mean().backward()
        self._enabled = False

        tracin = torch.zeros(n_train, dtype=torch.float64)
        similarity = torch.zeros(n_train, n_train, dtype=torch.float64)
        for layer in self.layers:
            A = getattr(layer, "_gram_A", None)
            B = getattr(layer, "_gram_B", None)
            if A is None or B is None:
                continue
            # Per-sample weight grad G_i = sum_t b_{i,t} a_{i,t}^T  -> [n, d_out, d_in]
            G = torch.bmm(B.float().transpose(1, 2), A.float()).flatten(1)  # [n, d_out*d_in]
            Gtr, Gval = G[:n_train], G[n_train:]
            tracin += (Gtr @ Gval.T).mean(dim=1).double().cpu()   # mean over val samples
            similarity += (Gtr @ Gtr.T).double().cpu()
            if hasattr(layer, "_gram_A"):
                del layer._gram_A
            if hasattr(layer, "_gram_B"):
                del layer._gram_B
        self.model.zero_grad(set_to_none=True)
        return tracin.numpy(), similarity.numpy()
