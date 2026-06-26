"""Pluggable per-step update/selection policies for the GradDotProd online-selection loop.

A policy is a pure function of the per-sample scores (the train-val gradient dot products the
ghost engine produces) plus two flags that tell the shared driver *how* to update:

- ``scores_needed`` — whether a ghost scoring pass runs at all. ``False`` is the no-selection
  baseline (a plain optimizer step on the whole drawn batch, no engine).
- ``reuse_recovery`` — whether the update reuses the scoring backward via subtract-val recovery
  over the **whole** scored train set (one backward; only valid when the update set is "all"),
  vs a fresh plain backward over a selected subset (two backwards).

The driver (``examples/lm/shared/selection_trainer.online_selection_step``) consumes these; the
policy itself never touches the model, optimizer, or engine, so it stays trivially testable.

Built-ins map onto the existing examples:
- ``UpdateAll``   — graddotprod_lm: score the batch (log dots), update on all via recovery.
- ``TopK``        — greats: keep the top-k by score, plain-update on them.
- ``BottomK``     — the rejected/ablation arm (sel50 bottom-fraction).
- ``Threshold``   — online sel50-style keep-above/keep-below a score threshold.
- ``NoSelection`` — the Regular baseline: no scoring, plain update on the whole batch.
"""

from typing import Optional

import torch


class SelectionPolicy:
    """Base class. Subclasses set the two flags and implement ``select``."""

    #: run a ghost scoring pass (False => no-selection baseline, plain update only).
    scores_needed: bool = True
    #: reuse subtract-val recovery over the whole scored set (True) vs a fresh backward on a subset.
    reuse_recovery: bool = False

    def select(self, scores: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the LongTensor of kept indices, or ``None`` to mean "all scored samples"."""
        raise NotImplementedError


class UpdateAll(SelectionPolicy):
    """Update on every scored sample, reusing the scoring backward (subtract-val). graddotprod_lm."""

    scores_needed = True
    reuse_recovery = True

    def select(self, scores: torch.Tensor) -> Optional[torch.Tensor]:
        return None


class TopK(SelectionPolicy):
    """Keep the ``k`` highest-scoring candidates; fresh plain update on them. GREATS."""

    scores_needed = True
    reuse_recovery = False

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(f"TopK requires k > 0, got {k}.")
        self.k = k

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.k, scores.numel())
        return torch.topk(scores, k).indices


class BottomK(SelectionPolicy):
    """Keep the ``k`` lowest-scoring candidates (the rejected/ablation arm)."""

    scores_needed = True
    reuse_recovery = False

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(f"BottomK requires k > 0, got {k}.")
        self.k = k

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.k, scores.numel())
        return torch.topk(scores, k, largest=False).indices


class Threshold(SelectionPolicy):
    """Keep candidates whose score is >= (``keep_above``) or < the threshold (online sel50-style).

    Falls back to the single best/worst candidate if the mask is empty, so a step always updates."""

    scores_needed = True
    reuse_recovery = False

    def __init__(self, threshold: float, keep_above: bool = True):
        self.threshold = float(threshold)
        self.keep_above = bool(keep_above)

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        mask = scores >= self.threshold if self.keep_above else scores < self.threshold
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:  # never produce an empty update batch
            best = torch.argmax(scores) if self.keep_above else torch.argmin(scores)
            idx = best.reshape(1)
        return idx


class NoSelection(SelectionPolicy):
    """No scoring pass; a plain optimizer step on the whole drawn batch. Regular baseline."""

    scores_needed = False
    reuse_recovery = False

    def select(self, scores: torch.Tensor) -> Optional[torch.Tensor]:
        return None
