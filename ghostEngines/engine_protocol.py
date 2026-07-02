"""Shared engine interface.

`GhostEngineManager` drives concrete engines (currently ``GradDotProdEngine``; the projection
engine ``GradProjLoraEngine`` is driven directly by its examples, not by the manager) through a
small lifecycle. This Protocol declares the **mandatory, common** subset of that lifecycle so the
manager can be typed against an interface instead of duck-typing, and so a new engine has an
explicit contract to implement.

It is intentionally minimal: only the methods every manager-driven engine must provide.
Engine-specific capabilities the manager invokes opportunistically — e.g.
``saved_tensors_context`` / ``accumulate_microbatch`` (GradDotProd) — are deliberately NOT part
of this Protocol; the manager guards those with ``hasattr`` so an engine that lacks them degrades
gracefully rather than being forced to define no-ops.
"""

from typing import Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class GhostEngine(Protocol):
    """The mandatory lifecycle every ghost engine implements (see module docstring)."""

    def attach(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Register hooks. ``optimizer`` is used by engines that update grads (GradDotProd);
        engines that only observe gradients may ignore it."""
        ...

    def detach(self) -> None:
        """Remove hooks and clean up engine-stamped attributes."""
        ...

    def attach_train_batch(self, X_train, Y_train, iter_num, batch_idx=None) -> None:
        """Record the current training batch / iteration for logging."""
        ...

    def prepare_gradients(self) -> None:
        """Post-backward: make the train-only gradients available on ``param.grad``."""
        ...

    def aggregate_and_log(self) -> None:
        """Aggregate this step's per-sample metrics into the engine's in-memory log."""
        ...

    def clear_gradients(self) -> None:
        """Post-optimizer-step: drop transient per-step gradient state."""
        ...
