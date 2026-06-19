"""Decoupled capture for compile-compatible ghost dot-products (`GHOST_DECOUPLED_FN=1`).

Goal (re-examination doc §3): realize the *model-compile* win (~+27% ceiling) that lever 1b
left on the table, **without** Phase 2's two regressions.

Phase 2 (`autograd_function_dotprod.py`) compiled the model but (a) replaced each layer's native
fused backward with hand-rolled matmuls (2x slower eager) and (b) fused the dot-product *into*
the joint backward graph via a ``+0.0*marker`` live-chain, which pinned heavy intermediates and
blew peak memory +38%. Both are self-inflicted by computing the dot-product inside the graph.

This module decouples instead:

  * A **transparent identity** ``autograd.Function`` (``_CaptureFn``) is applied to each
    supported layer's *output*. Its forward is the identity (the layer's own op runs natively,
    so cuBLAS/cuDNN fused backward is preserved — no 2x penalty). Its backward does **no
    dot-product math**: it just stores the layer input ``A`` and the output gradient ``B`` into
    preallocated per-layer buffers via an opaque buffer-mutating custom op, and passes the
    grad straight through.
  * After ``loss.backward()`` the captured ``(A, B)`` are fed to the **proven 1b grouped,
    compiled, post-backward pass** (``batched_dotprod.run_batched_dotprod``) — one batched op
    per shape-group across all 32 blocks, never inside the joint graph.

So the model regional-compiles (the big lever) while the dot-product stays decoupled (1b's win),
and the joint graph carries no dot-product compute to pin. Train grads are recovered via
subtract-val from autograd ``.grad`` and the buffered ``grad_val``.

Gated behind ``GHOST_DECOUPLED_FN=1`` (requires ``GHOST_SUBTRACT_VAL=1``). Default path untouched.
FAIL-LOUD on any unsupported parameterized leaf (plan §6 decision 4).
"""

from typing import Dict, List, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batched_dotprod import run_batched_dotprod


_DECOUPLED_FN = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"

_SUPPORTED = (nn.Linear, nn.Embedding, nn.RMSNorm)


# Opaque buffer write. Stores a *live* tensor (the layer input A, or the output grad B) into a
# preallocated external buffer from inside the capture Function's backward. Registered as a
# custom op with ``mutates_args`` so functionalization keeps it (a bare ``buf.copy_()`` becomes
# an "invalid graph output" under the partitioner). Returns a fresh fp32 scalar marker (never
# aliasing ``buf``) that the backward ties into the passthrough grad via ``+0.0*marker`` so the
# store node is not dropped as a dead side effect.
@torch.library.custom_op("ghost::capture_store", mutates_args={"buf"})
def _capture_store(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    buf.copy_(val)
    return torch.zeros((), dtype=torch.float32, device=val.device)


@_capture_store.register_fake
def _capture_store_fake(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=val.device)


class _CaptureFn(torch.autograd.Function):
    """Identity in forward; stores (A, B) into preallocated buffers in backward.

    ``output``   : the layer's native output (passed straight through).
    ``input_act``: the layer's input activation A (saved; stored in backward). For Embedding
                   this is the integer index tensor.
    ``a_buf`` / ``b_buf``: preallocated per-layer buffers written in-place by backward.
    """

    @staticmethod
    def forward(ctx, output, input_act, a_buf, b_buf):
        ctx.save_for_backward(input_act)
        ctx.a_buf = a_buf
        ctx.b_buf = b_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_act,) = ctx.saved_tensors
        m1 = torch.ops.ghost.capture_store(ctx.a_buf, input_act)
        m2 = torch.ops.ghost.capture_store(ctx.b_buf, grad_output)
        # Keep both stores live for the partitioner; grad passes straight through to the layer's
        # native backward. input_act gets no grad from this Function (None).
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        return grad_out, None, None, None


# ---------------------------------------------------------------------------------------
# Manager: wrap supported modules so forward = native-op then _CaptureFn; own per-layer buffers.
# ---------------------------------------------------------------------------------------


class GhostDecoupledManager:
    """Swaps supported modules' forward to ``native_op -> _CaptureFn`` and owns capture buffers.

    Buffers are lazily allocated on the first eager forward (the warmup before compile) and
    reused, so a later regional ``torch.compile`` traces against fixed-shape buffers. Post-
    backward, ``run_step_dotprod`` feeds the captured (A, B) to the grouped 1b pass and recovers
    subtract-val train grads.
    """

    def __init__(self, model: nn.Module, val_batch_size: int) -> None:
        self.model = model
        self.val_batch_size = val_batch_size
        self._orig_forward: Dict[int, object] = {}
        self._layers: List[Tuple[str, nn.Module]] = []
        self._cells: Dict[int, list] = {}  # id(layer) -> [a_buf, b_buf] cell

    # -- forward wrappers -----------------------------------------------------------------

    def _wrap_linear(self, layer: nn.Linear):
        cell = [None, None]
        self._cells[id(layer)] = cell
        weight = layer.weight

        def forward(x):
            out = F.linear(x, weight)
            if cell[0] is None:  # eager warmup only; not traced under compile
                cell[0] = torch.empty_like(x)
                cell[1] = torch.empty_like(out)
            return _CaptureFn.apply(out, x, cell[0], cell[1])

        return forward

    def _wrap_embedding(self, layer: nn.Embedding):
        cell = [None, None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        padding_idx = layer.padding_idx

        def forward(idx):
            out = F.embedding(idx, weight, padding_idx)
            if cell[0] is None:
                cell[0] = torch.empty_like(idx)
                cell[1] = torch.empty_like(out)
            return _CaptureFn.apply(out, idx, cell[0], cell[1])

        return forward

    def _wrap_rmsnorm(self, layer: nn.RMSNorm):
        cell = [None, None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        normalized_shape = tuple(layer.normalized_shape)
        eps = layer.eps if layer.eps is not None else 1e-5

        def forward(x):
            out = F.rms_norm(x, normalized_shape, weight, eps)
            if cell[0] is None:
                cell[0] = torch.empty_like(x)
                cell[1] = torch.empty_like(out)
            return _CaptureFn.apply(out, x, cell[0], cell[1])

        return forward

    # -- attach / detach ------------------------------------------------------------------

    def attach(self) -> None:
        for name, layer in self.model.named_modules():
            if isinstance(layer, _SUPPORTED):
                if any(p.requires_grad for p in layer.parameters(recurse=False)):
                    if isinstance(layer, nn.Linear) and layer.bias is not None:
                        raise RuntimeError(
                            f"GHOST_DECOUPLED_FN: Linear '{name}' has a bias; biases are not "
                            "supported by the decoupled-capture path."
                        )
                    setattr(layer, "name", name)
                    self._layers.append((name, layer))
                    self._orig_forward[id(layer)] = layer.forward
                    if isinstance(layer, nn.Linear):
                        layer.forward = self._wrap_linear(layer)
                    elif isinstance(layer, nn.Embedding):
                        layer.forward = self._wrap_embedding(layer)
                    elif isinstance(layer, nn.RMSNorm):
                        layer.forward = self._wrap_rmsnorm(layer)
            else:
                is_leaf = not list(layer.children())
                if is_leaf and any(p.requires_grad for p in layer.parameters(recurse=False)):
                    raise RuntimeError(
                        f"GHOST_DECOUPLED_FN: unsupported parameterized layer '{name}' "
                        f"({type(layer).__name__}). Supported: {[t.__name__ for t in _SUPPORTED]}."
                    )

    def warmup(self, example_input: torch.Tensor) -> None:
        """One eager forward+backward so every layer's capture buffers are allocated OUTSIDE the
        compiled graph. ``example_input`` is a combined train+val token batch of the SAME shape
        used in training."""
        was_training = self.model.training
        logits = self.model(example_input)
        loss = logits.float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)

    def detach(self) -> None:
        for _, layer in self._layers:
            orig = self._orig_forward.get(id(layer))
            if orig is not None:
                layer.forward = orig
        self._layers.clear()
        self._orig_forward.clear()
        self._cells.clear()

    # -- post-backward --------------------------------------------------------------------

    def run_step_dotprod(self) -> Optional[torch.Tensor]:
        """Feed captured (A, B) to the grouped 1b pass; set per-layer grad_dot_prod /
        _ghost_grad_val. Returns the aggregated per-train-sample dot-product [train_bs]."""
        pending: List[Tuple[nn.Module, torch.Tensor, torch.Tensor]] = []
        for _, layer in self._layers:
            cell = self._cells.get(id(layer))
            if cell is None or cell[0] is None:
                raise RuntimeError(
                    f"GHOST_DECOUPLED_FN: layer '{getattr(layer, 'name', '?')}' has no capture "
                    "buffer; warmup must run before the first traced step."
                )
            pending.append((layer, cell[0], cell[1]))

        run_batched_dotprod(pending, self.val_batch_size)

        total = None
        for _, layer in self._layers:
            dp = getattr(layer.weight, "grad_dot_prod", None)
            if dp is None:
                continue
            total = dp.detach().clone() if total is None else total + dp.detach()
        return total

    def recover_train_grads(self) -> None:
        """subtract-val: param.grad <- (total/train)*(autograd_grad - grad_val). Run after
        backward + run_step_dotprod (which populates _ghost_grad_val), before optimizer.step."""
        for name, layer in self._layers:
            param = layer.weight
            grad_val = getattr(param, "_ghost_grad_val", None)
            if grad_val is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' has no _ghost_grad_val.")
            if param.grad is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' weight has no autograd .grad.")
            # total_bs is the captured activation's leading dim; train half = total - val.
            total_bs = self._cells[id(layer)][0].shape[0]
            train_bs = total_bs - self.val_batch_size
            scale = float(total_bs) / float(train_bs)
            param.grad = (scale * (param.grad.float() - grad_val)).to(param.grad.dtype)
            if hasattr(param, "_ghost_grad_val"):
                del param._ghost_grad_val
            if hasattr(param, "grad_dot_prod"):
                del param.grad_dot_prod
