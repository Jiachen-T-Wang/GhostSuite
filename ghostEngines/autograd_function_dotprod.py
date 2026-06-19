"""Phase 2: graph-clean dot-product backward via custom ``torch.autograd.Function``.

The default ghost engine (``autograd_grad_sample_dotprod``) smuggles per-sample dot-products
out of the eager backward through tensor-level ``register_hook`` callbacks, a thread-locked
``_NamedSavedTensorManager``, and ``setattr`` on params/modules. None of that is traceable, so
``torch.compile`` cannot fuse the model fwd/bwd with the dot-products (see
``docs/investigations/compile_incompatibility_rootcause_2026-06-18.md``).

This module replaces that mechanism for the llama3 set (``nn.Linear`` / ``nn.Embedding`` /
``nn.RMSNorm``) with **custom autograd Functions**. Each supported layer's ``forward`` is
swapped to call a Function whose:

  * ``forward`` runs the real op and ``ctx.save_for_backward``s the activation (no manager,
    no lock, no scope stack);
  * ``backward`` returns ``grad_input`` **and** computes the per-sample dot-product and the
    subtract-val ``grad_val`` as ordinary tensor ops, written into **preallocated per-layer
    buffers** that are themselves Function inputs (so the writes functionalize under
    AOTAutograd; no ``setattr`` on params, no id-keyed dict).

The dot-product MATH mirrors ``supported_layers_grad_samplers_dotprod`` exactly (the ghost
associativity formula for Linear; index_add for Embedding; weight-only for RMSNorm), so the
results match the eager oracle to fp32 order noise. subtract-val semantics are preserved: the
engine recovers the train grad post-backward as ``(total/train)*(autograd_grad - grad_val)``.

Gated behind ``GHOST_AUTOGRAD_FN=1``. Default path untouched.

FAIL-LOUD: under the flag, any parameterized leaf layer not in the supported set (LayerNorm,
Conv1D, Conv2d, biases) raises -- no silent eager fallback (plan §6 decision 4).
"""

from typing import Dict, List, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


ACCUM_DTYPE = torch.float32

_AUTOGRAD_FN = os.getenv("GHOST_AUTOGRAD_FN", "0") == "1"

# Types handled by the custom-Function path. Anything else parameterized fails loudly.
_SUPPORTED = (nn.Linear, nn.Embedding, nn.RMSNorm)


# ---------------------------------------------------------------------------------------
# Per-layer buffer bundle. Holds preallocated output tensors the Function backward writes.
# ---------------------------------------------------------------------------------------


class _GhostBuffers:
    """Preallocated per-layer output buffers, threaded into the custom Function.

    ``dot``    : [train_bs]                 per-train-sample dot-product
    ``grad_val``: weight-shaped (Linear: [d_out, d_in]; Embedding: [vocab, d]; RMSNorm: [d])
    """

    __slots__ = ("dot", "grad_val")

    def __init__(self, dot: torch.Tensor, grad_val: torch.Tensor) -> None:
        self.dot = dot
        self.grad_val = grad_val


# ---------------------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------------------


class _GhostLinearFn(torch.autograd.Function):
    """nn.Linear forward + graph-resident dot-product/grad_val emission in backward.

    Extra (non-differentiable) inputs ``dot_buf`` / ``gradval_buf`` are preallocated buffers
    written in-place by backward; ``train_bs`` / ``val_bs`` are static python ints baked in
    at trace time (no ``.item()``).
    """

    @staticmethod
    def forward(ctx, input, weight, dot_buf, gradval_buf, train_bs, val_bs):
        ctx.save_for_backward(input, weight)
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        ctx.val_bs = val_bs
        return F.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        train_bs = ctx.train_bs
        val_bs = ctx.val_bs

        # Standard linear grads. grad_weight is the FULL combined-batch grad; subtract-val
        # recovers the train grad post-backward as (total/train)*(grad_weight - grad_val).
        grad_input = grad_output.matmul(weight)
        # grad_weight = sum_n grad_output_n^T @ input_n over the flattened (batch*seq) rows.
        go2d = grad_output.reshape(-1, grad_output.shape[-1])
        in2d = input.reshape(-1, input.shape[-1])
        grad_weight = go2d.t().matmul(in2d)

        # --- ghost associativity dot-product (mirrors _compute_linear_dot_product) ---
        A = input.detach()
        B = grad_output.detach()
        d_in = A.shape[-1]
        d_out = B.shape[-1]
        seq = A.shape[1]
        compute_dtype = B.dtype if B.is_floating_point() else weight.dtype

        A_flat = A.to(compute_dtype).reshape(-1, d_in)
        B_flat = B.to(compute_dtype).reshape(-1, d_out)
        split = train_bs * seq
        A_train = A_flat[:split]
        A_val = A_flat[split:]
        B_train = B_flat[:split]
        B_val = B_flat[split:]

        grad_val = torch.matmul(B_val.t(), A_val)                  # [d_out, d_in]
        grad_val_projected = torch.matmul(B_train, grad_val)       # [train_bs*seq, d_in]
        token_scores = (A_train * grad_val_projected).sum(dim=1)   # [train_bs*seq]
        dot = token_scores.view(train_bs, seq).sum(dim=1)          # [train_bs]

        ctx.dot_buf.copy_(dot)
        ctx.gradval_buf.copy_(grad_val.to(ACCUM_DTYPE))

        # grads for (input, weight, dot_buf, gradval_buf, train_bs, val_bs)
        return grad_input, grad_weight, None, None, None, None


# ---------------------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------------------


class _GhostEmbeddingFn(torch.autograd.Function):
    """nn.Embedding forward + graph-resident dot-product/grad_val in backward."""

    @staticmethod
    def forward(ctx, weight, input_idx, dot_buf, gradval_buf, train_bs, val_bs, padding_idx):
        ctx.save_for_backward(input_idx, weight)
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        ctx.val_bs = val_bs
        ctx.padding_idx = padding_idx
        return F.embedding(input_idx, weight, padding_idx)

    @staticmethod
    def backward(ctx, grad_output):
        input_idx, weight = ctx.saved_tensors
        train_bs = ctx.train_bs

        # --- embedding dot-product (mirrors _compute_embedding_dot_product) ---
        B = grad_output.detach()
        compute_dtype = B.dtype if B.is_floating_point() else weight.dtype
        idx = input_idx.long()
        idx_train = idx[:train_bs]
        idx_val = idx[train_bs:]
        B_train = B[:train_bs].to(compute_dtype)
        B_val = B[train_bs:].to(compute_dtype)

        vocab, d_f = weight.shape
        grad_val = torch.zeros((vocab, d_f), dtype=compute_dtype, device=B.device)
        grad_val.index_add_(0, idx_val.reshape(-1), B_val.reshape(-1, d_f))

        dot = (B_train * grad_val[idx_train]).to(ACCUM_DTYPE).sum(dim=[1, 2])

        ctx.dot_buf.copy_(dot)
        ctx.gradval_buf.copy_(grad_val.to(ACCUM_DTYPE))

        # The standard embedding weight grad is needed for subtract-val recovery; let autograd
        # produce it by returning the embedding grad_weight. F.embedding's native backward
        # builds it from grad_output; we replicate via index_add over the full batch.
        grad_weight = torch.zeros_like(weight)
        grad_weight.index_add_(0, idx.reshape(-1), B.reshape(-1, d_f).to(weight.dtype))
        if ctx.padding_idx is not None and ctx.padding_idx >= 0:
            grad_weight[ctx.padding_idx] = 0

        # grads for (weight, input_idx, dot_buf, gradval_buf, train_bs, val_bs, padding_idx)
        return grad_weight, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------------------


class _GhostRMSNormFn(torch.autograd.Function):
    """nn.RMSNorm forward + graph-resident weight dot-product/grad_val in backward."""

    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, dot_buf, gradval_buf, train_bs, val_bs):
        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        ctx.val_bs = val_bs
        return F.rms_norm(input, normalized_shape, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        train_bs = ctx.train_bs
        eps = ctx.eps

        # --- grad_input for RMSNorm (fp32, mirrors _compute_rmsnorm_grad_input) ---
        x_f = input.to(ACCUM_DTYPE)
        go_f = grad_output.to(ACCUM_DTYPE)
        go_w = go_f * weight.to(ACCUM_DTYPE)
        inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        x_hat = x_f * inv_rms
        go_xhat_mean = (go_w * x_hat).mean(dim=-1, keepdim=True)
        grad_input = (inv_rms * (go_w - x_hat * go_xhat_mean)).to(grad_output.dtype)

        # --- weight dot-product + grad_val (mirrors _compute_rmsnorm_dot_product) ---
        A = x_f
        B = go_f
        A_train = A[:train_bs]
        A_val = A[train_bs:]
        B_train = B[:train_bs]
        B_val = B[train_bs:]
        rms_train = torch.sqrt((A_train ** 2).mean(dim=-1, keepdim=True) + eps)
        rms_val = torch.sqrt((A_val ** 2).mean(dim=-1, keepdim=True) + eps)
        norm_A_train = A_train / rms_train
        norm_A_val = A_val / rms_val
        gw_train = B_train * norm_A_train
        gw_val = B_val * norm_A_val
        sum_dims_train = list(range(1, gw_train.dim() - 1))
        per_sample = gw_train.sum(dim=sum_dims_train) if sum_dims_train else gw_train  # [train,d]
        total_val = gw_val.sum(dim=list(range(gw_val.dim() - 1)))                      # [d]
        dot = torch.einsum("bf,f->b", per_sample, total_val)

        ctx.dot_buf.copy_(dot)
        ctx.gradval_buf.copy_(total_val.to(ACCUM_DTYPE))

        # weight grad: let autograd accumulate the full combined-batch grad for subtract-val.
        norm_A_full = A * torch.rsqrt((A ** 2).mean(dim=-1, keepdim=True) + eps)
        grad_weight = (B * norm_A_full).sum(dim=list(range(A.dim() - 1))).to(weight.dtype)

        # grads for (input, weight, normalized_shape, eps, dot_buf, gradval_buf, train_bs, val_bs)
        return grad_input, grad_weight, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------
# Module wiring: swap supported modules' forward to route through the custom Functions.
# ---------------------------------------------------------------------------------------


class GhostAutogradFnManager:
    """Swaps supported modules' forward to the custom Functions and owns per-layer buffers.

    Buffers are lazily allocated on the first forward (shapes known then) and reused, so a
    later ``torch.compile`` traces against fixed-shape buffers. Post-backward, the engine reads
    ``dot``/``grad_val`` straight off each layer's buffer bundle.
    """

    def __init__(self, model: nn.Module, val_batch_size: int) -> None:
        self.model = model
        self.val_batch_size = val_batch_size
        self._orig_forward: Dict[int, object] = {}
        self._layers: List[Tuple[str, nn.Module]] = []
        self._buffers: Dict[int, _GhostBuffers] = {}

    # -- buffer helpers -------------------------------------------------------------------

    def _get_or_alloc(self, layer, train_bs, grad_val_shape, device, weight) -> _GhostBuffers:
        key = id(layer)
        buf = self._buffers.get(key)
        if buf is None:
            dot = torch.zeros(train_bs, dtype=ACCUM_DTYPE, device=device)
            grad_val = torch.zeros(grad_val_shape, dtype=ACCUM_DTYPE, device=device)
            buf = _GhostBuffers(dot, grad_val)
            self._buffers[key] = buf
            layer._ghost_buffers = buf
        return buf

    # -- forward wrappers -----------------------------------------------------------------

    def _wrap_linear(self, layer: nn.Linear):
        mgr = self
        vbs = self.val_batch_size

        def forward(x, _layer=layer):
            total_bs = x.shape[0]
            train_bs = total_bs - vbs
            d_in = _layer.weight.shape[1]
            d_out = _layer.weight.shape[0]
            buf = mgr._get_or_alloc(
                _layer, train_bs, (d_out, d_in), x.device, _layer.weight
            )
            return _GhostLinearFn.apply(
                x, _layer.weight, buf.dot, buf.grad_val, train_bs, vbs
            )

        return forward

    def _wrap_embedding(self, layer: nn.Embedding):
        mgr = self
        vbs = self.val_batch_size

        def forward(idx, _layer=layer):
            total_bs = idx.shape[0]
            train_bs = total_bs - vbs
            vocab, d = _layer.weight.shape
            buf = mgr._get_or_alloc(_layer, train_bs, (vocab, d), idx.device, _layer.weight)
            padding_idx = _layer.padding_idx
            return _GhostEmbeddingFn.apply(
                _layer.weight, idx, buf.dot, buf.grad_val, train_bs, vbs, padding_idx
            )

        return forward

    def _wrap_rmsnorm(self, layer: nn.RMSNorm):
        mgr = self
        vbs = self.val_batch_size
        normalized_shape = tuple(layer.normalized_shape)
        eps = layer.eps if layer.eps is not None else 1e-5

        def forward(x, _layer=layer):
            total_bs = x.shape[0]
            train_bs = total_bs - vbs
            d = _layer.weight.shape[0]
            buf = mgr._get_or_alloc(_layer, train_bs, (d,), x.device, _layer.weight)
            return _GhostRMSNormFn.apply(
                x, _layer.weight, normalized_shape, eps,
                buf.dot, buf.grad_val, train_bs, vbs,
            )

        return forward

    # -- attach / detach ------------------------------------------------------------------

    def attach(self) -> None:
        for name, layer in self.model.named_modules():
            if isinstance(layer, _SUPPORTED):
                if any(p.requires_grad for p in layer.parameters(recurse=False)):
                    if isinstance(layer, nn.Linear) and layer.bias is not None:
                        raise RuntimeError(
                            f"GHOST_AUTOGRAD_FN: Linear '{name}' has a bias; biases are not "
                            "supported by the custom-Function path."
                        )
                    self._layers.append((name, layer))
                    self._orig_forward[id(layer)] = layer.forward
                    if isinstance(layer, nn.Linear):
                        layer.forward = self._wrap_linear(layer)
                    elif isinstance(layer, nn.Embedding):
                        layer.forward = self._wrap_embedding(layer)
                    elif isinstance(layer, nn.RMSNorm):
                        layer.forward = self._wrap_rmsnorm(layer)
            else:
                # FAIL-LOUD on any other parameterized leaf layer (plan §6 decision 4).
                is_leaf = not list(layer.children())
                if is_leaf and any(p.requires_grad for p in layer.parameters(recurse=False)):
                    raise RuntimeError(
                        f"GHOST_AUTOGRAD_FN: unsupported parameterized layer '{name}' "
                        f"({type(layer).__name__}). Supported: "
                        f"{[t.__name__ for t in _SUPPORTED]}. No silent eager fallback."
                    )

    def detach(self) -> None:
        for _, layer in self._layers:
            orig = self._orig_forward.get(id(layer))
            if orig is not None:
                layer.forward = orig
            if hasattr(layer, "_ghost_buffers"):
                delattr(layer, "_ghost_buffers")
        self._layers.clear()
        self._orig_forward.clear()
        self._buffers.clear()

    # -- post-backward collection ---------------------------------------------------------

    def collect_dot_products(self) -> Optional[torch.Tensor]:
        """Sum each layer's per-train-sample dot-product buffer → [train_bs]."""
        total = None
        for _, layer in self._layers:
            buf = self._buffers.get(id(layer))
            if buf is None:
                continue
            total = buf.dot.clone() if total is None else total + buf.dot
        return total

    def grad_val_for(self, param) -> Optional[torch.Tensor]:
        for _, layer in self._layers:
            if layer.weight is param:
                buf = self._buffers.get(id(layer))
                return None if buf is None else buf.grad_val
        return None
