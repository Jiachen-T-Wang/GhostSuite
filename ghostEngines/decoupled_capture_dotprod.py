"""Compile-compatible ghost dot-products via transparent in-graph identity wrappers
(``GHOST_DECOUPLED_FN=1``).

Goal (re-examination doc §3): realize the model-compile win that lever 1b left on the table,
without Phase 2's two regressions.

Two designs were measured on H200 (llama3-130M, seq 4096, train bs2 + val bs2):

  * **capture-full** (deprecated, ``GHOST_DECOUPLED_MODE=capture``): an identity Function
    siphons each layer's *full* ``(A, grad_output)`` into preallocated buffers for a post-
    backward grouped (1b) pass. RESULT: −44% tps / **+106% mem** — storing full A,B for 32
    blocks doubles activation memory and the ~50 GiB/step of copies dominate. Wrong move.

  * **in-graph dot** (default, ``GHOST_DECOUPLED_MODE=ingraph``): the identity Function computes
    the per-sample dot-product + grad_val *inside its backward* (Phase 2's math) but stores only
    the **small** results (``dot`` [train_bs], ``grad_val`` weight-shaped). Crucially it is a
    transparent identity on the layer *output*, so each layer keeps its **native fused backward**
    (Phase 2's −13% came partly from replacing that backward with hand-rolled matmuls). The
    heavy dot intermediates (grad_val_projected) are transient in the backward, not stored.

Both keep the eager-hook engine untouched and let ``torch.compile`` regional-compile each block
(no hooks / lock / setattr in the traced region). Train grads are recovered via subtract-val.

Gated behind ``GHOST_DECOUPLED_FN=1`` (requires ``GHOST_SUBTRACT_VAL=1``). FAIL-LOUD on any
unsupported parameterized leaf.
"""

from typing import Dict, List, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batched_dotprod import run_batched_dotprod


_DECOUPLED_FN = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"
_MODE = os.getenv("GHOST_DECOUPLED_MODE", "ingraph")  # "ingraph" (default) | "capture"

_SUPPORTED = (nn.Linear, nn.Embedding, nn.RMSNorm)
ACCUM_DTYPE = torch.float32


# Opaque buffer write (mutates_args -> kept by functionalization; a bare copy_ becomes an
# "invalid graph output" under the partitioner). Returns a fresh fp32 scalar marker the backward
# ties into the passthrough grad so the store node is not dropped as a dead side effect.
@torch.library.custom_op("ghost::capture_store", mutates_args={"buf"})
def _capture_store(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    buf.copy_(val)
    return torch.zeros((), dtype=torch.float32, device=val.device)


@_capture_store.register_fake
def _capture_store_fake(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=val.device)


def _store(buf, val):
    return torch.ops.ghost.capture_store(buf, val)


# =======================================================================================
# capture-full mode (deprecated; kept for reproducing the negative result)
# =======================================================================================


class _CaptureFn(torch.autograd.Function):
    """Identity in forward; stores full (A, B) for a post-backward grouped pass."""

    @staticmethod
    def forward(ctx, output, input_act, a_buf, b_buf):
        ctx.save_for_backward(input_act)
        ctx.a_buf = a_buf
        ctx.b_buf = b_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_act,) = ctx.saved_tensors
        m1 = _store(ctx.a_buf, input_act)
        m2 = _store(ctx.b_buf, grad_output)
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        return grad_out, None, None, None


# =======================================================================================
# in-graph-dot mode (default): transparent identity, native backward preserved, small stores
# =======================================================================================


class _IGLinearFn(torch.autograd.Function):
    """Identity on a Linear's output; computes the ghost dot-product in backward (small stores).

    The wrapped ``nn.Linear`` keeps its native (cuBLAS-fused) backward; this Function only reads
    the layer input ``A`` and output grad ``B`` to emit ``dot`` [train_bs] and ``grad_val``
    [d_out, d_in], mirroring ``_compute_linear_dot_product`` / the 1b ghost formula.
    """

    @staticmethod
    def forward(ctx, output, input_act, dot_buf, gradval_buf, train_bs, val_bs):
        ctx.save_for_backward(input_act)
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        ctx.val_bs = val_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A = ctx.saved_tensors[0]
        B = grad_output
        train_bs = ctx.train_bs
        d_in = A.shape[-1]
        d_out = B.shape[-1]
        seq = A.shape[1]
        compute_dtype = B.dtype if B.is_floating_point() else A.dtype
        A_flat = A.to(compute_dtype).reshape(-1, d_in)
        B_flat = B.to(compute_dtype).reshape(-1, d_out)
        split = train_bs * seq
        A_train, A_val = A_flat[:split], A_flat[split:]
        B_train, B_val = B_flat[:split], B_flat[split:]
        grad_val = torch.matmul(B_val.t(), A_val)                 # [d_out, d_in]
        grad_val_projected = torch.matmul(B_train, grad_val)      # [train*seq, d_in] (transient)
        token_scores = (A_train * grad_val_projected).sum(dim=1)  # [train*seq]
        dot = token_scores.view(train_bs, seq).sum(dim=1)         # [train_bs]
        m1 = _store(ctx.dot_buf, dot)
        m2 = _store(ctx.gradval_buf, grad_val.to(ACCUM_DTYPE))
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        return grad_out, None, None, None, None, None


class _IGEmbeddingFn(torch.autograd.Function):
    """Identity on an Embedding's output; computes the embedding dot-product in backward."""

    @staticmethod
    def forward(ctx, output, idx, weight_shape, dot_buf, gradval_buf, train_bs, val_bs):
        ctx.save_for_backward(idx)
        ctx.weight_shape = weight_shape
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        idx = ctx.saved_tensors[0]
        B = grad_output
        train_bs = ctx.train_bs
        compute_dtype = B.dtype if B.is_floating_point() else torch.float32
        idx_l = idx.long()
        idx_train, idx_val = idx_l[:train_bs], idx_l[train_bs:]
        B_train, B_val = B[:train_bs].to(compute_dtype), B[train_bs:].to(compute_dtype)
        vocab, d_f = ctx.weight_shape
        grad_val = torch.zeros((vocab, d_f), dtype=compute_dtype, device=B.device)
        grad_val.index_add_(0, idx_val.reshape(-1), B_val.reshape(-1, d_f))
        dot = (B_train * grad_val[idx_train]).to(ACCUM_DTYPE).sum(dim=[1, 2])
        m1 = _store(ctx.dot_buf, dot)
        m2 = _store(ctx.gradval_buf, grad_val.to(ACCUM_DTYPE))
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        return grad_out, None, None, None, None, None, None


class _IGRMSNormFn(torch.autograd.Function):
    """Identity on an RMSNorm's output; computes the weight dot-product in backward."""

    @staticmethod
    def forward(ctx, output, input_act, eps, dot_buf, gradval_buf, train_bs, val_bs):
        ctx.save_for_backward(input_act)
        ctx.eps = eps
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        A = ctx.saved_tensors[0].to(ACCUM_DTYPE)
        B = grad_output.to(ACCUM_DTYPE)
        train_bs = ctx.train_bs
        eps = ctx.eps
        A_train, A_val = A[:train_bs], A[train_bs:]
        B_train, B_val = B[:train_bs], B[train_bs:]
        rms_train = torch.sqrt((A_train ** 2).mean(dim=-1, keepdim=True) + eps)
        rms_val = torch.sqrt((A_val ** 2).mean(dim=-1, keepdim=True) + eps)
        gw_train = B_train * (A_train / rms_train)
        gw_val = B_val * (A_val / rms_val)
        per_sample = gw_train.sum(dim=list(range(1, gw_train.dim() - 1)))  # [train, d]
        total_val = gw_val.sum(dim=list(range(gw_val.dim() - 1)))          # [d]
        dot = torch.einsum("bf,f->b", per_sample, total_val)              # [train]
        m1 = _store(ctx.dot_buf, dot)
        m2 = _store(ctx.gradval_buf, total_val.to(ACCUM_DTYPE))
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        # grads for (output, input_act, eps, dot_buf, gradval_buf, train_bs, val_bs)
        return grad_out, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------------------


class GhostDecoupledManager:
    def __init__(self, model: nn.Module, val_batch_size: int) -> None:
        self.model = model
        self.val_batch_size = val_batch_size
        self.mode = _MODE
        self._orig_forward: Dict[int, object] = {}
        self._layers: List[Tuple[str, nn.Module]] = []
        self._cells: Dict[int, list] = {}  # id(layer) -> mode-specific buffer cell

    # -- forward wrappers (ingraph mode) --------------------------------------------------

    def _alloc_ingraph(self, layer, x, out):
        train_bs = x.shape[0] - self.val_batch_size
        dev = x.device
        if isinstance(layer, nn.Linear):
            d_out, d_in = layer.weight.shape
            gv = torch.zeros((d_out, d_in), dtype=ACCUM_DTYPE, device=dev)
        elif isinstance(layer, nn.Embedding):
            vocab, d = layer.weight.shape
            gv = torch.zeros((vocab, d), dtype=ACCUM_DTYPE, device=dev)
        else:  # RMSNorm
            d = layer.weight.shape[0]
            gv = torch.zeros((d,), dtype=ACCUM_DTYPE, device=dev)
        dot = torch.zeros((train_bs,), dtype=ACCUM_DTYPE, device=dev)
        return [dot, gv]

    def _wrap_ingraph_linear(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        vbs = self.val_batch_size

        def forward(x):
            out = F.linear(x, weight)
            if cell[0] is None:
                cell[0] = self._alloc_ingraph(layer, x, out)
            dot_buf, gv_buf = cell[0]
            return _IGLinearFn.apply(out, x, dot_buf, gv_buf, x.shape[0] - vbs, vbs)

        return forward

    def _wrap_ingraph_embedding(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        padding_idx = layer.padding_idx
        wshape = tuple(weight.shape)
        vbs = self.val_batch_size

        def forward(idx):
            out = F.embedding(idx, weight, padding_idx)
            if cell[0] is None:
                cell[0] = self._alloc_ingraph(layer, idx, out)
            dot_buf, gv_buf = cell[0]
            return _IGEmbeddingFn.apply(out, idx, wshape, dot_buf, gv_buf, idx.shape[0] - vbs, vbs)

        return forward

    def _wrap_ingraph_rmsnorm(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        normalized_shape = tuple(layer.normalized_shape)
        eps = layer.eps if layer.eps is not None else 1e-5
        vbs = self.val_batch_size

        def forward(x):
            out = F.rms_norm(x, normalized_shape, weight, eps)
            if cell[0] is None:
                cell[0] = self._alloc_ingraph(layer, x, out)
            dot_buf, gv_buf = cell[0]
            return _IGRMSNormFn.apply(out, x, eps, dot_buf, gv_buf, x.shape[0] - vbs, vbs)

        return forward

    # -- forward wrappers (capture mode, deprecated) --------------------------------------

    def _wrap_capture(self, layer, op):
        cell = [None, None]
        self._cells[id(layer)] = cell

        def forward(x):
            out = op(x)
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
                            f"GHOST_DECOUPLED_FN: Linear '{name}' has a bias; not supported."
                        )
                    setattr(layer, "name", name)
                    self._layers.append((name, layer))
                    self._orig_forward[id(layer)] = layer.forward
                    if self.mode == "capture":
                        weight = layer.weight
                        if isinstance(layer, nn.Linear):
                            op = lambda x, w=weight: F.linear(x, w)
                        elif isinstance(layer, nn.Embedding):
                            op = lambda x, w=weight, p=layer.padding_idx: F.embedding(x, w, p)
                        else:
                            ns = tuple(layer.normalized_shape)
                            ep = layer.eps if layer.eps is not None else 1e-5
                            op = lambda x, w=weight, ns=ns, ep=ep: F.rms_norm(x, ns, w, ep)
                        layer.forward = self._wrap_capture(layer, op)
                    else:  # ingraph
                        if isinstance(layer, nn.Linear):
                            layer.forward = self._wrap_ingraph_linear(layer)
                        elif isinstance(layer, nn.Embedding):
                            layer.forward = self._wrap_ingraph_embedding(layer)
                        else:
                            layer.forward = self._wrap_ingraph_rmsnorm(layer)
            else:
                is_leaf = not list(layer.children())
                if is_leaf and any(p.requires_grad for p in layer.parameters(recurse=False)):
                    raise RuntimeError(
                        f"GHOST_DECOUPLED_FN: unsupported parameterized layer '{name}' "
                        f"({type(layer).__name__})."
                    )

    def warmup(self, example_input: torch.Tensor) -> None:
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
        if self.mode == "capture":
            return self._run_capture()
        # ingraph: dot + grad_val already in buffers; just aggregate dot and publish grad_val.
        from .supported_layers_grad_samplers_dotprod import _maybe_store_grad_val
        total = None
        for _, layer in self._layers:
            cell = self._cells.get(id(layer))
            if cell is None or cell[0] is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{getattr(layer,'name','?')}' no buffer.")
            dot, gv = cell[0]
            _maybe_store_grad_val(layer.weight, gv)
            total = dot.detach().clone() if total is None else total + dot.detach()
        return total

    def _run_capture(self) -> Optional[torch.Tensor]:
        pending = []
        for _, layer in self._layers:
            cell = self._cells.get(id(layer))
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
        for name, layer in self._layers:
            param = layer.weight
            grad_val = getattr(param, "_ghost_grad_val", None)
            if grad_val is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' has no _ghost_grad_val.")
            if param.grad is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' weight has no autograd .grad.")
            # train_bs from the dot buffer length (ingraph) or captured activation (capture).
            if self.mode == "capture":
                total_bs = self._cells[id(layer)][0].shape[0]
            else:
                total_bs = self._cells[id(layer)][0][0].shape[0] + self.val_batch_size
            train_bs = total_bs - self.val_batch_size
            scale = float(total_bs) / float(train_bs)
            param.grad = (scale * (param.grad.float() - grad_val)).to(param.grad.dtype)
            if hasattr(param, "_ghost_grad_val"):
                del param._ghost_grad_val
            if hasattr(param, "grad_dot_prod"):
                del param.grad_dot_prod
