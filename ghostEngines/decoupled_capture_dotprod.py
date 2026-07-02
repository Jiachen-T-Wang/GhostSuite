"""Compile-compatible ghost dot-products via transparent in-graph identity wrappers
(``GHOST_DECOUPLED_FN=1``).

Goal (re-examination doc §3): realize the model-compile win that the eager engine leaves on the
table, without fusing the dot-product into the model's joint backward graph.

**Design (in-graph dot).** An identity ``autograd.Function`` wraps each supported layer's *output*
and computes the per-sample dot-product + ``grad_val`` *inside its backward*, storing only the
**small** results (``dot`` [train_bs], ``grad_val`` weight-shaped). Because the Function is a
transparent identity on the output, each layer keeps its **native fused backward**; the heavy dot
intermediates (``grad_val_projected``) are transient in the backward, not stored. This keeps the
eager-hook engine untouched and lets ``torch.compile`` regional-compile each block (no hooks /
lock / setattr in the traced region). Train grads are recovered via subtract-val.

Supported leaves: ``nn.Linear`` (weight + optional bias; ``_IGLinearFn``), ``nn.Embedding``,
``nn.RMSNorm``, and ``nn.LayerNorm`` (weight + optional bias; ``_IGLayerNormFn``). A **tied** weight
whose module also has a per-module bias is the one exception (fail-loud at attach). **Tied weights** — a weight shared by
two supported modules (e.g. GPT-2 ``wte`` ↔ ``lm_head``) — take a separate route: those (top-level,
not in compiled regions) capture their ``(A, B)`` and are combined post-backward by
``stash_tied_contribution`` / ``finalize_tied_param`` (the eager tied finalizer), so the shared
parameter's dot includes the cross-terms. Non-tied leaves use the fast in-graph dot.

Gated behind ``GHOST_DECOUPLED_FN=1`` (requires ``GHOST_SUBTRACT_VAL=1``). FAIL-LOUD on any
unsupported parameterized leaf.

**Module-reuse / buffer-clobber detection (uncompiled path only).** The in-graph store is an
overwrite, so a wrapped module invoked more than once in a single backward (ALBERT-style module
sharing) — or a second backward before ``collect_microbatch_dot`` reads the buffers — would
silently drop a contribution. Each in-graph Function backward therefore fails loud on a second
store into the same ``dot`` buffer within one pass. Limitation: inside a ``torch.compile``-d
region the Function backward is traced into the graph, so this Python-level check cannot run at
runtime there; compiled regions rely on the model being reuse-free (true for the supported
transformer blocks).
"""

from typing import Dict, List, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


_DECOUPLED_FN = os.getenv("GHOST_DECOUPLED_FN", "0") == "1"

_SUPPORTED = (nn.Linear, nn.Embedding, nn.RMSNorm, nn.LayerNorm)
ACCUM_DTYPE = torch.float32
_MISSING = object()  # sentinel: layer had no ``name`` attr before attach (restore by deleting)


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


def _mark_pass_store(dot_buf: torch.Tensor) -> None:
    """Fail loud on a second store into the same in-graph ``dot`` buffer within one pass.

    ``_capture_store`` overwrites, so a duplicate store silently clobbers the earlier
    contribution — either a wrapped module ran more than once in a single backward (module
    reuse), or a second backward ran before ``collect_microbatch_dot`` read the previous one's
    buffers. Skipped while tracing: under compile the Function backward is baked into the graph
    and this Python check cannot execute at runtime (see the module docstring)."""
    if torch.compiler.is_compiling():
        return
    if getattr(dot_buf, "_ghost_pass_stored", False):
        raise RuntimeError(
            "GHOST_DECOUPLED_FN: in-graph dot buffer stored twice in one pass. Either a wrapped "
            "module was invoked more than once per forward (module reuse is unsupported on the "
            "in-graph path), or a second backward ran before collect_microbatch_dot read the "
            "previous one's buffers. The store is an overwrite, so a contribution would be "
            "silently lost."
        )
    dot_buf._ghost_pass_stored = True


# =======================================================================================
# tied-weight capture: store full (A, B) for the post-backward cross-term finalizer
# =======================================================================================


class _CaptureFn(torch.autograd.Function):
    """Identity in forward; reduces this use's tied contribution INLINE in backward.

    Used for tied weights (a weight shared by >=2 supported modules, e.g. GPT-2 ``wte``/``lm_head``):
    the per-use in-graph dot would miss the cross-terms of the shared parameter's gradient, so each
    use accumulates its validation aggregate and stashes its (reduced) train factors via
    ``stash_tied_contribution`` *inside its own backward* — exactly like the eager hook engine —
    and ``finalize_tied_param`` combines the cross-terms post-backward. Doing the reduction inline
    (rather than buffering the full ``(A, B)`` for a post-backward combine) avoids holding a
    persistent vocab-sized ``B`` (the ``lm_head`` logits gradient) across the whole backward, which
    matches eager's memory. These layers are top-level / never in a compiled region, so arbitrary
    Python in the backward is safe.
    """

    @staticmethod
    def forward(ctx, output, input_act, layer, val_bs):
        ctx.save_for_backward(input_act)
        ctx.layer = layer
        ctx.val_bs = val_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from .supported_layers_grad_samplers_dotprod import stash_tied_contribution
        (input_act,) = ctx.saved_tensors
        # Record the combined batch size for the subtract-val recovery scale HERE, in the
        # backward: only a pass that actually contributes gradients can set the scale, so a
        # no_grad eval forward — or a grad-enabled forward whose graph is discarded — at another
        # batch size can never clobber it (recording in the forward wrapper could).
        ctx.layer._ghost_capture_total_bs = input_act.size(0)
        # Reduce immediately: accumulate gval + keep only the small train factors, then let the
        # captured (A, B) free with this backward node — no persistent vocab-sized buffer.
        stash_tied_contribution(ctx.layer, input_act, grad_output, ctx.val_bs)
        return grad_output, None, None, None


# =======================================================================================
# in-graph-dot Functions: transparent identity, native backward preserved, small stores
# =======================================================================================


class _IGLinearFn(torch.autograd.Function):
    """Identity on a Linear's output; computes the ghost dot-product in backward (small stores).

    The wrapped ``nn.Linear`` keeps its native (cuBLAS-fused) backward; this Function only reads
    the layer input ``A`` and output grad ``B`` to emit ``dot`` [train_bs] and ``grad_val``
    [d_out, d_in], mirroring ``_compute_linear_dot_product`` / the 1b ghost formula.

    With a bias, the bias gradient is ``B`` summed over tokens (it does NOT depend on ``A``), so the
    bias dot is folded into the same per-train-sample ``dot`` and its validation aggregate stored in
    ``gb_buf``. Because the bias term reads only ``B`` (the incoming backward grad, always live), it
    needs no extra ``save_for_backward`` and is unaffected by the min-cut partitioner's recompute of
    ``A`` (activation checkpointing).
    """

    @staticmethod
    def forward(ctx, output, input_act, dot_buf, gradval_buf, gb_buf, has_bias, train_bs, val_bs):
        ctx.save_for_backward(input_act)
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.gb_buf = gb_buf
        ctx.has_bias = has_bias
        ctx.train_bs = train_bs
        ctx.val_bs = val_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        A = ctx.saved_tensors[0]
        B = grad_output
        train_bs = ctx.train_bs
        d_in = A.shape[-1]
        d_out = B.shape[-1]
        compute_dtype = B.dtype if B.is_floating_point() else A.dtype
        A_flat = A.to(compute_dtype).reshape(-1, d_in)
        B_flat = B.to(compute_dtype).reshape(-1, d_out)
        # tokens-per-sample inferred from the flattened batch: rank-generic, so this handles
        # 2-D [batch, d_in] (seq=1) and >3-D inputs, matching the eager linear path.
        seq = A_flat.size(0) // A.size(0)
        split = train_bs * seq
        A_train, A_val = A_flat[:split], A_flat[split:]
        B_train, B_val = B_flat[:split], B_flat[split:]
        grad_val = torch.matmul(B_val.t(), A_val)                 # [d_out, d_in]
        grad_val_projected = torch.matmul(B_train, grad_val)      # [train*seq, d_in] (transient)
        token_scores = (A_train * grad_val_projected).sum(dim=1)  # [train*seq]
        dot = token_scores.view(train_bs, seq).sum(dim=1)         # [train_bs] (weight contribution)

        # --- bias: grad = B summed over tokens (no dependence on A) ---
        m = _store(ctx.gradval_buf, grad_val.to(ACCUM_DTYPE))
        if ctx.has_bias:
            per_sample_b = B_train.view(train_bs, seq, d_out).sum(dim=1)  # [train_bs, d_out]
            total_b = B_val.sum(dim=0)                                    # [d_out]
            dot = dot + torch.einsum("bf,f->b", per_sample_b, total_b)
            m = m + _store(ctx.gb_buf, total_b.to(ACCUM_DTYPE))
        m = m + _store(ctx.dot_buf, dot)                           # store the full (weight+bias) dot once

        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, dot_buf, gradval_buf, gb_buf, has_bias, train_bs, val_bs)
        return grad_out, None, None, None, None, None, None, None


class _IGEmbeddingFn(torch.autograd.Function):
    """Identity on an Embedding's output; computes the embedding dot-product in backward."""

    @staticmethod
    def forward(ctx, output, idx, weight_shape, padding_idx, dot_buf, gradval_buf,
                train_bs, val_bs):
        ctx.save_for_backward(idx)
        ctx.weight_shape = weight_shape
        ctx.padding_idx = padding_idx
        ctx.dot_buf = dot_buf
        ctx.gradval_buf = gradval_buf
        ctx.train_bs = train_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        idx = ctx.saved_tensors[0]
        B = grad_output
        train_bs = ctx.train_bs
        compute_dtype = B.dtype if B.is_floating_point() else torch.float32
        idx_l = idx.long()
        idx_train, idx_val = idx_l[:train_bs], idx_l[train_bs:]
        B_train, B_val = B[:train_bs].to(compute_dtype), B[train_bs:].to(compute_dtype)
        vocab, d_f = ctx.weight_shape
        # The native embedding backward zeroes the padding_idx row of dL/dW, so pad positions
        # must not contribute to grad_val (mask, not filter — shapes stay static under compile).
        # With grad_val[padding_idx] == 0, train pad positions then contribute
        # B_train . grad_val[padding_idx] == 0 to the dot automatically.
        if ctx.padding_idx is not None:
            B_val = B_val * (idx_val != ctx.padding_idx).unsqueeze(-1).to(B_val.dtype)
        grad_val = torch.zeros((vocab, d_f), dtype=compute_dtype, device=B.device)
        grad_val.index_add_(0, idx_val.reshape(-1), B_val.reshape(-1, d_f))
        # Reduce over every dim except the per-sample batch dim, so a 1-D [batch] index
        # (one id per sample) works as well as [batch, seq]; see _compute_embedding_dot_product.
        prod = (B_train * grad_val[idx_train]).to(ACCUM_DTYPE)
        dot = prod.sum(dim=tuple(range(1, prod.dim())))
        m1 = _store(ctx.dot_buf, dot)
        m2 = _store(ctx.gradval_buf, grad_val.to(ACCUM_DTYPE))
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        return grad_out, None, None, None, None, None, None, None


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
        _mark_pass_store(ctx.dot_buf)
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
        # Rank-2 [batch, d] input has no token dims: sum(dim=[]) reduces ALL dims, so guard
        # (mirrors _IGLayerNormFn and the eager path).
        sum_dims = list(range(1, gw_train.dim() - 1))
        per_sample = gw_train.sum(dim=sum_dims) if sum_dims else gw_train    # [train, d]
        total_val = gw_val.sum(dim=list(range(gw_val.dim() - 1)))            # [d]
        dot = torch.einsum("bf,f->b", per_sample, total_val)              # [train]
        m1 = _store(ctx.dot_buf, dot)
        m2 = _store(ctx.gradval_buf, total_val.to(ACCUM_DTYPE))
        grad_out = grad_output + (0.0 * (m1 + m2)).to(grad_output.dtype)
        # grads for (output, input_act, eps, dot_buf, gradval_buf, train_bs, val_bs)
        return grad_out, None, None, None, None, None, None


class _IGLayerNormFn(torch.autograd.Function):
    """Identity on a LayerNorm's output; computes weight (and bias) dot-products in backward.

    Mirrors the eager ``_compute_layernorm_dot_product`` reference: the per-sample gradient for the
    weight is ``B * normalized_A`` and for the bias is ``B``. We store the combined per-train-sample
    ``dot`` plus the validation-side weight/bias gradients (``grad_val``) for subtract-val recovery.
    The wrapped ``nn.LayerNorm`` keeps its native backward (this Function is a transparent identity
    on the output), so weight/bias ``.grad`` are still produced by autograd.
    """

    @staticmethod
    def forward(ctx, output, input_act, normalized_shape, eps, has_bias,
                dot_buf, gw_buf, gb_buf, train_bs, val_bs):
        ctx.save_for_backward(input_act)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.has_bias = has_bias
        ctx.dot_buf = dot_buf
        ctx.gw_buf = gw_buf
        ctx.gb_buf = gb_buf
        ctx.train_bs = train_bs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        A = ctx.saved_tensors[0].to(ACCUM_DTYPE)
        B = grad_output.to(ACCUM_DTYPE)
        train_bs = ctx.train_bs
        ns = ctx.normalized_shape
        eps = ctx.eps
        A_train, A_val = A[:train_bs], A[train_bs:]
        B_train, B_val = B[:train_bs], B[train_bs:]

        # --- weight: grad = B * normalized_A (recompute normalized input without affine) ---
        norm_A_train = F.layer_norm(A_train, ns, eps=eps)
        norm_A_val = F.layer_norm(A_val, ns, eps=eps)
        gw_train = B_train * norm_A_train
        gw_val = B_val * norm_A_val
        sum_dims_w = list(range(1, gw_train.dim() - 1))
        per_sample_w = gw_train.sum(dim=sum_dims_w) if sum_dims_w else gw_train  # [train, F]
        total_w = gw_val.sum(dim=list(range(gw_val.dim() - 1)))                  # [F]
        dot = torch.einsum("bf,f->b", per_sample_w, total_w)                    # [train]

        # --- bias: grad = B (folded into the same per-sample dot) ---
        if ctx.has_bias:
            sum_dims_b = list(range(1, B_train.dim() - 1))
            per_sample_b = B_train.sum(dim=sum_dims_b) if sum_dims_b else B_train  # [train, F]
            total_b = B_val.sum(dim=list(range(B_val.dim() - 1)))                  # [F]
            dot = dot + torch.einsum("bf,f->b", per_sample_b, total_b)

        m = _store(ctx.dot_buf, dot) + _store(ctx.gw_buf, total_w)
        if ctx.has_bias:
            m = m + _store(ctx.gb_buf, total_b)

        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, normalized_shape, eps, has_bias,
        #            dot_buf, gw_buf, gb_buf, train_bs, val_bs)
        return grad_out, None, None, None, None, None, None, None, None, None


# =======================================================================================
# separate-val Functions (``separate_val=True``): project the train backward against a
# CACHED val gradient. The val gradient is harvested once per optimizer step from a plain
# autograd backward on the val batch alone (``harvest_val_grads``) — it is constant across
# a step's microbatches because the weights do not change between them. These Functions
# therefore never see val rows: the whole batch is train, there is no per-layer grad_val
# GEMM, no grad_val buffer store, and no subtract-val recovery (autograd's ``.grad`` IS the
# train gradient). Tied weights need no special path — autograd already summed all uses'
# val contributions into the shared ``.grad``, and ``<G_i, gval> = sum_uses <g_i^use, gval>``
# is linear, so each use's in-graph projection against the same total ``gval`` just adds.
# =======================================================================================


class _SVLinearFn(torch.autograd.Function):
    """Identity on a Linear's output; dots the per-sample train grad with cached ``gval``.

    ``dot_i = sum_t A_i[t] . (B_i[t] @ gval_w)`` (+ bias term ``sum_t B_i[t] . gval_b``).
    The projection GEMM is the only non-trivial cost; token products are accumulated in
    fp32 before the per-sample reduction.
    """

    @staticmethod
    def forward(ctx, output, input_act, dot_buf, gvalw_buf, gvalb_buf, has_bias):
        ctx.save_for_backward(input_act)
        ctx.dot_buf = dot_buf
        ctx.gvalw_buf = gvalw_buf
        ctx.gvalb_buf = gvalb_buf
        ctx.has_bias = has_bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        A = ctx.saved_tensors[0]
        B = grad_output
        bs = A.size(0)
        d_in = A.shape[-1]
        d_out = B.shape[-1]
        compute_dtype = B.dtype if B.is_floating_point() else A.dtype
        A_flat = A.to(compute_dtype).reshape(-1, d_in)
        B_flat = B.to(compute_dtype).reshape(-1, d_out)
        seq = A_flat.size(0) // bs
        gval = ctx.gvalw_buf.to(compute_dtype)
        proj = torch.matmul(B_flat, gval)                                # [bs*seq, d_in]
        token_scores = (A_flat * proj).to(ACCUM_DTYPE).sum(dim=1)        # [bs*seq]
        dot = token_scores.view(bs, seq).sum(dim=1)                      # [bs]
        if ctx.has_bias:
            per_sample_b = B_flat.view(bs, seq, d_out).to(ACCUM_DTYPE).sum(dim=1)
            dot = dot + torch.einsum("bf,f->b", per_sample_b, ctx.gvalb_buf)
        m = _store(ctx.dot_buf, dot)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, dot_buf, gvalw_buf, gvalb_buf, has_bias)
        return grad_out, None, None, None, None, None


class _SVEmbeddingFn(torch.autograd.Function):
    """Identity on an Embedding's output; dots the train grad with cached ``gval``.

    ``dot_i = sum_t B_i[t] . gval[idx_i[t]]`` — a row gather, no index_add. Pad positions
    need no masking: the harvested val ``.grad`` already has a zero ``padding_idx`` row
    (the native embedding backward zeroes it), so their contribution is exactly 0.
    """

    @staticmethod
    def forward(ctx, output, idx, dot_buf, gval_buf):
        ctx.save_for_backward(idx)
        ctx.dot_buf = dot_buf
        ctx.gval_buf = gval_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        idx = ctx.saved_tensors[0].long()
        prod = (grad_output.to(ACCUM_DTYPE) * ctx.gval_buf[idx].to(ACCUM_DTYPE))
        dot = prod.sum(dim=tuple(range(1, prod.dim())))
        m = _store(ctx.dot_buf, dot)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, idx, dot_buf, gval_buf)
        return grad_out, None, None, None


class _SVRMSNormFn(torch.autograd.Function):
    """Identity on an RMSNorm's output; dots the per-sample train weight-grad with ``gval``."""

    @staticmethod
    def forward(ctx, output, input_act, eps, dot_buf, gval_buf):
        ctx.save_for_backward(input_act)
        ctx.eps = eps
        ctx.dot_buf = dot_buf
        ctx.gval_buf = gval_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        A = ctx.saved_tensors[0].to(ACCUM_DTYPE)
        B = grad_output.to(ACCUM_DTYPE)
        rms = torch.sqrt((A ** 2).mean(dim=-1, keepdim=True) + ctx.eps)
        gw = B * (A / rms)
        sum_dims = list(range(1, gw.dim() - 1))
        per_sample = gw.sum(dim=sum_dims) if sum_dims else gw               # [bs, d]
        dot = torch.einsum("bf,f->b", per_sample, ctx.gval_buf)
        m = _store(ctx.dot_buf, dot)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, eps, dot_buf, gval_buf)
        return grad_out, None, None, None, None


class _SVLayerNormFn(torch.autograd.Function):
    """Identity on a LayerNorm's output; dots train weight (and bias) grads with ``gval``."""

    @staticmethod
    def forward(ctx, output, input_act, normalized_shape, eps, has_bias,
                dot_buf, gw_buf, gb_buf):
        ctx.save_for_backward(input_act)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.has_bias = has_bias
        ctx.dot_buf = dot_buf
        ctx.gw_buf = gw_buf
        ctx.gb_buf = gb_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _mark_pass_store(ctx.dot_buf)
        A = ctx.saved_tensors[0].to(ACCUM_DTYPE)
        B = grad_output.to(ACCUM_DTYPE)
        norm_A = F.layer_norm(A, ctx.normalized_shape, eps=ctx.eps)
        gw = B * norm_A
        sum_dims = list(range(1, gw.dim() - 1))
        per_sample_w = gw.sum(dim=sum_dims) if sum_dims else gw             # [bs, F]
        dot = torch.einsum("bf,f->b", per_sample_w, ctx.gw_buf)
        if ctx.has_bias:
            per_sample_b = B.sum(dim=sum_dims) if sum_dims else B           # [bs, F]
            dot = dot + torch.einsum("bf,f->b", per_sample_b, ctx.gb_buf)
        m = _store(ctx.dot_buf, dot)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, normalized_shape, eps, has_bias, dot_buf, gw_buf, gb_buf)
        return grad_out, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------------------


class GhostDecoupledManager:
    def __init__(self, model: nn.Module, val_batch_size: int,
                 score_exclude_params=None, separate_val: bool = False,
                 gval_dtype: Optional[torch.dtype] = None) -> None:
        self.model = model
        self.val_batch_size = val_batch_size
        # separate-val mode: batches passed through the wrapped model are ALL-TRAIN; the val
        # gradient is harvested once per optimizer step from a plain backward on the val batch
        # (``harvest_val_grads``) and cached in per-param ``gval`` buffers that the _SV*Fn
        # backwards project against. No combined batch, no grad_val GEMMs, no subtract-val
        # recovery. ``gval_dtype`` sets the Linear-weight buffer dtype (pass the autocast
        # compute dtype, e.g. bf16, so the projection GEMM runs on tensor cores without a
        # per-backward cast); norm/embedding/bias buffers stay fp32 (their dots are fp32
        # elementwise math, and fp32 matches the eager reference).
        self.separate_val = bool(separate_val)
        self.gval_dtype = gval_dtype
        self._param_gval: Dict[int, torch.Tensor] = {}  # id(param) -> gval buffer (separate mode)
        self._val_harvested = False
        self._excluded_layer_ids: set = set()  # separate mode: layers excluded from the score
        # MODULE-name substrings whose dot-product is dropped from the logged SCORE only
        # (grad_val is still published, so training is unaffected) — the decoupled analogue of the
        # eager engine's param-name score_exclude_params. A tied weight is excluded when ANY of
        # its sharing modules' names match (so "lm_head" excludes the tied wte/lm_head weight even
        # though it finalizes under the first tied use's name). Used e.g. to drop the dominant
        # tied wte/lm_head term from GREATS selection without changing the update.
        self.score_exclude_params = list(score_exclude_params or [])
        # When disabled, the wrapped forwards fall through to the native op (no dot Function),
        # so the model can take a plain forward/backward while the manager stays attached — e.g.
        # the GREATS update pass, a plain step on the selected subset (no val, no ghost).
        self._enabled = True
        self._orig_forward: Dict[int, object] = {}
        self._layers: List[Tuple[str, nn.Module]] = []
        # Tied weights (shared by >=2 supported modules, e.g. wte/lm_head) are handled separately:
        # the per-use in-graph dot would miss the cross-terms of the shared parameter's gradient, so
        # these layers capture (A, B) and are combined post-backward via stash/finalize_tied_param.
        self._tied_layers: List[Tuple[str, nn.Module]] = []
        self._cells: Dict[int, list] = {}  # id(layer) -> mode-specific ACTIVE buffer cell
        # Per-layer cache of in-graph buffers keyed by total batch size, so a caller running
        # several fixed batch shapes through one attached manager (e.g. GREATS' scoring pass
        # at N+m and update pass at k+m) gets a STABLE buffer object per shape. The compiled
        # block graph for each shape then always binds the same buffers (no realloc inside the
        # trace, no stale-buffer recompile). Single-shape callers (e.g. graddotprod_lm) never
        # populate this beyond one entry and keep the original lazy behavior.
        self._cell_bufs: Dict[int, dict] = {}  # id(layer) -> {total_bs: [dot, gvs]}
        # Track attach-time mutations so detach() restores the model cleanly (no leaked attrs):
        self._flagged_weights: List[torch.Tensor] = []   # weights WE set ``_ghost_tied`` on
        self._name_restore: Dict[int, object] = {}       # id(layer) -> prior ``name`` (or _MISSING)
        self._attached = False
        # Step-state machine (idle -> collecting -> recovered): begin_step opens an accumulation
        # step, collect_microbatch_dot counts its microbatches, recover_train_grads closes it.
        # run_step_dotprod refuses to run inside an open accumulation step (it would wipe the
        # accumulated val grad and re-read only the last microbatch's buffers).
        self._step_state = "idle"
        self._microbatches_collected = 0

    # -- forward wrappers (ingraph mode) --------------------------------------------------

    def _ensure_bufs_ingraph(self, layer, total_bs, device):
        """Return the in-graph buffer set for this ``total_bs``, allocating + caching on first
        use. A stable object per (layer, total_bs) so each compiled shape binds fixed buffers."""
        cache = self._cell_bufs[id(layer)]
        bufs = cache.get(total_bs)
        if bufs is None:
            bufs = self._alloc_ingraph(layer, total_bs, device)
            cache[total_bs] = bufs
        return bufs

    def set_enabled(self, flag: bool) -> None:
        """Toggle dot capture. Disabled => wrapped forwards run the native op only (a plain
        forward/backward), so a caller can take a normal optimizer step on a selected subset
        without detaching. Re-enable before the next scoring pass."""
        self._enabled = bool(flag)

    def clear_tied_pass_state(self) -> None:
        """Drop any per-pass tied-weight accumulation (stash + val aggregate). Tied uses now stash
        inline in `_CaptureFn.backward`, so a backward that is not followed by finalize/recover
        (e.g. the attach-time warmup) would otherwise leak its stash into the next pass and
        double-count. A complete scoring step cleans itself up via finalize_tied_param + recover.
        Also clears the in-graph per-pass store flags for the same reason (a warmup backward sets
        them; the next real backward would otherwise be misread as a duplicate store)."""
        for _, layer in self._tied_layers:
            w = layer.weight
            for attr in ("_ghost_tied_stash", "_ghost_tied_train_bs", "_ghost_tied_log_norms",
                         "_ghost_grad_val", "_ghost_tied_gval", "grad_dot_prod"):
                if hasattr(w, attr):
                    delattr(w, attr)
        self._clear_store_flags()

    def _clear_store_flags(self) -> None:
        """Reset the per-pass duplicate-store flags on EVERY cached in-graph dot buffer (all
        shapes, not just the active cell) so a completed/abandoned pass never poisons the next."""
        for cache in self._cell_bufs.values():
            for bufs in cache.values():
                if hasattr(bufs[0], "_ghost_pass_stored"):
                    del bufs[0]._ghost_pass_stored

    def prepare_shape(self, total_bs: int) -> None:
        """Point every in-graph layer's active cell at the buffers for this combined batch size.

        Multi-shape callers (e.g. GREATS, whose scoring and update passes use different batch
        sizes) must call this BEFORE each forward so allocation never happens inside a compiled
        region. Tied-weight capture layers are eager (not compiled) and resize lazily, so they
        are not handled here. No-op-safe for single-shape callers that never call it."""
        for _, layer in self._layers:
            self._cells[id(layer)][0] = self._ensure_bufs_ingraph(
                layer, total_bs, layer.weight.device
            )

    def _gval_buf(self, param: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Return the (per-param, allocated-once) separate-mode ``gval`` buffer. Keyed by param
        id so tied uses share one buffer — autograd's val ``.grad`` on the shared param already
        sums all uses, which is exactly what each use's projection must dot against."""
        buf = self._param_gval.get(id(param))
        if buf is None:
            buf = torch.zeros(param.shape, dtype=dtype, device=param.device)
            self._param_gval[id(param)] = buf
        return buf

    def _alloc_ingraph(self, layer, total_bs, dev):
        """Allocate the per-layer buffers: one shared ``dot`` [train_bs] plus a ``grad_val`` buffer
        per trainable parameter. Returns ``[dot, [(param, grad_val_buf), ...]]``. ``nn.Linear`` and
        ``nn.LayerNorm`` carry weight + optional bias; the other layer types have a single weight.

        separate mode: the whole batch is train (``train_bs = total_bs``) and the per-param
        buffers are the attach-allocated ``gval`` caches (shared across shapes and tied uses),
        not fresh per-shape grad_val stores."""
        if self.separate_val:
            dot = torch.zeros((total_bs,), dtype=ACCUM_DTYPE, device=dev)
            w_dtype = self.gval_dtype or layer.weight.dtype
            gvs = [(layer.weight, self._gval_buf(
                layer.weight, w_dtype if isinstance(layer, nn.Linear) else ACCUM_DTYPE))]
            bias = getattr(layer, "bias", None)
            if bias is not None:
                gvs.append((bias, self._gval_buf(bias, ACCUM_DTYPE)))
            return [dot, gvs]
        train_bs = total_bs - self.val_batch_size
        dot = torch.zeros((train_bs,), dtype=ACCUM_DTYPE, device=dev)
        gvs = []
        if isinstance(layer, nn.Linear):
            d_out, d_in = layer.weight.shape
            gvs.append((layer.weight, torch.zeros((d_out, d_in), dtype=ACCUM_DTYPE, device=dev)))
            if layer.bias is not None:
                gvs.append((layer.bias, torch.zeros((d_out,), dtype=ACCUM_DTYPE, device=dev)))
        elif isinstance(layer, nn.Embedding):
            vocab, d = layer.weight.shape
            gvs.append((layer.weight, torch.zeros((vocab, d), dtype=ACCUM_DTYPE, device=dev)))
        elif isinstance(layer, nn.LayerNorm):
            d = layer.weight.shape[0]
            gvs.append((layer.weight, torch.zeros((d,), dtype=ACCUM_DTYPE, device=dev)))
            if layer.bias is not None:
                gvs.append((layer.bias, torch.zeros((d,), dtype=ACCUM_DTYPE, device=dev)))
        else:  # RMSNorm
            d = layer.weight.shape[0]
            gvs.append((layer.weight, torch.zeros((d,), dtype=ACCUM_DTYPE, device=dev)))
        return [dot, gvs]

    def _wrap_ingraph_linear(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        bias = layer.bias
        has_bias = bias is not None
        vbs = self.val_batch_size
        separate = self.separate_val

        def forward(x):
            out = F.linear(x, weight, bias)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
            gb_buf = cell[0][1][1][1] if has_bias else None
            if separate:
                return _SVLinearFn.apply(out, x, dot_buf, gv_buf, gb_buf, has_bias)
            return _IGLinearFn.apply(
                out, x, dot_buf, gv_buf, gb_buf, has_bias, x.shape[0] - vbs, vbs
            )

        return forward

    def _wrap_ingraph_embedding(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        padding_idx = layer.padding_idx
        wshape = tuple(weight.shape)
        vbs = self.val_batch_size

        separate = self.separate_val

        def forward(idx):
            out = F.embedding(idx, weight, padding_idx)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, idx.shape[0], weight.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
            if separate:
                return _SVEmbeddingFn.apply(out, idx, dot_buf, gv_buf)
            return _IGEmbeddingFn.apply(
                out, idx, wshape, padding_idx, dot_buf, gv_buf, idx.shape[0] - vbs, vbs
            )

        return forward

    def _wrap_ingraph_rmsnorm(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        normalized_shape = tuple(layer.normalized_shape)
        # Pass layer.eps through UNCHANGED (F.rms_norm accepts None => machine eps of the input
        # dtype), so the wrapped forward's numerics match the native module exactly. The dot
        # recompute needs a concrete value; it runs in ACCUM_DTYPE, so mirror None with that
        # dtype's machine eps there.
        eps = layer.eps
        fn_eps = eps if eps is not None else torch.finfo(ACCUM_DTYPE).eps
        vbs = self.val_batch_size

        separate = self.separate_val

        def forward(x):
            out = F.rms_norm(x, normalized_shape, weight, eps)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
            if separate:
                return _SVRMSNormFn.apply(out, x, fn_eps, dot_buf, gv_buf)
            return _IGRMSNormFn.apply(out, x, fn_eps, dot_buf, gv_buf, x.shape[0] - vbs, vbs)

        return forward

    def _wrap_ingraph_layernorm(self, layer):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        bias = layer.bias
        has_bias = bias is not None
        normalized_shape = tuple(layer.normalized_shape)
        eps = layer.eps if layer.eps is not None else 1e-5
        vbs = self.val_batch_size

        separate = self.separate_val

        def forward(x):
            out = F.layer_norm(x, normalized_shape, weight, bias, eps)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gw_buf = cell[0][1][0][1]
            gb_buf = cell[0][1][1][1] if has_bias else None
            if separate:
                return _SVLayerNormFn.apply(
                    out, x, normalized_shape, eps, has_bias, dot_buf, gw_buf, gb_buf
                )
            return _IGLayerNormFn.apply(
                out, x, normalized_shape, eps, has_bias,
                dot_buf, gw_buf, gb_buf, x.shape[0] - vbs, vbs,
            )

        return forward

    # -- forward wrapper (tied weights: capture full (A, B)) ------------------------------

    def _wrap_capture(self, layer, op):
        vbs = self.val_batch_size

        def forward(x):
            out = op(x)
            if not self._enabled:
                return out
            # Tied-capture layers are eager (top-level, never in a compiled region): the reduction
            # happens inside _CaptureFn.backward (stash_tied_contribution), so there is no persistent
            # (A, B) buffer to size — this naturally serves any batch shape (e.g. GREATS' two
            # per-step shapes). The combined batch size for the subtract-val recovery scale is
            # recorded in _CaptureFn.backward, so enabled forwards WITHOUT a backward (no_grad
            # eval, discarded logging forwards) cannot clobber it.
            return _CaptureFn.apply(out, x, layer, vbs)

        return forward

    # -- attach / detach ------------------------------------------------------------------

    def _flag_tied_weights(self) -> None:
        """Flag weights shared by >=2 supported requires-grad modules (e.g. tied wte/lm_head)."""
        users: Dict[int, int] = {}
        for _, layer in self.model.named_modules():
            if isinstance(layer, _SUPPORTED) and any(
                p.requires_grad for p in layer.parameters(recurse=False)
            ):
                w = getattr(layer, "weight", None)
                if w is not None and w.requires_grad:
                    users[id(w)] = users.get(id(w), 0) + 1
        for _, layer in self.model.named_modules():
            w = getattr(layer, "weight", None)
            if w is not None and users.get(id(w), 0) >= 2 and not getattr(w, "_ghost_tied", False):
                w._ghost_tied = True
                self._flagged_weights.append(w)  # record so detach() clears only what WE set

    def attach(self) -> None:
        if self._attached:
            raise RuntimeError(
                "GHOST_DECOUPLED_FN: attach() called on an already-attached manager; detach() "
                "first. Re-attaching would wrap the wrapped forwards again (detach would then "
                "restore a wrapper) and duplicate every layer entry, double-counting dots and "
                "subtracting the val grad twice."
            )
        self._flag_tied_weights()
        try:
            self._attach_walk()
        except Exception:
            # Mid-walk failure (e.g. an unsupported parameterized leaf): restore the layers
            # already wrapped instead of leaving the model half-monkeypatched.
            self.detach()
            raise
        if self.separate_val:
            # Score exclusion with tied-group semantics: a layer is dropped from the SCORE when
            # its own module name matches, or when its weight is shared and ANY sharing module's
            # name matches (e.g. excluding "lm_head" must also drop the tied wte use — the old
            # tied finalizer excluded the whole shared weight the same way).
            groups: Dict[int, List[str]] = {}
            for n, l in self._layers:
                groups.setdefault(id(l.weight), []).append(n)
            for n, l in self._layers:
                names = groups[id(l.weight)] if len(groups[id(l.weight)]) > 1 else [n]
                if any(self._excluded(g) for g in names):
                    self._excluded_layer_ids.add(id(l))
        self._attached = True

    def _attach_walk(self) -> None:
        for name, layer in self.model.named_modules():
            if isinstance(layer, _SUPPORTED):
                if any(p.requires_grad for p in layer.parameters(recurse=False)):
                    tied = getattr(layer.weight, "_ghost_tied", False)
                    # separate mode: tied weights need NO special route. The cached ``gval`` is
                    # the shared param's autograd val ``.grad`` (all uses + cross-terms summed
                    # by autograd), the dot is linear in the train gradient, so each use's
                    # in-graph projection against the same buffer just adds. A per-module bias
                    # on a tied Linear is fine too (the bias is private to that module).
                    if self.separate_val:
                        tied = False
                    # All per-layer fail-loud checks run BEFORE any mutation of this layer, so a
                    # raise here leaves it pristine and the rollback in attach() only has to
                    # restore fully-registered layers.
                    # Non-tied biased Linear is handled by _IGLinearFn (bias dot folded in-graph).
                    # A tied weight whose module also has a per-module bias stays unsupported: the
                    # tied capture path (_CaptureFn / stash_tied_contribution) handles only the
                    # shared weight, and a bias on a tied Linear is rare (tied lm_head is ~always
                    # bias=False). Fail loud rather than silently drop the bias.
                    if isinstance(layer, nn.Linear) and layer.bias is not None and tied:
                        raise RuntimeError(
                            f"GHOST_DECOUPLED_FN: tied Linear '{name}' has a bias; not supported "
                            "(tied weight + per-module bias)."
                        )
                    if getattr(layer.weight, "_ghost_tied", False) and not isinstance(
                        layer, (nn.Linear, nn.Embedding)
                    ):
                        raise RuntimeError(
                            f"GHOST_DECOUPLED_FN: tied weight on unsupported layer '{name}' "
                            f"({type(layer).__name__}); only Linear/Embedding tying is handled."
                        )
                    # The ghost forwards call F.embedding(idx, weight, padding_idx) only, so any
                    # other non-default embedding option would be silently dropped from the
                    # wrapped forward — and the captured math ignores them. Fail loud.
                    if isinstance(layer, nn.Embedding) and (
                        layer.max_norm is not None or layer.scale_grad_by_freq or layer.sparse
                    ):
                        raise ValueError(
                            f"GHOST_DECOUPLED_FN: Embedding '{name}' uses max_norm/"
                            "scale_grad_by_freq/sparse, which the ghost wrappers do not "
                            "implement (only padding_idx is supported)."
                        )
                    if id(layer) not in self._name_restore:
                        self._name_restore[id(layer)] = getattr(layer, "name", _MISSING)
                    setattr(layer, "name", name)
                    self._orig_forward[id(layer)] = layer.forward
                    if tied:
                        # Tied weight: capture (A, B) eagerly and combine post-backward with
                        # cross-terms (these layers are top-level, not in compiled regions).
                        layer.forward = self._wrap_capture(layer, self._build_op(layer))
                        self._tied_layers.append((name, layer))
                    else:  # ingraph (non-tied)
                        if isinstance(layer, nn.Linear):
                            layer.forward = self._wrap_ingraph_linear(layer)
                        elif isinstance(layer, nn.Embedding):
                            layer.forward = self._wrap_ingraph_embedding(layer)
                        elif isinstance(layer, nn.LayerNorm):
                            layer.forward = self._wrap_ingraph_layernorm(layer)
                        else:
                            layer.forward = self._wrap_ingraph_rmsnorm(layer)
                        self._cell_bufs[id(layer)] = {}  # total_bs -> buffers (per-shape cache)
                        self._layers.append((name, layer))
            else:
                is_leaf = not list(layer.children())
                if is_leaf and any(p.requires_grad for p in layer.parameters(recurse=False)):
                    raise RuntimeError(
                        f"GHOST_DECOUPLED_FN: unsupported parameterized layer '{name}' "
                        f"({type(layer).__name__})."
                    )

    @staticmethod
    def _build_op(layer):
        weight = layer.weight
        if isinstance(layer, nn.Linear):
            return lambda x, w=weight: F.linear(x, w)
        if isinstance(layer, nn.Embedding):
            return lambda x, w=weight, p=layer.padding_idx: F.embedding(x, w, p)
        ns = tuple(layer.normalized_shape)
        ep = layer.eps  # pass through unchanged; F.rms_norm treats None as machine eps
        return lambda x, w=weight, ns=ns, ep=ep: F.rms_norm(x, ns, w, ep)

    def warmup(self, example_input: torch.Tensor) -> None:
        was_training = self.model.training
        logits = self.model(example_input)
        loss = logits.float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)
        # The warmup backward stored into the in-graph buffers without a collect; drop the
        # per-pass store flags so the first real backward isn't misread as a duplicate store.
        self._clear_store_flags()

    def detach(self) -> None:
        for _, layer in (self._layers + self._tied_layers):
            orig = self._orig_forward.get(id(layer))
            if orig is not None:
                layer.forward = orig
            # Restore the ``name`` attribute to its pre-attach state (delete if we added it).
            prior = self._name_restore.get(id(layer), _MISSING)
            if prior is _MISSING:
                if hasattr(layer, "name"):
                    del layer.name
            else:
                layer.name = prior
            if hasattr(layer, "_ghost_capture_total_bs"):
                del layer._ghost_capture_total_bs
        # Clear only the tied flags WE set, so a model reused across cycles doesn't leak them.
        for w in self._flagged_weights:
            if hasattr(w, "_ghost_tied"):
                del w._ghost_tied
        self._layers.clear()
        self._tied_layers.clear()
        self._orig_forward.clear()
        self._cells.clear()
        self._cell_bufs.clear()
        self._flagged_weights.clear()
        self._name_restore.clear()
        self._param_gval.clear()
        self._excluded_layer_ids.clear()
        self._val_harvested = False
        self._attached = False
        self._step_state = "idle"
        self._microbatches_collected = 0

    # -- post-backward --------------------------------------------------------------------

    @staticmethod
    def _accumulate_grad_val(param, gv) -> None:
        """Sum the per-microbatch validation gradient into ``param._ghost_grad_val_accum``.

        Under gradient accumulation the autograd ``.grad`` accumulates every microbatch's combined
        (train+val) gradient, so the subtract-val recovery must subtract the *sum* of the
        per-microbatch val grads. ``gv`` is a reused per-layer buffer, hence the clone on first use.
        At ``N == 1`` this stores exactly one microbatch's val grad (identical to the old path)."""
        acc = getattr(param, "_ghost_grad_val_accum", None)
        gv = gv.detach().to(ACCUM_DTYPE)
        param._ghost_grad_val_accum = gv.clone() if acc is None else acc.add_(gv)

    def begin_step(self) -> None:
        """Reset per-step accumulation state before the microbatch loop.

        Clears any leftover ``_ghost_grad_val_accum`` / tied stash / ``grad_dot_prod`` so a step's
        accumulation starts clean. Required before the first ``collect_microbatch_dot`` of a step.
        Opens the step-state machine (idle -> collecting)."""
        self._step_state = "collecting"
        self._microbatches_collected = 0
        self._clear_store_flags()
        # separate mode: force a fresh val-grad harvest each step (the val gradient changes with
        # the weights, so a stale gval must never be projected against).
        self._val_harvested = False
        for _, layer in self._layers:
            for p in layer.parameters(recurse=False):
                for attr in ("_ghost_grad_val_accum", "_ghost_grad_val", "grad_dot_prod"):
                    if hasattr(p, attr):
                        delattr(p, attr)
        for _, layer in self._tied_layers:
            w = layer.weight
            for attr in ("_ghost_grad_val_accum", "_ghost_grad_val", "grad_dot_prod",
                         "_ghost_tied_stash", "_ghost_tied_train_bs", "_ghost_tied_log_norms",
                         "_ghost_tied_gval"):
                if hasattr(w, attr):
                    delattr(w, attr)

    def harvest_val_grads(self, clear_grads: bool = True) -> None:
        """separate mode: copy each supported param's autograd ``.grad`` — produced by a plain
        backward on the val batch alone, with the wrappers disabled — into its ``gval`` buffer.

        Must run once per optimizer step, BEFORE the train microbatches (their in-graph dots
        read these buffers). The copy is in-place so the compiled graphs keep binding the same
        buffer objects. ``clear_grads`` drops the harvested ``.grad`` so the subsequent train
        backwards accumulate the train gradient from zero."""
        if not self.separate_val:
            raise RuntimeError(
                "GHOST_DECOUPLED_FN: harvest_val_grads() is only part of the separate-val "
                "lifecycle (separate_val=True); the combined-batch path stores grad_val "
                "in-backward instead."
            )
        seen = set()
        for name, layer in self._layers:
            cell = self._cells.get(id(layer))
            bufs = cell[0] if cell and cell[0] is not None else None
            if bufs is None:
                # Layer not yet run at any shape (no dot buffer): gval buffers exist per param
                # from _alloc_ingraph only after a first forward. Allocate now via param map.
                w_dtype = self.gval_dtype or layer.weight.dtype
                params = [(layer.weight, w_dtype if isinstance(layer, nn.Linear) else ACCUM_DTYPE)]
                bias = getattr(layer, "bias", None)
                if bias is not None:
                    params.append((bias, ACCUM_DTYPE))
                gvs = [(p, self._gval_buf(p, dt)) for p, dt in params]
            else:
                gvs = bufs[1]
            for p, buf in gvs:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if p.grad is None:
                    raise RuntimeError(
                        f"GHOST_DECOUPLED_FN: separate-val harvest found no .grad on a param of "
                        f"'{name}'. Run a plain backward on the val batch (wrappers disabled via "
                        "set_enabled(False)) before harvest_val_grads()."
                    )
                buf.copy_(p.grad.detach())
                if clear_grads:
                    p.grad = None
        self._val_harvested = True

    def collect_microbatch_dot(self) -> Optional[torch.Tensor]:
        """Read this microbatch's per-layer buffers: return its ``[train_bs]`` dot and fold its
        validation gradient into the per-step accumulator.

        Must be called once after each microbatch's ``backward()`` — before the next backward
        overwrites the reused per-layer buffers. The returned per-microbatch dots are concatenated
        by the caller into the step's ``[N*train_bs]`` score vector. ``recover_train_grads`` then
        consumes the accumulated val grad once, after the loop.

        separate mode: the dots were projected against the step's harvested ``gval`` buffers;
        there is no val grad to fold and no tied finalizer — just sum the per-layer dots
        (tied-group-aware score exclusion precomputed at attach)."""
        if self.separate_val:
            if not self._val_harvested:
                raise RuntimeError(
                    "GHOST_DECOUPLED_FN: separate-val collect before harvest_val_grads(); the "
                    "in-graph dots would have been projected against stale/zero gval buffers."
                )
            total = None
            for name, layer in self._layers:
                cell = self._cells.get(id(layer))
                if cell is None or cell[0] is None:
                    raise RuntimeError(
                        f"GHOST_DECOUPLED_FN: '{getattr(layer, 'name', '?')}' no buffer."
                    )
                dot = cell[0][0]
                if hasattr(dot, "_ghost_pass_stored"):
                    del dot._ghost_pass_stored
                if id(layer) in self._excluded_layer_ids:
                    continue
                total = dot.detach().clone() if total is None else total + dot.detach()
            self._microbatches_collected += 1
            return total

        from .supported_layers_grad_samplers_dotprod import finalize_tied_param
        total = None
        # Non-tied layers: dot + grad_val already in buffers; accumulate grad_val, sum the dot.
        for name, layer in self._layers:
            cell = self._cells.get(id(layer))
            if cell is None or cell[0] is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{getattr(layer,'name','?')}' no buffer.")
            dot, gvs = cell[0]
            # This microbatch's store is consumed; let the next backward store afresh.
            if hasattr(dot, "_ghost_pass_stored"):
                del dot._ghost_pass_stored
            for param, gv in gvs:
                self._accumulate_grad_val(param, gv)  # publish for recover (training unaffected)
            if self._excluded(name):
                continue  # drop from the score only
            total = dot.detach().clone() if total is None else total + dot.detach()

        # Tied weights: each use already stashed its val aggregate + train factors INLINE (in
        # _CaptureFn.backward). Finalize this microbatch's cross-term dot per unique shared weight,
        # fold its val grad into the accumulator, then clear _ghost_grad_val so the next microbatch's
        # inline stash starts from prev=None (first_use_this_pass keys off the deleted
        # _ghost_tied_stash, which finalize_tied_param removed).
        # Score-exclusion for a tied weight matches ANY of its sharing modules' names: e.g.
        # score_exclude_params=["lm_head"] must exclude the shared wte/lm_head weight even though
        # it finalizes under the first tied use's name (transformer.wte on tied GPT-2).
        tied_names: Dict[int, List[str]] = {}
        for n, l in self._tied_layers:
            tied_names.setdefault(id(l.weight), []).append(n)
        seen = set()
        for name, layer in self._tied_layers:
            w = layer.weight
            if id(w) in seen or not hasattr(w, "_ghost_tied_stash"):
                continue
            seen.add(id(w))
            finalize_tied_param(w)            # sets w.grad_dot_prod, leaves _ghost_grad_val for the fold
            if not any(self._excluded(n) for n in tied_names[id(w)]):
                dp = w.grad_dot_prod
                total = dp.detach().clone() if total is None else total + dp.detach()
            if hasattr(w, "_ghost_grad_val"):
                self._accumulate_grad_val(w, w._ghost_grad_val)
                del w._ghost_grad_val
            if hasattr(w, "grad_dot_prod"):
                del w.grad_dot_prod
        self._microbatches_collected += 1
        return total

    def _reset_accumulators(self) -> None:
        """Clear ONLY the per-step val-grad accumulator (+ transient ``grad_dot_prod``), preserving
        any inline tied stash / ``_ghost_grad_val`` that the just-finished backward produced. Used by
        the single-shot ``run_step_dotprod`` (called AFTER a backward, unlike ``begin_step`` which
        runs before the loop and may safely wipe everything)."""
        for _, layer in self._layers:
            for p in layer.parameters(recurse=False):
                for attr in ("_ghost_grad_val_accum", "grad_dot_prod"):
                    if hasattr(p, attr):
                        delattr(p, attr)
        for _, layer in self._tied_layers:
            w = layer.weight
            for attr in ("_ghost_grad_val_accum", "grad_dot_prod"):
                if hasattr(w, attr):
                    delattr(w, attr)

    def run_step_dotprod(self) -> Optional[torch.Tensor]:
        """Single-microbatch convenience (``gradient_accumulation_steps == 1``): collect once.

        Called AFTER one backward, so it must NOT ``begin_step`` (that would wipe this backward's
        inline tied stash before ``collect_microbatch_dot`` reads it). Resets only the accumulator so
        the lone microbatch's val grad lands fresh in ``_ghost_grad_val_accum``. Refuses to run
        inside an open ``begin_step`` accumulation step (mixing the two lifecycles would wipe the
        accumulated val grad and re-read only the last microbatch's buffers)."""
        if self._step_state == "collecting" and self._microbatches_collected > 0:
            raise RuntimeError(
                "GHOST_DECOUPLED_FN: run_step_dotprod called inside an open begin_step "
                f"accumulation step ({self._microbatches_collected} microbatch(es) already "
                "collected). It would wipe the accumulated val grad and re-read only the last "
                "microbatch's buffers. Finish the step with recover_train_grads (or discard_step), "
                "or drop the begin_step/collect_microbatch_dot lifecycle for single-shot use."
            )
        self._reset_accumulators()
        return self.collect_microbatch_dot()

    def discard_step(self) -> None:
        """Close an open accumulation step WITHOUT recovery (the reselect path scores, then takes
        a plain backward on a subset). Resets the step-state machine so a later single-shot
        ``run_step_dotprod`` is not misread as lifecycle mixing; the caller is responsible for
        clearing the per-param scoring state (``discard_scores`` does)."""
        self._step_state = "idle"
        self._microbatches_collected = 0

    def _excluded(self, name: str) -> bool:
        return any(pat in name for pat in self.score_exclude_params)

    def recover_train_grads(self) -> None:
        """subtract-val recovery, once per optimizer step, from the accumulated val grad.

        ``mean_train_grad = (total/train) * (autograd.grad - sum_m grad_val_m)``, with per-microbatch
        ``total = train + val``. The ``sum_m grad_val_m`` lives in ``_ghost_grad_val_accum`` (one term
        at ``N == 1``)."""
        if self.separate_val:
            # Nothing to recover: the train backwards never saw val rows, so autograd's .grad
            # already IS the train gradient. Close the step-state machine and reset the harvest
            # flag so the next step must harvest a fresh val grad.
            self._val_harvested = False
            self._step_state = "recovered"
            self._microbatches_collected = 0
            return
        for name, layer in self._layers:
            # train_bs from the dot buffer length; the parameters to recover are the layer's
            # trainable params (weight + optional bias for LayerNorm).
            cell0 = self._cells[id(layer)][0]
            total_bs = cell0[0].shape[0] + self.val_batch_size
            params = [p for p, _ in cell0[1]]
            train_bs = total_bs - self.val_batch_size
            scale = float(total_bs) / float(train_bs)
            for param in params:
                grad_val = getattr(param, "_ghost_grad_val_accum", None)
                if grad_val is None:
                    raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' has no _ghost_grad_val_accum.")
                if param.grad is None:
                    raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' param has no autograd .grad.")
                param.grad = (scale * (param.grad.float() - grad_val)).to(param.grad.dtype)
                del param._ghost_grad_val_accum
                if hasattr(param, "grad_dot_prod"):
                    del param.grad_dot_prod

        # Tied weights: recover once per unique shared weight (collect_microbatch_dot accumulated the
        # full validation aggregate across microbatches in _ghost_grad_val_accum).
        seen = set()
        for name, layer in self._tied_layers:
            w = layer.weight
            if id(w) in seen:
                continue
            seen.add(id(w))
            total_bs = getattr(layer, "_ghost_capture_total_bs", None)
            if total_bs is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: tied '{name}' missing captured batch size.")
            train_bs = total_bs - self.val_batch_size
            scale = float(total_bs) / float(train_bs)
            grad_val = getattr(w, "_ghost_grad_val_accum", None)
            if grad_val is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: tied '{name}' has no _ghost_grad_val_accum.")
            if w.grad is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: tied '{name}' weight has no autograd .grad.")
            w.grad = (scale * (w.grad.float() - grad_val)).to(w.grad.dtype)
            del w._ghost_grad_val_accum
            if hasattr(w, "grad_dot_prod"):
                del w.grad_dot_prod
            if hasattr(w, "_ghost_grad_val"):
                del w._ghost_grad_val
            if hasattr(w, "_ghost_tied_gval"):
                del w._ghost_tied_gval

        # Close the step-state machine: the accumulated state is consumed.
        self._step_state = "recovered"
        self._microbatches_collected = 0
