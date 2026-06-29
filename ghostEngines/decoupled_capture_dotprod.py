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

Supported leaves: ``nn.Linear`` (no bias), ``nn.Embedding``, ``nn.RMSNorm``, and
``nn.LayerNorm`` (weight + optional bias; ``_IGLayerNormFn``). **Tied weights** — a weight shared by
two supported modules (e.g. GPT-2 ``wte`` ↔ ``lm_head``) — take a separate route: those (top-level,
not in compiled regions) capture their ``(A, B)`` and are combined post-backward by
``stash_tied_contribution`` / ``finalize_tied_param`` (the eager tied finalizer), so the shared
parameter's dot includes the cross-terms. Non-tied leaves use the fast in-graph dot.

Gated behind ``GHOST_DECOUPLED_FN=1`` (requires ``GHOST_SUBTRACT_VAL=1``). FAIL-LOUD on any
unsupported parameterized leaf.
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
        # Reduce over every dim except the per-sample batch dim, so a 1-D [batch] index
        # (one id per sample) works as well as [batch, seq]; see _compute_embedding_dot_product.
        prod = (B_train * grad_val[idx_train]).to(ACCUM_DTYPE)
        dot = prod.sum(dim=tuple(range(1, prod.dim())))
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


# ---------------------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------------------


class GhostDecoupledManager:
    def __init__(self, model: nn.Module, val_batch_size: int,
                 score_exclude_params=None) -> None:
        self.model = model
        self.val_batch_size = val_batch_size
        # Layer-name substrings whose dot-product is dropped from the logged SCORE only
        # (grad_val is still published, so training is unaffected) — mirrors the eager engine's
        # score_exclude_params. Used e.g. to drop the dominant tied wte/lm_head term from GREATS
        # selection without changing the update.
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
        double-count. A complete scoring step cleans itself up via finalize_tied_param + recover."""
        for _, layer in self._tied_layers:
            w = layer.weight
            for attr in ("_ghost_tied_stash", "_ghost_tied_train_bs", "_ghost_tied_log_norms",
                         "_ghost_grad_val", "_ghost_tied_gval", "grad_dot_prod"):
                if hasattr(w, attr):
                    delattr(w, attr)

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

    def _alloc_ingraph(self, layer, total_bs, dev):
        """Allocate the per-layer buffers: one shared ``dot`` [train_bs] plus a ``grad_val`` buffer
        per trainable parameter. Returns ``[dot, [(param, grad_val_buf), ...]]``. All layer types
        except ``nn.LayerNorm`` (weight + optional bias) have a single weight parameter."""
        train_bs = total_bs - self.val_batch_size
        dot = torch.zeros((train_bs,), dtype=ACCUM_DTYPE, device=dev)
        gvs = []
        if isinstance(layer, nn.Linear):
            d_out, d_in = layer.weight.shape
            gvs.append((layer.weight, torch.zeros((d_out, d_in), dtype=ACCUM_DTYPE, device=dev)))
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
        vbs = self.val_batch_size

        def forward(x):
            out = F.linear(x, weight)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
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
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, idx.shape[0], weight.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
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
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gv_buf = cell[0][1][0][1]
            return _IGRMSNormFn.apply(out, x, eps, dot_buf, gv_buf, x.shape[0] - vbs, vbs)

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

        def forward(x):
            out = F.layer_norm(x, normalized_shape, weight, bias, eps)
            if not self._enabled:
                return out
            if cell[0] is None:
                cell[0] = self._ensure_bufs_ingraph(layer, x.shape[0], x.device)
            dot_buf = cell[0][0]
            gw_buf = cell[0][1][0][1]
            gb_buf = cell[0][1][1][1] if has_bias else None
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
            # per-step shapes). Record the combined batch for the subtract-val recovery scale.
            layer._ghost_capture_total_bs = x.shape[0]
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
        self._flag_tied_weights()
        for name, layer in self.model.named_modules():
            if isinstance(layer, _SUPPORTED):
                if any(p.requires_grad for p in layer.parameters(recurse=False)):
                    tied = getattr(layer.weight, "_ghost_tied", False)
                    if isinstance(layer, nn.Linear) and layer.bias is not None:
                        raise RuntimeError(
                            f"GHOST_DECOUPLED_FN: Linear '{name}' has a bias; not supported."
                        )
                    if id(layer) not in self._name_restore:
                        self._name_restore[id(layer)] = getattr(layer, "name", _MISSING)
                    setattr(layer, "name", name)
                    self._orig_forward[id(layer)] = layer.forward
                    if tied:
                        # Tied weight: capture (A, B) eagerly and combine post-backward with
                        # cross-terms (these layers are top-level, not in compiled regions).
                        if not isinstance(layer, (nn.Linear, nn.Embedding)):
                            raise RuntimeError(
                                f"GHOST_DECOUPLED_FN: tied weight on unsupported layer '{name}' "
                                f"({type(layer).__name__}); only Linear/Embedding tying is handled."
                            )
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
        ep = layer.eps if layer.eps is not None else 1e-5
        return lambda x, w=weight, ns=ns, ep=ep: F.rms_norm(x, ns, w, ep)

    def warmup(self, example_input: torch.Tensor) -> None:
        was_training = self.model.training
        logits = self.model(example_input)
        loss = logits.float().pow(2).mean()
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)

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

    # -- post-backward --------------------------------------------------------------------

    def run_step_dotprod(self) -> Optional[torch.Tensor]:
        # Non-tied layers: dot + grad_val already in buffers; aggregate dot and publish grad_val.
        from .supported_layers_grad_samplers_dotprod import (
            _maybe_store_grad_val, finalize_tied_param,
        )
        total = None
        for name, layer in self._layers:
            cell = self._cells.get(id(layer))
            if cell is None or cell[0] is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: '{getattr(layer,'name','?')}' no buffer.")
            dot, gvs = cell[0]
            for param, gv in gvs:
                _maybe_store_grad_val(param, gv)  # always publish grad_val (training unaffected)
            if self._excluded(name):
                continue  # drop from the score only
            total = dot.detach().clone() if total is None else total + dot.detach()

        # Tied weights: each use already stashed its val aggregate + train factors inline (in
        # _CaptureFn.backward); just combine the cross-terms per unique shared weight.
        seen = set()
        for name, layer in self._tied_layers:
            w = layer.weight
            if id(w) not in seen and hasattr(w, "_ghost_tied_stash"):
                finalize_tied_param(w)  # publishes _ghost_grad_val (for recover) + grad_dot_prod
                if self._excluded(name):
                    if hasattr(w, "grad_dot_prod"):
                        del w.grad_dot_prod  # excluded from the score; drop the leftover attr
                else:
                    dp = w.grad_dot_prod
                    total = dp.detach().clone() if total is None else total + dp.detach()
                seen.add(id(w))
        return total

    def _excluded(self, name: str) -> bool:
        return any(pat in name for pat in self.score_exclude_params)

    def recover_train_grads(self) -> None:
        for name, layer in self._layers:
            # train_bs from the dot buffer length; the parameters to recover are the layer's
            # trainable params (weight + optional bias for LayerNorm).
            cell0 = self._cells[id(layer)][0]
            total_bs = cell0[0].shape[0] + self.val_batch_size
            params = [p for p, _ in cell0[1]]
            train_bs = total_bs - self.val_batch_size
            scale = float(total_bs) / float(train_bs)
            for param in params:
                grad_val = getattr(param, "_ghost_grad_val", None)
                if grad_val is None:
                    raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' has no _ghost_grad_val.")
                if param.grad is None:
                    raise RuntimeError(f"GHOST_DECOUPLED_FN: '{name}' param has no autograd .grad.")
                param.grad = (scale * (param.grad.float() - grad_val)).to(param.grad.dtype)
                if hasattr(param, "_ghost_grad_val"):
                    del param._ghost_grad_val
                if hasattr(param, "grad_dot_prod"):
                    del param.grad_dot_prod

        # Tied weights: recover once per unique shared weight (stash_tied_contribution already
        # accumulated the full validation aggregate in _ghost_grad_val).
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
            grad_val = getattr(w, "_ghost_grad_val", None)
            if grad_val is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: tied '{name}' has no _ghost_grad_val.")
            if w.grad is None:
                raise RuntimeError(f"GHOST_DECOUPLED_FN: tied '{name}' weight has no autograd .grad.")
            w.grad = (scale * (w.grad.float() - grad_val)).to(w.grad.dtype)
            if hasattr(w, "_ghost_grad_val"):
                del w._ghost_grad_val
            if hasattr(w, "grad_dot_prod"):
                del w.grad_dot_prod
            if hasattr(w, "_ghost_tied_gval"):
                del w._ghost_tied_gval
