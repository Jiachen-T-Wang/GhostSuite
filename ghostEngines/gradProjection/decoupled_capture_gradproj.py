"""Compile-compatible per-sample gradient *projection* via in-graph identity Functions.

This is the projection analog of ``decoupled_capture_dotprod.py``. The eager
``GradProjLoraEngine`` captures per-sample projected gradients with forward + full-backward
module hooks (``_ghost_A_raw`` / ``_ghost_grad_proj`` + ``setattr``), which force graph breaks and
make the wrapped blocks incompatible with ``torch.compile`` (measured: compiling the blocks bypasses
the backward hook entirely, so nothing is captured). This module keeps the capture but moves it
*in-graph*:

  * Each supported leaf's ``forward`` is monkeypatched to wrap its **output** in a transparent
    identity ``autograd.Function`` (``_IGProjDenseFn`` for ``nn.Linear``, ``_IGProjEmbeddingFn`` for
    ``nn.Embedding``). The wrapped layer keeps its native fused backward; the Function only reads the
    layer input ``A`` and output grad ``B`` and computes the projected per-sample gradient
    ``P_o @ (Σ_t B_t A_tᵀ) @ P_iᵀ`` → ``[B, k_o, k_i]`` **inside its backward**, storing it into a
    preallocated per-layer buffer via an opaque custom op (``ghost::gradproj_store``,
    ``mutates_args`` so functionalization keeps the write).
  * No hooks / lock / ``setattr`` in the traced region ⇒ each transformer block can be
    regional-compiled (and composed with the Inductor min-cut partitioner for activation
    checkpointing).

**Supported leaves:** ``nn.Linear`` and ``nn.Embedding`` only. Unlike the eager
``GradProjLoraEngine`` (which never replaces ``forward``), this path monkeypatches ``forward`` to
``F.linear``, so transformers ``Conv1D`` (weight ``[in, out]``, forward ``x @ W + b``) is **not**
supported and is rejected at ``attach()`` — Conv1D models must use the eager engine. DVEmb's
``shared/gpt2.py`` is all ``nn.Linear``.

Unlike the dot-product decoupled path this is **much simpler**: the projection engine only *observes*
gradients (no subtract-val / no ``.grad`` rewrite) and treats each matched module as an independent
block (no tied cross-term finalizer — capturing BOTH modules of a tied weight, e.g. GPT-2
``wte``/``lm_head``, would drop the shared weight's cross-terms and is unsupported). DVEmb also runs a
single fixed batch shape, so there is one buffer per layer and no multi-shape cache.

Projection matrices, per-layer dims, concatenation order, metadata and the disk-save format are
**reused verbatim** from an (unattached) ``GradProjLoraEngine`` instance, so a decoupled run and a
hook run with the same config produce byte-for-identical ``P`` and identically-ordered projection
vectors (equivalence-tested against the hook engine).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autograd_gradproj import check_embedding_supported, project_dense, project_embedding
from .gradproj_engine import GradProjLoraEngine

ACCUM_DTYPE = torch.float32


# Opaque buffer write: mutates_args keeps the store under functionalization (a bare copy_ becomes an
# "invalid graph output" under the min-cut partitioner). Returns a fresh fp32 scalar marker the
# backward ties into the passthrough grad so the store node is not pruned as a dead side effect.
@torch.library.custom_op("ghost::gradproj_store", mutates_args={"buf"})
def _gradproj_store(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    buf.copy_(val)
    return torch.zeros((), dtype=torch.float32, device=val.device)


@_gradproj_store.register_fake
def _gradproj_store_fake(buf: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=val.device)


def _store(buf, val):
    return torch.ops.ghost.gradproj_store(buf, val)


class _IGProjDenseFn(torch.autograd.Function):
    """Identity on a dense layer's output; computes the projected per-sample grad in backward.

    Uses the shared :func:`project_dense` kernel (same math as ``GradProjHooks``): with layer
    input ``A`` [B, T, n_i] and output grad ``B`` [B, T, n_o], the per-sample gradient
    ``dL/dW = Σ_t B_t A_tᵀ`` is projected as ``P_o dL/dW P_iᵀ = Σ_t (P_o B_t)(P_i A_t)ᵀ`` →
    ``[B, k_o, k_i]`` (× batch_size to undo a mean-reduced loss's 1/B, in float32 == ACCUM_DTYPE).
    ``P_i``/``P_o`` are read-only constants stashed on ``ctx``; only the input activation is
    ``save_for_backward`` (so the min-cut partitioner may recompute it under AC).
    """

    @staticmethod
    def forward(ctx, output, input_act, P_i, P_o, proj_buf):
        ctx.save_for_backward(input_act)
        ctx.P_i = P_i
        ctx.P_o = P_o
        ctx.proj_buf = proj_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (A,) = ctx.saved_tensors
        gradG = project_dense(A, grad_output, ctx.P_i, ctx.P_o)  # [B, k_o, k_i] fp32
        m = _store(ctx.proj_buf, gradG)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, P_i, P_o, proj_buf)
        return grad_out, None, None, None, None


class _IGProjEmbeddingFn(torch.autograd.Function):
    """Identity on an Embedding's output; computes the projected per-sample grad in backward.

    Uses the shared :func:`project_embedding` kernel (same math as ``GradProjHooks``): project
    the output grads ``B_proj = B @ P_oᵀ`` [B, T, k_o] and gather the input-projection column per
    token id (``P_iᵀ[idx]`` → [B, T, k_i]), then ``Σ_t`` outer product → ``[B, k_o, k_i]``
    (× batch_size). ``padding_idx`` positions are masked out of the accumulation to match the
    native embedding backward (which zeroes ``dL/dW[padding_idx]``).
    """

    @staticmethod
    def forward(ctx, output, idx, P_i, P_o, proj_buf, padding_idx):
        ctx.save_for_backward(idx)
        ctx.P_i = P_i
        ctx.P_o = P_o
        ctx.proj_buf = proj_buf
        ctx.padding_idx = padding_idx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        gradG = project_embedding(idx, grad_output, ctx.P_i, ctx.P_o,
                                  padding_idx=ctx.padding_idx)  # [B, k_o, k_i] fp32
        m = _store(ctx.proj_buf, gradG)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, idx, P_i, P_o, proj_buf, padding_idx)
        return grad_out, None, None, None, None, None


class GradProjDecoupledManager:
    """In-graph, compile-friendly capture of per-sample projected gradients.

    Owns an **unattached** ``GradProjLoraEngine`` for the projection setup (matched layers, ``P``
    matrices, dims, slice ranges, metadata) and the disk-save format, and swaps the capture mechanism
    from module hooks to per-layer identity-Function forward wrappers writing into preallocated
    buffers. ``collect_batch`` copies the buffers back into the engine's ``_ghost_grad_proj`` slots
    and delegates to the engine, so ordering / dtype / save / ``extra`` handling are identical.
    """

    _MISSING = object()

    def __init__(self, module: nn.Module, **engine_kwargs):
        # The engine builds P / dims / slices / metadata in its __init__ (no hooks registered until
        # attach(), which we never call). We drive capture ourselves.
        self.engine = GradProjLoraEngine(module, **engine_kwargs)
        self.module = module
        self.matched_layers = self.engine.matched_layers
        self.projection_matrices = self.engine.projection_matrices
        self.projection_dims = self.engine.projection_dims
        self.total_proj_dim = self.engine.total_proj_dim
        self.proj_dtype = self.engine.proj_dtype

        self._orig_forward: Dict[int, object] = {}
        self._cells: Dict[int, list] = {}          # id(layer) -> [buffer or None]
        # (region, original_forward) pairs that a compile harness swapped to torch.compile; detach()
        # restores them so the model object is left clean (no stale compiled graphs) for reuse.
        self._compiled_regions: List[Tuple[nn.Module, object]] = []
        self._warmup_bs: Optional[int] = None      # batch size the buffers/graph were built for
        self._enabled = True
        self.is_attached = False

    # -- forward wrappers -----------------------------------------------------------------

    def _ensure_buf(self, layer, batch_size, device):
        """Preallocate this layer's ``[B, k_o, k_i]`` fp32 projection buffer (once, outside graph)."""
        # Once warmup has locked the shape, a compiled block's traced graph writes the warmup-time
        # buffer tensor; reallocating for a different batch here would desync that write from what
        # collect_batch reads. Fail loud (DVEmb uses a fixed batch, so this never triggers there).
        if self._warmup_bs is not None and batch_size != self._warmup_bs:
            raise RuntimeError(
                f"GradProjDecoupledManager: batch size {batch_size} != warmup batch "
                f"{self._warmup_bs}. The decoupled/compiled path is bound to the warmup batch shape; "
                "re-attach + re-warm up for a different shape.")
        cell = self._cells[id(layer)]
        if cell[0] is None or cell[0].shape[0] != batch_size:
            name = self._name_of(layer)
            k_i, k_o = self.projection_dims[name]
            cell[0] = torch.zeros((batch_size, k_o, k_i), dtype=ACCUM_DTYPE, device=device)
        return cell[0]

    def _name_of(self, layer):
        return self._layer_names[id(layer)]

    def _wrap_dense(self, layer, name):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        bias = layer.bias
        P_i, P_o = self.projection_matrices[name]

        def forward(x):
            out = F.linear(x, weight, bias)
            if not self._enabled:
                return out
            if cell[0] is None or cell[0].shape[0] != x.shape[0]:
                self._ensure_buf(layer, x.shape[0], _param_device(weight))
            return _IGProjDenseFn.apply(out, x, P_i, P_o, cell[0])

        return forward

    def _wrap_embedding(self, layer, name):
        cell = [None]
        self._cells[id(layer)] = cell
        weight = layer.weight
        # attach() rejects max_norm / scale_grad_by_freq / sparse (the capture math does not
        # model them); still forward every arg so the base op stays exact by construction.
        padding_idx = layer.padding_idx
        max_norm = layer.max_norm
        norm_type = layer.norm_type
        scale_grad = layer.scale_grad_by_freq
        sparse = layer.sparse
        P_i, P_o = self.projection_matrices[name]

        def forward(idx):
            out = F.embedding(idx, weight, padding_idx, max_norm, norm_type, scale_grad, sparse)
            if not self._enabled:
                return out
            if cell[0] is None or cell[0].shape[0] != idx.shape[0]:
                self._ensure_buf(layer, idx.shape[0], _param_device(weight))
            return _IGProjEmbeddingFn.apply(out, idx, P_i, P_o, cell[0], padding_idx)

        return forward

    # -- attach / detach ------------------------------------------------------------------

    def attach(self):
        if self.is_attached:
            return
        # Validate every embedding before wrapping any forward, so an unsupported
        # option fails atomically instead of leaving earlier layers wrapped.
        for name, layer in self.matched_layers.items():
            if isinstance(layer, nn.Embedding):
                check_embedding_supported(layer, name)
        self._layer_names = {id(layer): name for name, layer in self.matched_layers.items()}
        for name, layer in self.matched_layers.items():
            if isinstance(layer, nn.Embedding):
                self._orig_forward[id(layer)] = layer.forward
                layer.forward = self._wrap_embedding(layer, name)
            elif isinstance(layer, nn.Linear):
                self._orig_forward[id(layer)] = layer.forward
                layer.forward = self._wrap_dense(layer, name)
            else:
                # transformers Conv1D (weight [in,out], forward x@W+b) would need a different base
                # op than F.linear (x@Wᵀ). The eager hook engine supports it because it never
                # replaces forward; this in-graph path does, so wrapping it with F.linear would
                # corrupt BOTH the forward and the projection. Fail loud rather than silently wrong;
                # use the eager GradProjLoraEngine for Conv1D models. (DVEmb's shared/gpt2.py is all
                # nn.Linear, so this is not hit there.)
                raise NotImplementedError(
                    f"GradProjDecoupledManager: layer '{name}' ({type(layer).__name__}) is not "
                    "supported by the decoupled path (only nn.Linear and nn.Embedding). "
                    "Conv1D-based models must use the eager GradProjLoraEngine.")
        # Mark the borrowed engine ready to collect (we manage attachment via wrappers, not hooks).
        self.engine.is_attached = True
        self.is_attached = True
        print(f"[INFO] Attached decoupled projection wrappers to {len(self.matched_layers)} layers")

    def detach(self):
        if not self.is_attached:
            return
        for _, layer in self.matched_layers.items():
            orig = self._orig_forward.get(id(layer))
            if orig is not None:
                layer.forward = orig
            for attr in ("_ghost_grad_proj", "_ghost_A_raw"):
                if hasattr(layer, attr):
                    delattr(layer, attr)
        # Restore any block/top-level forwards a compile harness swapped to torch.compile, so the
        # model object is left clean (no stale compiled graphs) if it is reused after detach.
        for region, orig in self._compiled_regions:
            region.forward = orig
        self._compiled_regions.clear()
        self._orig_forward.clear()
        self._cells.clear()
        self._warmup_bs = None
        self.engine.is_attached = False
        self.is_attached = False
        print(f"[INFO] Detached decoupled projection wrappers from {len(self.matched_layers)} layers")

    def set_enabled(self, flag: bool):
        """Disable to run a plain forward/backward (no capture) while staying attached."""
        self._enabled = bool(flag)

    def warmup(self, warmup_fn):
        """Run one eager fwd/bwd so every per-layer buffer is allocated OUTSIDE any traced region.

        ``warmup_fn`` is a zero-arg callable doing one forward+backward at the real (fixed) batch
        shape. After this, ``torch.compile`` can trace the blocks with the buffers bound as closure
        cells (read, not allocated, during tracing)."""
        was_training = self.module.training
        warmup_fn()
        self.module.zero_grad(set_to_none=True)
        self.module.train(was_training)
        # Lock the batch shape: buffers are now allocated, and once compiled the traced graph binds
        # them. _ensure_buf rejects a later different batch (would desync graph write vs collect read).
        bufs = [c[0] for c in self._cells.values() if c[0] is not None]
        self._warmup_bs = bufs[0].shape[0] if bufs else None

    # -- collect --------------------------------------------------------------------------

    def collect_batch(self, batch_indices: Optional[List[int]] = None,
                      extra: Optional[dict] = None, save: bool = True) -> torch.Tensor:
        """Copy each layer's in-graph buffer into the engine's slot and delegate to the engine.

        Reusing ``engine.collect_batch`` keeps concatenation order, storage dtype, save format,
        ``extra`` validation and iteration bookkeeping identical to the hook engine."""
        if not self.is_attached:
            raise RuntimeError("Manager is not attached. Call attach() first.")
        for name, layer in self.matched_layers.items():
            buf = self._cells[id(layer)][0]
            if buf is None:
                raise RuntimeError(f"No captured projection for layer {name}; run a backward first.")
            # engine.collect_batch expects [B, k_o, k_i] in _ghost_grad_proj; clone so a later
            # backward overwriting the reused buffer can't corrupt an in-flight collect.
            layer._ghost_grad_proj = buf.clone()
        return self.engine.collect_batch(batch_indices=batch_indices, extra=extra, save=save)

    def clear_gradients(self):
        for _, layer in self.matched_layers.items():
            for attr in ("_ghost_grad_proj", "_ghost_A_raw"):
                if hasattr(layer, attr):
                    delattr(layer, attr)


def _param_device(p):
    return p.device


# ---------------------------------------------------------------------------------------
# attach -> warmup -> regional compile (+ optional min-cut AC) harness
# ---------------------------------------------------------------------------------------


def attach_and_compile_gradproj(module, warmup_fn, *, engine_kwargs,
                                compile_regions=None, extra_regions=None, compile_kwargs=None,
                                activation_memory_budget=None):
    """Attach the decoupled projection manager, warm up its buffers, then regional-compile.

    Mirrors ``decoupled_compile.attach_and_compile_decoupled`` for the projection path.

    Args:
        module: the uncompiled model whose supported leaves get wrapped.
        warmup_fn: zero-arg callable running one eager fwd/bwd at the real (fixed) batch shape.
        engine_kwargs: kwargs forwarded to ``GradProjLoraEngine`` (proj_layers, ranks, seed, dtype,
            proj_dir, ...).
        compile_regions: iterable of submodules to ``torch.compile`` in place (e.g. transformer
            blocks). ``None`` => attach + warmup only (decoupled-eager; correct, for equivalence).
            Regions are restored on ``mgr.detach()``.
        compile_kwargs: forwarded to ``torch.compile`` (default backend='inductor', fullgraph=True).
        activation_memory_budget: if in (0, 1], set the Inductor min-cut partitioner budget
            (compile-native activation checkpointing) before compiling; lower => recompute more.
            NOTE: this sets a *process-global* (``torch._functorch.config.activation_memory_budget``)
            that persists after this call (matches ``decoupled_compile``); it is not restored, so a
            later compile in the same process inherits it unless reset.

    Returns:
        The attached ``GradProjDecoupledManager``.
    """
    mgr = GradProjDecoupledManager(module, **engine_kwargs)
    mgr.attach()
    mgr.warmup(warmup_fn)

    if compile_regions is not None:
        if activation_memory_budget is not None:
            import torch._functorch.config as _fcfg
            _fcfg.activation_memory_budget = float(activation_memory_budget)
            print(f"[INFO] GradProj decoupled: activation_memory_budget={activation_memory_budget} "
                  "(min-cut partitioner recompute).")
        kwargs = {"backend": "inductor", "fullgraph": True}
        if compile_kwargs:
            kwargs.update(compile_kwargs)
        n = 0
        for region in compile_regions:
            mgr._compiled_regions.append((region, region.forward))
            region.forward = torch.compile(region.forward, **kwargs)
            n += 1
        ne = 0
        for region in (extra_regions or []):
            mgr._compiled_regions.append((region, region.forward))
            region.forward = torch.compile(region.forward, **kwargs)
            ne += 1
        msg = f"[INFO] GradProj decoupled: regional-compiled {n} block(s)"
        if ne:
            msg += f" + {ne} top-level region(s)"
        print(msg + f" (backend={kwargs['backend']}, fullgraph={kwargs.get('fullgraph')}).")
    else:
        print("[INFO] GradProj decoupled: attached + warmed up (no compile).")

    return mgr
