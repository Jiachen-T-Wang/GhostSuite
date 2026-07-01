"""Compile-compatible per-sample gradient *projection* via in-graph identity Functions.

This is the projection analog of ``decoupled_capture_dotprod.py``. The eager
``GradProjLoraEngine`` captures per-sample projected gradients with forward + full-backward
module hooks (``_ghost_A_raw`` / ``_ghost_grad_proj`` + ``setattr``), which force graph breaks and
make the wrapped blocks incompatible with ``torch.compile`` (measured: compiling the blocks bypasses
the backward hook entirely, so nothing is captured). This module keeps the capture but moves it
*in-graph*:

  * Each supported dense leaf's ``forward`` is monkeypatched to wrap its **output** in a transparent
    identity ``autograd.Function`` (``_IGProjDenseFn``). The wrapped ``nn.Linear`` keeps its native
    fused backward; the Function only reads the layer input ``A`` and output grad ``B`` and computes
    the projected per-sample gradient ``P_o @ (Σ_t B_t A_tᵀ) @ P_iᵀ`` → ``[B, k_o, k_i]`` **inside
    its backward**, storing it into a preallocated per-layer buffer via an opaque custom op
    (``ghost::gradproj_store``, ``mutates_args`` so functionalization keeps the write).
  * No hooks / lock / ``setattr`` in the traced region ⇒ each transformer block can be
    regional-compiled (and composed with the Inductor min-cut partitioner for activation
    checkpointing).

Unlike the dot-product decoupled path this is **much simpler**: the projection engine only *observes*
gradients (no subtract-val / no ``.grad`` rewrite) and treats each matched module as an independent
block (no tied cross-term finalizer — the shared-weight cross-terms are out of scope here, see
``docs/issues/open/dve-projection-ignores-weight-tying-cross-terms_2026-07-01.md``). DVE also runs a
single fixed batch shape, so there is one buffer per layer and no multi-shape cache.

Projection matrices, per-layer dims, concatenation order, metadata and the disk-save format are
**reused verbatim** from an (unattached) ``GradProjLoraEngine`` instance, so a decoupled run and a
hook run with the same config produce byte-for-identical ``P`` and identically-ordered projection
vectors (equivalence-tested in ``tests/test_gradproj_decoupled_equiv.py``).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Mirrors ``GradProjHooks._compute_dense_proj``: with layer input ``A`` [B, T, n_i] and output
    grad ``B`` [B, T, n_o], the per-sample gradient is ``dL/dW = Σ_t B_t A_tᵀ``, projected as
    ``P_o dL/dW P_iᵀ = Σ_t (P_o B_t)(P_i A_t)ᵀ`` → ``[B, k_o, k_i]``. The ``* batch_size`` factor
    undoes the ``1/B`` that a mean-reduced loss puts on ``B`` (matches the eager engine and the naive
    reduction='sum' reference). ``P_i``/``P_o`` are read-only constants stashed on ``ctx``; only the
    input activation is ``save_for_backward`` (so the min-cut partitioner may recompute it under AC).
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
        P_i, P_o = ctx.P_i, ctx.P_o
        B = grad_output
        batch_size = A.shape[0]
        # Rank-generic token flatten: [B, ..., n] -> [B, T, n]; handles 2-D (T=1) and >3-D alike.
        A2 = A.reshape(batch_size, -1, A.shape[-1]).to(P_i.dtype)   # [B, T, n_i]
        B2 = B.reshape(batch_size, -1, B.shape[-1]).to(P_o.dtype)   # [B, T, n_o]
        A_proj = torch.matmul(A2, P_i.t())                          # [B, T, k_i]
        B_proj = torch.matmul(B2, P_o.t())                          # [B, T, k_o]
        gradG = torch.einsum('bti,btj->bij', B_proj, A_proj)        # [B, k_o, k_i]
        gradG = (gradG * batch_size).to(ACCUM_DTYPE)
        m = _store(ctx.proj_buf, gradG)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        # grads for (output, input_act, P_i, P_o, proj_buf)
        return grad_out, None, None, None, None


class _IGProjEmbeddingFn(torch.autograd.Function):
    """Identity on an Embedding's output; computes the projected per-sample grad in backward.

    Mirrors ``GradProjHooks._compute_embedding_proj``: project the output grads
    ``B_proj = B @ P_oᵀ`` [B, T, k_o] and gather the input-projection column per token id
    (``P_iᵀ[idx]`` → [B, T, k_i]), then ``Σ_t`` outer product → ``[B, k_o, k_i]`` (× batch_size).
    """

    @staticmethod
    def forward(ctx, output, idx, P_i, P_o, proj_buf):
        ctx.save_for_backward(idx)
        ctx.P_i = P_i
        ctx.P_o = P_o
        ctx.proj_buf = proj_buf
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (idx,) = ctx.saved_tensors
        P_i, P_o = ctx.P_i, ctx.P_o
        B = grad_output
        batch_size = idx.shape[0]
        idx_flat = idx.reshape(batch_size, -1)                                # [B, T]
        grad_flat = B.reshape(batch_size, -1, B.shape[-1]).to(P_o.dtype)      # [B, T, D]
        B_proj = torch.matmul(grad_flat, P_o.t())                            # [B, T, k_o]
        A_proj = P_i.t()[idx_flat]                                           # [B, T, k_i]
        gradG = torch.einsum('bto,bti->boi', B_proj, A_proj)                 # [B, k_o, k_i]
        gradG = (gradG * batch_size).to(ACCUM_DTYPE)
        m = _store(ctx.proj_buf, gradG)
        grad_out = grad_output + (0.0 * m).to(grad_output.dtype)
        return grad_out, None, None, None, None


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
        self._enabled = True
        self.is_attached = False

    # -- forward wrappers -----------------------------------------------------------------

    def _ensure_buf(self, layer, batch_size, device):
        """Preallocate this layer's ``[B, k_o, k_i]`` fp32 projection buffer (once, outside graph)."""
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
        padding_idx = layer.padding_idx
        P_i, P_o = self.projection_matrices[name]

        def forward(idx):
            out = F.embedding(idx, weight, padding_idx)
            if not self._enabled:
                return out
            if cell[0] is None or cell[0].shape[0] != idx.shape[0]:
                self._ensure_buf(layer, idx.shape[0], _param_device(weight))
            return _IGProjEmbeddingFn.apply(out, idx, P_i, P_o, cell[0])

        return forward

    # -- attach / detach ------------------------------------------------------------------

    def attach(self):
        if self.is_attached:
            return
        self._layer_names = {id(layer): name for name, layer in self.matched_layers.items()}
        for name, layer in self.matched_layers.items():
            self._orig_forward[id(layer)] = layer.forward
            if isinstance(layer, nn.Embedding):
                layer.forward = self._wrap_embedding(layer, name)
            else:  # nn.Linear / transformers Conv1D (dense: same projection math)
                layer.forward = self._wrap_dense(layer, name)
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
        self._orig_forward.clear()
        self._cells.clear()
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
        compile_kwargs: forwarded to ``torch.compile`` (default backend='inductor', fullgraph=True).
        activation_memory_budget: if in (0, 1], set the Inductor min-cut partitioner budget
            (compile-native activation checkpointing) before compiling; lower => recompute more.

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
            region.forward = torch.compile(region.forward, **kwargs)
            n += 1
        ne = 0
        for region in (extra_regions or []):
            region.forward = torch.compile(region.forward, **kwargs)
            ne += 1
        msg = f"[INFO] GradProj decoupled: regional-compiled {n} block(s)"
        if ne:
            msg += f" + {ne} top-level region(s)"
        print(msg + f" (backend={kwargs['backend']}, fullgraph={kwargs.get('fullgraph')}).")
    else:
        print("[INFO] GradProj decoupled: attached + warmed up (no compile).")

    return mgr
