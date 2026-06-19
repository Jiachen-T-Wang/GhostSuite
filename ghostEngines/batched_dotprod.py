"""Lever 1b: decoupled, grouped, optionally-compiled dot-product compute.

The default ghost path computes each layer's gradient dot-product *inside* the autograd
backward via a per-layer tensor hook (`output.register_hook`), one tiny launch per layer.
For llama3-130M that is ~200 launch-overhead-bound ops/step.

This module implements the alternative explored in the compile plan (gaps G1+G2):

  1. The per-layer backward tensor hook becomes **store-only** — it just resolves and stashes
     ``(layer, activation, grad_output)``; no dot-product math runs in the backward.
  2. After ``loss.backward()`` finishes, a single post-backward pass groups the stashed layers
     by ``(type, A.shape, B.shape, use_ghost)`` — llama3 has only ~7 distinct linear shapes,
     each repeated across the 12 transformer blocks — and computes each group's dot-products
     with **one batched op** (stacked bmm / einsum) instead of one op per layer. ~200 ops
     collapse to ~10 grouped calls.

Each group's batched math mirrors the per-layer functions in
``supported_layers_grad_samplers_dotprod`` exactly (same fp32 reductions, same grad_val), so
the resulting ``param.grad_dot_prod`` / ``param._ghost_grad_val`` are numerically equivalent;
the only differences are fp-accumulation-order effects (fp32, ~1e-6 relative on the per-layer
scalar; the engine sums across layers so the logged aggregate matches to the same bar).

Gated behind ``GHOST_BATCHED_DOTPROD=1`` and only meaningful with ``GHOST_SUBTRACT_VAL=1``
(it relies on subtract-val's no-masking recovery; ``_maybe_store_grad_val`` runs here, still
before ``prepare_gradients()``). The default eager path is untouched.
"""

from typing import Dict, List, Optional, Tuple

import os
import time

import torch
from torch import nn

from .supported_layers_grad_samplers_dotprod import _maybe_store_grad_val


_BATCHED_DOTPROD = os.getenv("GHOST_BATCHED_DOTPROD", "0") == "1"
_BATCHED_COMPILE = os.getenv("GHOST_BATCHED_DOTPROD_COMPILE", "0") == "1"

# Investigation-only: per-group timing of the grouped dot-product pass. Default OFF so the
# tps runs are unaffected. When GHOST_BATCHED_DOTPROD_BENCH=1, each group's compute is wrapped
# in CUDA-synced timers and an average per-group (ms) is printed after a warmup, mirroring the
# per-layer eager bench in autograd_grad_sample_dotprod. Used to check per-layer/per-group
# uniformity of the 1b speedup.
_BATCHED_BENCH = os.getenv("GHOST_BATCHED_DOTPROD_BENCH", "0") == "1"
_BATCHED_BENCH_WARMUP = int(os.getenv("GHOST_BATCHED_DOTPROD_BENCH_WARMUP", "5"))
_BATCHED_BENCH_STATS: Dict[str, list] = {}
_BATCHED_BENCH_STEP = [0]


def _bench_group(tag: str, device, fn, *fn_args):
    """Run fn(*fn_args) and, under GHOST_BATCHED_DOTPROD_BENCH, record CUDA-synced elapsed ms."""
    if not _BATCHED_BENCH:
        return fn(*fn_args)
    is_cuda = getattr(device, "type", None) == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn(*fn_args)
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    _BATCHED_BENCH_STATS.setdefault(tag, []).append(elapsed_ms)
    return out


def _bench_report():
    """Print averaged per-group timings (post-warmup) and reset for the next step."""
    if not _BATCHED_BENCH:
        return
    step = _BATCHED_BENCH_STEP[0]
    _BATCHED_BENCH_STEP[0] += 1
    if step < _BATCHED_BENCH_WARMUP:
        _BATCHED_BENCH_STATS.clear()
        return
    if not _BATCHED_BENCH_STATS:
        return
    total = 0.0
    lines = []
    for tag, vals in _BATCHED_BENCH_STATS.items():
        ms = vals[-1]
        total += ms
        lines.append((ms, tag))
    lines.sort(reverse=True)
    print(f"[ghost batched bench] step={step} total_grouped_ms={total:.3f}")
    for ms, tag in lines:
        print(f"[ghost batched bench]   {ms:8.3f} ms  {tag}")
    _BATCHED_BENCH_STATS.clear()

# Types supported by the batched path. Anything else under the flag fails loudly (plan §4
# scope decision #4): no silent fallback that could quietly diverge.
_BATCHED_SUPPORTED = (nn.Linear, nn.Embedding, nn.RMSNorm)

ACCUM_DTYPE = torch.float32


# ---------------------------------------------------------------------------------------
# Grouped core functions. Each takes stacked tensors for G same-shape layers and returns a
# stacked result, mirroring the per-layer math in supported_layers_grad_samplers_dotprod.
# These are the regions wrapped in a single torch.compile (compile-once-per-shape-group),
# never per layer — avoiding the per-call Dynamo overhead that regressed the prior attempt.
# ---------------------------------------------------------------------------------------


def _linear_group_ghost(
    A: torch.Tensor,  # [G, total_bs, seq, d_in]  (compute dtype)
    B: torch.Tensor,  # [G, total_bs, seq, d_out]
    train_bs: int,
    val_bs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ghost (associativity) path: returns (dot_prod [G, train_bs], grad_val [G, d_out, d_in]).

    Matmuls in input (compute) dtype, mirroring the per-layer eager ghost path.
    """
    G = A.shape[0]
    seq = A.shape[2]
    d_in = A.shape[-1]
    d_out = B.shape[-1]

    A_flat = A.reshape(G, -1, d_in)
    B_flat = B.reshape(G, -1, d_out)
    split = train_bs * seq

    A_train = A_flat[:, :split]
    A_val = A_flat[:, split:]
    B_train = B_flat[:, :split]
    B_val = B_flat[:, split:]

    grad_val = torch.bmm(B_val.transpose(1, 2), A_val)  # [G, d_out, d_in]
    grad_val_projected = torch.bmm(B_train, grad_val)  # [G, train_bs*seq, d_in]
    token_scores = (A_train * grad_val_projected).sum(dim=2)  # [G, train_bs*seq]
    dot = token_scores.reshape(G, train_bs, seq).sum(dim=2)  # [G, train_bs]
    return dot, grad_val.to(ACCUM_DTYPE)


def _rmsnorm_group(
    A: torch.Tensor,  # [G, total_bs, seq, d]
    B: torch.Tensor,  # [G, total_bs, seq, d]
    train_bs: int,
    val_bs: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm weight-only path: returns (dot [G, train_bs], grad_val [G, d])."""
    A = A.to(ACCUM_DTYPE)
    B = B.to(ACCUM_DTYPE)

    A_train = A[:, :train_bs]
    A_val = A[:, train_bs:]
    B_train = B[:, :train_bs]
    B_val = B[:, train_bs:]

    rms_train = torch.sqrt((A_train ** 2).mean(dim=-1, keepdim=True) + eps)
    rms_val = torch.sqrt((A_val ** 2).mean(dim=-1, keepdim=True) + eps)
    norm_A_train = A_train / rms_train
    norm_A_val = A_val / rms_val

    grad_weight_train = B_train * norm_A_train  # [G, train_bs, seq, d]
    grad_weight_val = B_val * norm_A_val

    # per-sample: sum over seq -> [G, train_bs, d]
    per_sample = grad_weight_train.sum(dim=2)
    # val aggregate: sum over batch and seq -> [G, d]
    total_val = grad_weight_val.sum(dim=(1, 2))

    dot = torch.einsum("gbf,gf->gb", per_sample, total_val)  # [G, train_bs]
    return dot, total_val


def _embedding_single(
    layer: nn.Embedding,
    A: torch.Tensor,
    B: torch.Tensor,
    train_bs: int,
    val_bs: int,
) -> None:
    """Embedding dot-product (one layer; not grouped — typically a single tok_embeddings).

    Mirrors ``_compute_embedding_dot_product`` exactly.
    """
    A = A.detach()
    B = B.detach()
    compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype

    A_train = A[:train_bs]
    A_val = A[train_bs:]
    B_train = B[:train_bs].to(compute_dtype)
    B_val = B[train_bs:].to(compute_dtype)

    A_train_long = A_train.long()
    A_val_long = A_val.long()

    vocab_size, d_f = layer.weight.shape
    grad_val = torch.zeros((vocab_size, d_f), dtype=compute_dtype, device=B_val.device)
    grad_val.index_add_(0, A_val_long.reshape(-1), B_val.reshape(-1, d_f))

    dot = (B_train * grad_val[A_train_long]).to(ACCUM_DTYPE).sum(dim=[1, 2])
    layer.weight.grad_dot_prod = dot
    _maybe_store_grad_val(layer.weight, grad_val)


# ---------------------------------------------------------------------------------------
# Compiled wrappers (compile once per group function, reused across steps & shape-groups).
# ---------------------------------------------------------------------------------------

_COMPILED: Dict[str, object] = {}


def _get_fn(name: str, fn):
    if not _BATCHED_COMPILE:
        return fn
    cached = _COMPILED.get(name)
    if cached is None:
        cached = torch.compile(fn, dynamic=False)
        _COMPILED[name] = cached
    return cached


# ---------------------------------------------------------------------------------------
# Dispatcher: group the stashed (layer, A, B) entries and run batched compute.
# ---------------------------------------------------------------------------------------


def run_batched_dotprod(pending: List[Tuple[nn.Module, torch.Tensor, torch.Tensor]],
                        val_batch_size: int) -> None:
    """Compute all stashed layers' dot-products in batched, grouped ops.

    ``pending`` is a list of ``(layer, activation, grad_output)`` captured store-only during
    the backward. Sets ``param.grad_dot_prod`` and (under subtract-val) ``param._ghost_grad_val``
    for every layer, matching the per-layer path.
    """
    if not pending:
        return

    # Group linears / rmsnorms by structural signature; handle embeddings individually.
    linear_groups: Dict[tuple, List] = {}
    rmsnorm_groups: Dict[tuple, List] = {}

    for layer, A, B in pending:
        if not isinstance(layer, _BATCHED_SUPPORTED):
            raise RuntimeError(
                f"GHOST_BATCHED_DOTPROD: layer {getattr(layer, 'name', '?')} of type "
                f"{type(layer).__name__} is not supported by the batched path. "
                f"Supported: {[t.__name__ for t in _BATCHED_SUPPORTED]}."
            )

        if isinstance(layer, nn.Embedding):
            total_bs = A.shape[0]
            train_bs = total_bs - val_batch_size
            _bench_group(
                f"Embedding[{getattr(layer, 'name', '?')}] (single)", B.device,
                _embedding_single, layer, A, B, train_bs, val_batch_size,
            )
            continue

        if isinstance(layer, nn.Linear):
            # A: [total_bs, seq, d_in] (reshaped to input_shape already by resolve)
            A3 = A
            if A3.dim() != 3:
                raise RuntimeError(
                    f"GHOST_BATCHED_DOTPROD: expected 3D Linear activation, got {tuple(A.shape)}"
                )
            # Use the ghost (associativity) formula for ALL linear groups: it yields the same
            # dot-product as the materialize path but never materializes the [G, d_out, d_in]
            # per-sample grad_train, so grouping stays memory-light even for the large FF
            # weights (where the eager per-layer path picks materialize). grad_val is identical.
            key = (tuple(A3.shape), tuple(B.shape))
            linear_groups.setdefault(key, []).append((layer, A3, B))
            continue

        if isinstance(layer, nn.RMSNorm):
            eps = getattr(layer, "eps", 1e-5)
            key = (tuple(A.shape), tuple(B.shape), round(float(eps), 12))
            rmsnorm_groups.setdefault(key, []).append((layer, A, B))
            continue

    val_bs = val_batch_size

    # --- Linear groups (ghost formula for all; see dispatcher note) ---
    for key, items in linear_groups.items():
        layers = [it[0] for it in items]
        A_stack = torch.stack([it[1] for it in items], dim=0)  # [G, total_bs, seq, d_in]
        B_stack = torch.stack([it[2] for it in items], dim=0)
        total_bs = A_stack.shape[1]
        train_bs = total_bs - val_bs
        fn = _get_fn("linear_ghost", _linear_group_ghost)
        d_in = A_stack.shape[-1]
        d_out = B_stack.shape[-1]
        tag = f"Linear ghost x{len(layers)} [d_in={d_in},d_out={d_out}] (e.g. {getattr(layers[0], 'name', '?')})"
        dot, grad_val = _bench_group(tag, A_stack.device, fn, A_stack, B_stack, train_bs, val_bs)
        for g, layer in enumerate(layers):
            layer.weight.grad_dot_prod = dot[g]
            _maybe_store_grad_val(layer.weight, grad_val[g])

    # --- RMSNorm groups ---
    for key, items in rmsnorm_groups.items():
        eps = key[2]
        layers = [it[0] for it in items]
        A_stack = torch.stack([it[1] for it in items], dim=0)
        B_stack = torch.stack([it[2] for it in items], dim=0)
        total_bs = A_stack.shape[1]
        train_bs = total_bs - val_bs
        fn = _get_fn("rmsnorm", _rmsnorm_group)
        tag = f"RMSNorm x{len(layers)} [shape={tuple(A_stack.shape[1:])}] (e.g. {getattr(layers[0], 'name', '?')})"
        dot, grad_val = _bench_group(tag, A_stack.device, fn, A_stack, B_stack, train_bs, val_bs, eps)
        for g, layer in enumerate(layers):
            layer.weight.grad_dot_prod = dot[g]
            _maybe_store_grad_val(layer.weight, grad_val[g])

    _bench_report()
