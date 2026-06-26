"""General attach -> warmup -> compile harness for the decoupled in-graph ghost dot-product path.

This is the model-agnostic Step-3 deliverable of docs/plans/optimize_graddotprod_lm_2026-06-25.md.
It packages the TorchTitan deferred-compile recipe (attach the ghost manager so the supported
leaves are monkeypatched FIRST, run one eager warmup so the per-layer dot/grad_val buffers are
allocated OUTSIDE the traced region, THEN regional-compile) behind a single call so each example
needs only a few lines of wiring instead of a bespoke trainer.

The caller supplies:
  * ``warmup_fn`` — a zero-arg callable running ONE eager forward+backward at the real training
    shape (combined train+val batch). This is necessarily model-specific (forward signature, loss),
    so it stays with the caller; everything else here is generic.
  * ``compile_regions`` — the submodules to ``torch.compile`` (e.g. the repeated transformer
    blocks, the analogue of TorchTitan's ``model.layers``). ``None`` disables compilation
    (attach + warmup only — correct but eager, useful for correctness checks).

The decoupled path keeps each layer's native fused backward and computes the dot-product as a small
transient inside the backward, so regional compile folds it into the block graph with no graph
breaks. Train grads are recovered via subtract-val (``GHOST_SUBTRACT_VAL``).
"""

import torch

from .decoupled_capture_dotprod import GhostDecoupledManager


def _compile_forward_in_place(module, compile_kwargs):
    """Compile a module's bound ``forward`` in place, preserving the module object.

    Mirrors TorchTitan's ``apply_compile_top_level`` idiom: replacing the module with an
    ``OptimizedModule`` would break ``if self.<submodule>`` truthiness checks and parameter
    identity; compiling the (already ghost-monkeypatched) bound forward keeps the original object
    while folding its in-graph dot into a compiled region.
    """
    module.forward = torch.compile(module.forward, **compile_kwargs)


def attach_and_compile_decoupled(
    model,
    val_batch_size,
    warmup_fn,
    *,
    compile_regions=None,
    extra_regions=None,
    compile_kwargs=None,
    activation_memory_budget=None,
    score_exclude_params=None,
    warmup_shapes=None,
):
    """Attach the decoupled manager, warm up its buffers outside the graph, then compile.

    Args:
        model: the uncompiled model whose supported leaves get ghost-wrapped.
        val_batch_size: trailing rows of each batch that form the fixed validation batch.
        warmup_fn: zero-arg callable running one eager forward+backward at the real train shape.
        compile_regions: iterable of submodules to ``torch.compile`` (None => no compilation).
        extra_regions: additional non-repeated submodules to compile in place (e.g. the top-level
            ``lm_head``/``norm``); compiled only when ``compile_regions`` is also given.
        compile_kwargs: forwarded to ``torch.compile`` (default backend='inductor', fullgraph=True).
        activation_memory_budget: if set (in (0, 1]), the Inductor min-cut partitioner is told to
            recompute activations in backward to fit this fraction of the save-everything memory —
            the compile-native form of activation checkpointing. 1.0 = save everything (default);
            lower = recompute more (less peak memory, more compute). Cuts the in-graph-dot path's
            pinned ``save_for_backward`` activations. Only affects the compiled regions.

    Returns:
        The attached ``GhostDecoupledManager``. Call ``run_step_dotprod()`` then
        ``recover_train_grads()`` each step, and ``detach()`` at teardown.
    """
    mgr = GhostDecoupledManager(model, val_batch_size,
                                score_exclude_params=score_exclude_params)
    mgr.attach()

    # Eager warmup: allocate the per-layer dot/grad_val buffers before any tracing. compile() must
    # not allocate inside the traced region; the buffers become closure cells read (not allocated)
    # during tracing. Multi-shape callers (e.g. GREATS, with distinct scoring/update batch sizes)
    # pass ``warmup_shapes`` so EACH shape's buffers are allocated up front via ``prepare_shape``;
    # ``warmup_fn`` then takes the combined batch size and runs one fwd/bwd at that shape.
    if warmup_shapes:
        for total_bs in warmup_shapes:
            mgr.prepare_shape(total_bs)
            warmup_fn(total_bs)
    else:
        warmup_fn()
    model.zero_grad(set_to_none=True)

    if compile_regions is not None:
        kwargs = {"backend": "inductor", "fullgraph": True}
        if compile_kwargs:
            kwargs.update(compile_kwargs)
        if activation_memory_budget is not None:
            # Set before compile so the partitioner builds the recompute plan for these graphs.
            import torch._functorch.config as _fcfg
            _fcfg.activation_memory_budget = float(activation_memory_budget)
            print(f"[INFO] Ghost decoupled: activation_memory_budget={activation_memory_budget} "
                  "(min-cut partitioner recompute).")
        n = 0
        for region in compile_regions:
            _compile_forward_in_place(region, kwargs)
            n += 1
        # Top-level layers: compile only the in-graph (non-tied) ones. A tied layer is on the
        # capture path (eager finalize); its in-graph dot isn't what we'd be folding in, so skip it.
        tied_ids = {id(layer) for _, layer in mgr._tied_layers}
        extra = [r for r in (extra_regions or []) if id(r) not in tied_ids]
        for region in extra:
            _compile_forward_in_place(region, kwargs)
        msg = f"[INFO] Ghost decoupled: regional-compiled {n} block(s)"
        if extra:
            msg += f" + {len(extra)} top-level layer(s)"
        print(msg + f" (backend={kwargs['backend']}, fullgraph={kwargs.get('fullgraph')}).")
    else:
        print("[INFO] Ghost decoupled: attached + warmed up (no compile).")

    return mgr
