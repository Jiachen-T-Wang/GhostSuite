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
    compile_kwargs=None,
):
    """Attach the decoupled manager, warm up its buffers outside the graph, then compile.

    Args:
        model: the uncompiled model whose supported leaves get ghost-wrapped.
        val_batch_size: trailing rows of each batch that form the fixed validation batch.
        warmup_fn: zero-arg callable running one eager forward+backward at the real train shape.
        compile_regions: iterable of submodules to ``torch.compile`` (None => no compilation).
        compile_kwargs: forwarded to ``torch.compile`` (default backend='inductor', fullgraph=True).

    Returns:
        The attached ``GhostDecoupledManager``. Call ``run_step_dotprod()`` then
        ``recover_train_grads()`` each step, and ``detach()`` at teardown.
    """
    mgr = GhostDecoupledManager(model, val_batch_size)
    mgr.attach()

    # Eager warmup: allocate the per-layer dot/grad_val buffers before any tracing. compile() must
    # not allocate inside the traced region; the buffers become closure cells read (not allocated)
    # during tracing.
    warmup_fn()
    model.zero_grad(set_to_none=True)

    if compile_regions is not None:
        kwargs = {"backend": "inductor", "fullgraph": True}
        if compile_kwargs:
            kwargs.update(compile_kwargs)
        n = 0
        for region in compile_regions:
            _compile_forward_in_place(region, kwargs)
            n += 1
        print(f"[INFO] Ghost decoupled: regional-compiled {n} region(s) "
              f"(backend={kwargs['backend']}, fullgraph={kwargs.get('fullgraph')}).")
    else:
        print("[INFO] Ghost decoupled: attached + warmed up (no compile).")

    return mgr
