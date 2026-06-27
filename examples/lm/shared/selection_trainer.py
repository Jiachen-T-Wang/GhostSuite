"""Shared online-selection training step, driven by a pluggable ``SelectionPolicy``.

One entry point — ``online_selection_step`` — covers the whole family:
- ``NoSelection`` : Regular baseline — no scoring pass, a plain step on the drawn batch.
- ``UpdateAll``   : graddotprod_lm — score the batch (log dots), recover over all, step. 1 backward.
- ``TopK``/``BottomK``/``Threshold`` : score, select a subset, fresh plain backward on it. 2 backwards.

The caller supplies a ``forward_fn(model, X, Y) -> loss`` (so the model's forward signature stays
out of the engine) and owns data loading / LR schedule / eval / logging. The ghost dance —
scoring, the eager-vs-fn dot read, recover-vs-reselect, capture toggling — lives here.
"""

import torch


def _clip_step(optimizer, scaler, model, grad_clip):
    scaler.unscale_(optimizer)
    if grad_clip and grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def _set_ddp_sync(model, ddp, micro, grad_accum):
    """With DDP + grad-accum, only sync grads on the final micro-batch (no-op otherwise)."""
    if ddp:
        model.require_backward_grad_sync = (micro == grad_accum - 1)


def _plain_update(model, optimizer, scaler, ctx, forward_fn, X, Y, grad_clip,
                  manager=None, grad_accum=1, ddp=False):
    """Plain forward/backward + optimizer step on (X, Y): no val concat, dot capture off.

    Used by the Regular baseline (whole batch) and the subset update of selection policies.
    Scaler calls are no-ops when the scaler is disabled (bf16 / fn-path)."""
    fn = manager is not None and getattr(manager, "is_fn_path", False)
    if fn:
        manager.set_capture_enabled(False)
    try:
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for micro in range(grad_accum):
            _set_ddp_sync(model, ddp, micro, grad_accum)
            with ctx:
                l = forward_fn(model, X, Y)
                if grad_accum > 1:
                    l = l / grad_accum
            scaler.scale(l).backward()
            loss = l
        _clip_step(optimizer, scaler, model, grad_clip)
        optimizer.zero_grad(set_to_none=True)
    finally:
        if fn:
            manager.set_capture_enabled(True)
    return loss


def online_selection_step(*, manager, model, optimizer, scaler, ctx, forward_fn,
                          X, Y, policy, iter_num, grad_clip,
                          grad_accum=1, save_dots=False, score_metric="dot",
                          batch_idx=None, ddp=False):
    """Run one online-selection step. Returns ``(scores, idx, loss)``.

    ``scores`` / ``idx`` are ``None`` for the no-selection baseline (and ``idx`` is ``None`` for
    a reuse-recovery policy, which updates on every scored sample). ``batch_idx`` is recorded in
    the logged dot entry (for offline selection replay)."""
    # --- No-selection baseline: a plain step on the whole drawn batch (no ghost). ---
    if not policy.scores_needed:
        loss = _plain_update(model, optimizer, scaler, ctx, forward_fn, X, Y, grad_clip,
                             manager=manager, grad_accum=grad_accum, ddp=ddp)
        return None, None, loss

    if manager is None:
        raise ValueError("online_selection_step requires a ghost manager when scores are needed.")

    # --- Scoring pass: the single ghost forward/backward over [batch ++ val]. ---
    manager.attach_train_batch(X, Y, iter_num, batch_idx)
    manager.prepare_decoupled_shape(X.shape[0] + manager.val_batch_size)
    loss = None
    for micro in range(grad_accum):
        _set_ddp_sync(model, ddp, micro, grad_accum)
        with manager.saved_tensors_context():
            with ctx:
                Xf, Yf = manager.prepare_forward_input(X, Y)
                l = forward_fn(model, Xf, Yf)
                if grad_accum > 1:
                    l = l / grad_accum
            scaler.scale(l).backward()
        loss = l
    scores = manager.read_scores(metric=score_metric)
    idx = policy.select(scores)

    if policy.reuse_recovery:
        # Update on ALL scored samples, reusing the scoring backward (subtract-val recovery).
        if idx is not None:
            raise ValueError(
                "reuse_recovery policies must update on all scored samples (select() must return "
                "None): subtract-val recovers the whole-batch mean gradient, not a subset.")
        manager.recover_train_grads()
        _clip_step(optimizer, scaler, model, grad_clip)
        if save_dots and manager.should_save_metrics(iter_num):
            manager.save_metrics(iter_num)
        manager.finish_step()
        optimizer.zero_grad(set_to_none=True)
    else:
        # Discard the scoring grads/dots; fresh plain backward on the selected subset.
        manager.discard_scores()
        optimizer.zero_grad(set_to_none=True)
        dev_idx = idx.to(X.device)
        Xs = X.index_select(0, dev_idx)
        Ys = Y.index_select(0, dev_idx)
        loss = _plain_update(model, optimizer, scaler, ctx, forward_fn, Xs, Ys, grad_clip,
                             manager=manager, grad_accum=1)
    return scores, idx, loss
