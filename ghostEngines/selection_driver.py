"""Online-selection training driver, driven by a pluggable ``SelectionPolicy``.

One entry point — ``online_selection_step`` — covers the whole family:
- ``NoSelection`` : Regular baseline — no scoring pass, a plain step on the drawn batch.
- ``UpdateAll``   : graddotprod_lm — score the batch (log dots), recover over all, step. 1 backward.
- ``TopK``/``BottomK``/``Threshold`` : score, select a subset, fresh plain backward on it. 2 backwards.

This is the engine-side choreography of the ghost dance (scoring, the eager-vs-fn dot read,
recover-vs-reselect, capture toggling). It stays framework-light: the caller injects a
``forward_fn(model, X, Y) -> loss`` (so the model's forward signature stays out of the engine) and
an optimizer / GradScaler / autocast ``ctx`` / optional ``draw_microbatch``, and owns data loading,
the LR schedule, eval, and logging. The engine package therefore gains no dependency on any example
training loop. ``examples/lm/shared/selection_trainer.py`` re-exports this for backward compatibility.
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
                  manager=None, grad_accum=1, ddp=False, draw_microbatch=None):
    """Plain forward/backward + optimizer step: no val concat, dot capture off.

    Used by the Regular baseline (whole batch) and the subset update of selection policies. When
    ``draw_microbatch`` is given each microstep draws a DISTINCT (X, Y) sub-batch (so accumulation
    spans real data); otherwise every microstep reuses (X, Y). Scaler calls are no-ops when the
    scaler is disabled (bf16 / fn-path). Returns the step loss as the mean of the UNSCALED
    per-microstep losses (the backward still uses the ``1/grad_accum``-scaled loss), so logged
    losses are comparable across ``grad_accum`` settings."""
    fn = manager is not None and getattr(manager, "is_fn_path", False)
    if fn:
        manager.set_capture_enabled(False)
    try:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = None
        for micro in range(grad_accum):
            _set_ddp_sync(model, ddp, micro, grad_accum)
            Xm, Ym = (X, Y) if draw_microbatch is None else draw_microbatch(micro)[:2]
            with ctx:
                l = forward_fn(model, Xm, Ym)
                scaled = l / grad_accum if grad_accum > 1 else l
            scaler.scale(scaled).backward()
            loss_sum = l.detach() if loss_sum is None else loss_sum + l.detach()
        _clip_step(optimizer, scaler, model, grad_clip)
        optimizer.zero_grad(set_to_none=True)
    finally:
        if fn:
            manager.set_capture_enabled(True)
    return loss_sum / grad_accum


def online_selection_step(*, manager, model, optimizer, scaler, ctx, forward_fn,
                          X=None, Y=None, policy, iter_num, grad_clip,
                          grad_accum=1, save_dots=False, score_metric="dot",
                          batch_idx=None, ddp=False, draw_microbatch=None):
    """Run one online-selection step. Returns ``(scores, idx, loss)``.

    ``scores`` / ``idx`` are ``None`` for the no-selection baseline (and ``idx`` is ``None`` for
    a reuse-recovery policy, which updates on every scored sample). ``batch_idx`` is recorded in
    the logged dot entry (for offline selection replay). ``loss`` is the mean of the UNSCALED
    per-microstep losses (not pre-divided by ``grad_accum``), detached.

    Gradient accumulation (``grad_accum > 1``): pass ``draw_microbatch(micro) -> (X, Y, batch_idx)``
    so each microstep draws a DISTINCT sub-batch — replaying one batch with ``1/N`` loss scaling is
    numerically identical to ``grad_accum == 1``. The scoring pass concatenates the per-microstep
    per-sample dots into a pooled ``[N*train_bs]`` score vector; selection policies select across the
    whole pool and update on the gathered subset, while UpdateAll recovers over all via subtract-val.
    When ``draw_microbatch`` is None every microstep uses ``(X, Y, batch_idx)`` (back-compat; correct
    only at ``grad_accum == 1``)."""
    if draw_microbatch is None:
        if X is None or Y is None:
            raise ValueError("online_selection_step needs either (X, Y) or draw_microbatch.")

        def draw_microbatch(micro):
            return X, Y, batch_idx

    # --- No-selection baseline: a plain step on the drawn batch(es) (no ghost). ---
    # _plain_update draws every microstep itself (draw_microbatch is always non-None here), so
    # no batch is drawn eagerly: the baseline consumes exactly grad_accum draws per step and
    # stays batch-for-batch comparable with the scored arms.
    if not policy.scores_needed:
        loss = _plain_update(model, optimizer, scaler, ctx, forward_fn, X, Y, grad_clip,
                             manager=manager, grad_accum=grad_accum, ddp=ddp,
                             draw_microbatch=draw_microbatch)
        return None, None, loss

    if manager is None:
        raise ValueError("online_selection_step requires a ghost manager when scores are needed.")

    # --- Scoring pass: one ghost forward/backward over [batch ++ val] per microstep. ---
    manager.begin_step()
    microbatches = []   # (X, Y) per microstep, kept for the subset gather of selection policies
    loss_sum = None
    for micro in range(grad_accum):
        Xm, Ym, bidx = draw_microbatch(micro)
        microbatches.append((Xm, Ym))
        manager.attach_train_batch(Xm, Ym, iter_num, bidx)
        if getattr(manager, "is_fn_path", False):
            # In-graph buffers are per-shape (tensor batches only: the fn-path rejects dict
            # inputs at init, so ``.shape`` is safe inside this branch).
            manager.prepare_decoupled_shape(Xm.shape[0] + manager.val_batch_size)
        _set_ddp_sync(model, ddp, micro, grad_accum)
        with manager.saved_tensors_context():
            with ctx:
                Xf, Yf = manager.prepare_forward_input(Xm, Ym)
                l = forward_fn(model, Xf, Yf)
                scaled = l / grad_accum if grad_accum > 1 else l
            scaler.scale(scaled).backward()
        manager.collect_microbatch()
        loss_sum = l.detach() if loss_sum is None else loss_sum + l.detach()
    loss = loss_sum / grad_accum   # mean UNSCALED microstep loss (backward used the scaled one)
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
        # Persist the scoring dots BEFORE discarding them (save_dots would otherwise be silently
        # ignored for subset policies: discard_scores clears the log). Mirrors the reuse branch.
        if save_dots and manager.should_save_metrics(iter_num):
            manager.save_metrics(iter_num)
        # Discard the scoring grads/dots; fresh plain backward on the selected subset, gathered from
        # the (possibly several) drawn microbatches — pooled global index i maps to microstep
        # i // train_bs, local i % train_bs, so concatenating the microbatches in order indexes
        # directly by the pooled score index.
        manager.discard_scores()
        optimizer.zero_grad(set_to_none=True)
        if len(microbatches) == 1:
            Xcat, Ycat = microbatches[0]
        else:
            Xcat = torch.cat([mb[0] for mb in microbatches], dim=0)
            Ycat = torch.cat([mb[1] for mb in microbatches], dim=0)
        dev_idx = idx.to(Xcat.device)
        Xs = Xcat.index_select(0, dev_idx)
        Ys = Ycat.index_select(0, dev_idx)
        loss = _plain_update(model, optimizer, scaler, ctx, forward_fn, Xs, Ys, grad_clip,
                             manager=manager, grad_accum=1)
    return scores, idx, loss
