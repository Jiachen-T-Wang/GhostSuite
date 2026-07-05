"""OPUS scoring on the GhostSuite gradient-projection engine.

Computes, in ONE forward/backward over ``[candidates(N) ++ proxy(m)]``, the two things the
OPUS selection rule needs (github.com/gszfwsb/OPUS, ICML 2026):

- per-candidate utility ``s_i = <sketch(g_i), mean_j sketch(g_vj)>``
- the candidate-candidate similarity (Gram) ``sim_ij = <sketch(g_i), sketch(g_j)>``

where ``sketch(g)`` is the per-sample gradient in projected space. The OPUS reference
implementation's default ("random projection") mode CountSketches each per-sample gradient by
materializing it in row chunks; here the sketch is the engine's two-sided factorized projection
``P_o (Σ_t b_t a_tᵀ) P_iᵀ`` computed inside the backward hooks WITHOUT ever materializing a
per-sample gradient — same unbiased-inner-product guarantee (Gaussian ``N(0, 1/k)`` /
calibrated row-orthonormal ``P``), much less work per layer.

Preconditioning (the "optimizer-induced" part of OPUS): ``adamw_scalar`` applies OPUS's
per-layer scalar factors — the AdamW step-size factor
``C_t = lr·(1-β1)·sqrt(1-β2^t)/(1-β1^t)`` times the ``1/sqrt(numel)`` layer normalization —
to the TRAIN-side sketches, so scores carry ``c_l`` and the Gram carries ``c_l²``, exactly
the reference's left-side weighting semantics. The reference's *element-wise* AdamW diagonal
``1/(sqrt(v_t)+eps)`` is NOT supported: it cannot pass through a factorized projection
without materializing the gradient, which is the very cost this scorer avoids (a
rank-1-factored diagonal folded into ``P_o``/``P_i`` is possible future work).

Scored layers default to the transformer-block Linears (no tied ``wte``/``lm_head``) —
the recommended configuration from the GREATS pretrain experiment, and the projection
engine does not model tied-weight cross-terms.
"""

import math
import os

import torch

from ghostEngines import GradProjLoraEngine


class OpusProjScorer:
    """One-pass OPUS scorer: per-sample gradient sketches -> (scores, Gram)."""

    def __init__(self, model, optimizer, config, device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.preconditioner = config.opus_preconditioner
        if self.preconditioner not in ("none", "adamw_scalar"):
            raise ValueError(
                f"Unknown opus_preconditioner {self.preconditioner!r}; the OPUS reference's "
                "element-wise AdamW diagonal / Muon modes need materialized per-sample "
                "gradients and are not supported by the factorized-projection scorer "
                "(see module docstring)."
            )

        # proj_dir is only used if a caller ever saves projections; the scorer always
        # collects with save=False, so nothing is written there during training.
        self.engine = GradProjLoraEngine(
            module=model,
            proj_layers=config.proj_layers,
            proj_rank_total=config.proj_dim,
            proj_rank_min=config.proj_rank_min,
            proj_seed=config.proj_seed,
            proj_dtype="float32",
            proj_dir=os.path.join(config.result_dir, "opus_proj"),
            proj_row_orthonormal=config.proj_orthonormal,
        )
        # Hooks are attached only for the duration of each scoring pass (see score()), so
        # the plain update pass and evaluation run completely hook-free.

    # ------------------------------------------------------------------ #
    # Preconditioner scalars
    # ------------------------------------------------------------------ #
    def _find_group(self, param):
        for group in self.optimizer.param_groups:
            for p in group.get("params", ()):
                if p is param:
                    return group
        raise ValueError("OpusProjScorer: scored layer weight not found in any optimizer group.")

    def _optimizer_step_t(self):
        """Upstream's step counter: the max AdamW ``step`` across optimizer state, min 1."""
        t = 1
        for group in self.optimizer.param_groups:
            for p in group.get("params", ()):
                st = self.optimizer.state.get(p, None)
                if st is not None and "step" in st:
                    t = max(t, int(st["step"]))
        return t

    def layer_scales(self):
        """Per-scored-layer scalar ``c_l = C_t(group) / sqrt(numel)`` (adamw_scalar mode)."""
        t = self._optimizer_step_t()
        scales = {}
        for name, layer in self.engine.matched_layers.items():
            p = layer.weight
            group = self._find_group(p)
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            num = max(1e-16, 1.0 - beta2 ** t) ** 0.5
            den = max(1e-16, 1.0 - beta1 ** t)
            c_t = lr * (1.0 - beta1) * num / den
            scales[name] = c_t / math.sqrt(p.numel())
        return scales

    # ------------------------------------------------------------------ #
    # Scoring pass
    # ------------------------------------------------------------------ #
    def score(self, forward_fn, ctx, scaler, X_cand, Y_cand, X_val, Y_val):
        """Run the OPUS scoring pass. Returns ``(scores [N], gram [N, N])`` on CPU (float32).

        The backward is UNSCALED (sketch magnitudes feed the Boltzmann selection), so a
        GradScaler must be disabled — bf16/fp32 training only, like the GREATS fast path.
        """
        if scaler is not None and scaler.is_enabled():
            raise RuntimeError(
                "OPUS scoring reads sketches from an unscaled backward; fp16 GradScaler "
                "training is unsupported. Use --train_dtype bfloat16 (or float32)."
            )
        n_cand = X_cand.shape[0]
        L = self.config.score_seq_len
        if L is not None and L < X_cand.shape[1]:
            # Upstream scores a shorter window than it trains on (their `score_len`); prefix
            # truncation of our fixed-length rows plays the same role.
            X_cand, Y_cand = X_cand[:, :L], Y_cand[:, :L]
            X_val, Y_val = X_val[:, :L], Y_val[:, :L]
        X = torch.cat([X_cand, X_val], dim=0)
        Y = torch.cat([Y_cand, Y_val], dim=0)

        self.engine.attach(verbose=False)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            with ctx:
                loss = forward_fn(self.model, X, Y)
            loss.backward()
            sketches = self.engine.collect_batch(save=False).float()  # [N+m, K]
        finally:
            self.engine.detach(verbose=False)
            # The scoring backward's .grad is a combined-batch gradient we never step on.
            self.optimizer.zero_grad(set_to_none=True)

        s_train = sketches[:n_cand]
        s_val = sketches[n_cand:]
        if self.preconditioner == "adamw_scalar":
            scales = self.layer_scales()
            for name, (start, end) in self.engine.slice_ranges.items():
                s_train[:, start:end] *= scales[name]

        scores = s_train @ s_val.mean(dim=0)   # [N], <g_i, mean val grad> in sketch space
        gram = s_train @ s_train.T             # [N, N]
        return scores.cpu(), gram.cpu()
