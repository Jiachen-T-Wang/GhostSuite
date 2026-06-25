from typing import Optional, TYPE_CHECKING

import os

import torch
import torch.nn.functional as F
import transformers.pytorch_utils
from torch import nn

from jaxtyping import Float, Int


# subtract-val (default on): stash the validation-gradient contribution per weight so the engine
# can recover the train grad post-backward as (total/train)*(full_grad - grad_val), instead of
# masking the saved activation. See autograd_grad_sample_dotprod._SUBTRACT_VAL. The default must
# match that module's default, since the engine's subtract-val recovery requires this grad_val.
_SUBTRACT_VAL = os.getenv("GHOST_SUBTRACT_VAL", "1") == "1"


def _maybe_store_grad_val(param, grad_val) -> None:
    if _SUBTRACT_VAL and param is not None:
        # fp32 for a clean subtraction against the fp32 autograd grad.
        param._ghost_grad_val = grad_val.detach().to(torch.float32)


# --------------------------------------------------------------------------- #
# Tied-weight handling
#
# When one weight tensor is shared by >=2 supported modules (e.g. GPT-2 ties the
# token embedding `wte` to the output Linear `lm_head`), the per-module samplers
# would each write `weight.grad_dot_prod` / `weight._ghost_grad_val` and clobber
# one another -- and even if summed, the per-use dot products miss the
# cross-terms <g_use_a, g_use_b> of the (shared) parameter's true gradient.
#
# Correct handling: each use accumulates its validation-gradient aggregate into
# `weight._ghost_grad_val` (so the post-backward train-grad recovery sees the
# full val grad), and stashes its TRAIN factors. `finalize_tied_param` then
# materializes the combined per-sample gradient G_i = sum_uses g_use_i for the
# shared weight and computes the exact dot product <G_i, g_val_total>, the
# per-sample train-grad norm ||G_i||, and the val-grad norm -- all including the
# cross-terms. This is the only parameter materialized per-sample, so the memory
# cost is one [out, in] tensor at a time.
# --------------------------------------------------------------------------- #
def stash_tied_contribution(layer, A, B, val_batch_size, log_grad_norms=False,
                            accum_dtype: torch.dtype = torch.float32) -> None:
    weight = layer.weight
    A = A.detach()
    B = B.detach()
    train_bs = A.size(0) - val_batch_size
    if train_bs <= 0:
        raise ValueError("stash_tied_contribution: non-positive train batch size.")
    vocab, embed = weight.shape

    if isinstance(layer, nn.Embedding):
        kind = "embedding"
        A_long = A.long()
        A_tr, A_val = A_long[:train_bs], A_long[train_bs:]
        Bc = B.to(accum_dtype)
        B_tr, B_val = Bc[:train_bs], Bc[train_bs:]
        gval = torch.zeros((vocab, embed), dtype=accum_dtype, device=B.device)
        gval.index_add_(0, A_val.reshape(-1), B_val.reshape(-1, embed))
        train_factors = (kind, A_tr, B_tr)
    elif isinstance(layer, nn.Linear):
        kind = "linear"
        d_in, d_out = A.size(-1), B.size(-1)
        Af = A.to(accum_dtype).reshape(-1, d_in)
        Bf = B.to(accum_dtype).reshape(-1, d_out)
        seq = Af.size(0) // A.size(0)
        split = train_bs * seq
        gval = torch.matmul(Bf[split:].T, Af[split:])           # [d_out, d_in] = [vocab, embed]
        train_factors = (kind, Af[:split].view(train_bs, seq, d_in),
                         Bf[:split].view(train_bs, seq, d_out))
    else:
        raise TypeError(f"Tied weight on unsupported module type {type(layer).__name__}.")

    prev = getattr(weight, "_ghost_grad_val", None)
    weight._ghost_grad_val = gval if prev is None else (prev + gval)
    # Keep a reference for finalize_tied_param: the subtract-val train-grad
    # recovery (prepare_gradients) deletes _ghost_grad_val before aggregate_and_log
    # runs, but the tensor survives through this alias.
    weight._ghost_tied_gval = weight._ghost_grad_val
    if not hasattr(weight, "_ghost_tied_stash"):
        weight._ghost_tied_stash = []
        weight._ghost_tied_train_bs = train_bs
        weight._ghost_tied_log_norms = bool(log_grad_norms)
    weight._ghost_tied_stash.append(train_factors)


def finalize_tied_param(weight, accum_dtype: torch.dtype = torch.float32) -> None:
    stash = weight._ghost_tied_stash
    train_bs = weight._ghost_tied_train_bs
    log_norms = weight._ghost_tied_log_norms
    gval = weight._ghost_tied_gval.to(accum_dtype)              # [vocab, embed] total val aggregate
    vocab, embed = weight.shape
    device = gval.device

    if not log_norms:
        # Vectorized (default): dot[i] = <G_i, gval> = sum_uses <G_i^use, gval>, each per-use term
        # in ghost (associativity) form so no [vocab, embed] per-sample materialization and no
        # Python loop over train_bs. gval already holds the summed val aggregate, so the cross-terms
        # between uses (e.g. <g_emb, g_lin>) are included.
        dot = torch.zeros(train_bs, dtype=accum_dtype, device=device)
        for kind, A_tr, B_tr in stash:
            if kind == "embedding":
                # A_tr [train, seq] indices; B_tr [train, seq, embed]. Gather gval rows at the
                # token ids and dot with B: <G_emb, gval> = sum_t B[t] . gval[idx_t].
                contrib = (B_tr * gval[A_tr]).sum(dim=(1, 2))
            else:  # linear: A_tr [train, seq, embed], B_tr [train, seq, vocab]
                # <B^T A, gval> = sum_t B[t] . (gval^T B[t] folded) = sum_t (B[t] @ gval) . A[t].
                proj = torch.matmul(B_tr, gval)                # [train, seq, embed] (contract vocab)
                contrib = (A_tr * proj).sum(dim=(1, 2))
            dot = dot + contrib
        weight.grad_dot_prod = dot
        del (weight._ghost_tied_stash, weight._ghost_tied_train_bs,
             weight._ghost_tied_log_norms, weight._ghost_tied_gval)
        return

    # log_grad_norms path: the exact per-sample train-grad norm ||G_i|| has an O(seq^2) cross-term,
    # so fall back to materializing G_i one sample at a time (rare; log_grad_norms defaults False).
    dot = torch.empty(train_bs, dtype=accum_dtype, device=device)
    tnorm = torch.empty(train_bs, dtype=accum_dtype, device=device)
    for i in range(train_bs):
        Gi = torch.zeros((vocab, embed), dtype=accum_dtype, device=device)
        for kind, A_tr, B_tr in stash:
            if kind == "embedding":
                Gi.index_add_(0, A_tr[i].reshape(-1), B_tr[i].reshape(-1, embed))
            else:  # linear: G_i = sum_t b_t a_t^T = B_i^T A_i
                Gi += torch.matmul(B_tr[i].transpose(0, 1), A_tr[i])
        dot[i] = (Gi * gval).sum()
        tnorm[i] = (Gi * Gi).sum()

    weight.grad_dot_prod = dot
    weight.grad_train_norm = tnorm
    weight.grad_val_norm_sq = (gval * gval).sum()
    del (weight._ghost_tied_stash, weight._ghost_tied_train_bs,
         weight._ghost_tied_log_norms, weight._ghost_tied_gval)


def _should_use_ghost_computation(
    layer: nn.Module,
    A: Float[torch.Tensor, "batch ..."],
    B: Float[torch.Tensor, "batch ..."],
    conv: bool = False,
):
    """
    Determines whether to use the "ghost" computation method for linear layers.
    Args:
        layer: The neural network layer.
        A: The activation tensor.
        B: The backpropagation tensor.
        conv: Flag indicating if the layer is a convolutional layer.
    """

    # The check only needs to be performed once per layer.
    if hasattr(layer, "use_ghost_computation"):
        return

    if not conv:
        # For linear layers
        seq_len = torch.prod(torch.tensor(A.shape[1:-1])).item() if A.dim() > 2 else 1
    else:
        # For convolutional layers (after unfolding)
        seq_len = A.shape[-1]

    # The total number of parameters in the weight matrix
    num_weight_params = layer.weight.numel()

    # criterion: 2*seq_len^2 <= num_weight_params (this is just a heuristic)
    layer.use_ghost_computation = bool(2 * seq_len**2 <= num_weight_params)


def _create_or_accumulate_train_grad(
    param: Float[torch.Tensor, "..."],
    grad: Float[torch.Tensor, "..."],
) -> None:
    """Creates or accumulates a gradient for a given parameter in the .train_grad attribute.

    This function adds a computed gradient to the .train_grad attribute of a parameter.
    It handles both the initial creation of the .train_grad attribute and the
    subsequent accumulation of gradients, preventing contamination of the standard
    .grad attribute.

    Args:
        param: The model parameter to which the training gradient will be added.
        grad: The newly computed training gradient.
    """
    # Ensure the new gradient has the same shape as the parameter.
    assert grad.shape == param.shape, \
        f"Gradient shape ({grad.shape}) does not match parameter shape ({param.shape})"

    # Detach the gradient to ensure it's not part of any further computation graph.
    new_grad = grad.detach()

    # Check if the parameter already has a train_grad attribute.
    if hasattr(param, 'train_grad'):
        # Add the new gradient to the existing one in-place.
        param.train_grad.add_(new_grad)
    else:
        # If the parameter does not have a train_grad attribute yet, create it.
        param.train_grad = new_grad


def _compute_linear_dot_product(
    layer: nn.Linear,
    A: Float[torch.Tensor, "batch ... d_in"],
    B: Float[torch.Tensor, "batch ... d_out"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
):
    """Computes the gradient dot-product for an nn.Linear layer."""

    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    total_bs = A.size(0)
    train_bs = total_bs - val_batch_size
    
    # Setup Dimensions
    d_in = A.size(-1)
    d_out = B.size(-1)
    A_flat = A.to(compute_dtype).reshape(-1, d_in)
    B_flat = B.to(compute_dtype).reshape(-1, d_out)

    # Tokens-per-sample, inferred from the flattened batch so this works for both
    # 3D (B, T, C) LM activations (seq_len = T) and 2D (B, C) MLP activations
    # (seq_len = 1), as well as any higher-rank input.
    seq_len = A_flat.size(0) // total_bs
    split_idx = train_bs * seq_len

    A_train = A_flat[:split_idx]  # [train_bs*seq_len, d_in]
    A_val = A_flat[split_idx:]    # [val_bs*seq_len, d_in]
    B_train = B_flat[:split_idx]  # [train_bs*seq_len, d_out]
    B_val = B_flat[split_idx:]    # [val_bs*seq_len, d_out]

    # Pre-declare variables for logging reuse
    grad_val_for_norm = None
    
    # Decide whether to use ghost computation
    _should_use_ghost_computation(layer, A, B)

    if layer.use_ghost_computation:
        # --- ghost computation with associativity trick ---

        # compute validation gradient [d_out, d_in]
        grad_val = torch.matmul(B_val.T, A_val)

        # project grad_val by B_train to remove the d_out dimension 
        # [train_bs*seq_len, d_out] @ [d_out, d_in] = [train_bs*seq_len, d_in]
        grad_val_projected = torch.matmul(B_train, grad_val)

        # element-wise product and sum over the d_in dimension
        # [ train_bs*seq_len ]
        token_scores = torch.sum(A_train * grad_val_projected, dim=1)

        # [ train_bs*seq_len ] -> [ train_bs ]
        layer.weight.grad_dot_prod = token_scores.view(train_bs, seq_len).sum(dim=1)
        _maybe_store_grad_val(layer.weight, grad_val)

        if log_grad_norms:
            # Per-sample ||G_train_i||^2 via the Gram trick (no materialization):
            # ||sum_t b_t a_t^T||^2 = sum_{t,t'} <a_t,a_t'> <b_t,b_t'>.
            A_tr3 = A_train.view(train_bs, seq_len, d_in)
            B_tr3 = B_train.view(train_bs, seq_len, d_out)
            AA = torch.bmm(A_tr3, A_tr3.transpose(1, 2))
            BB = torch.bmm(B_tr3, B_tr3.transpose(1, 2))
            layer.weight.grad_train_norm = (AA * BB).to(accum_dtype).sum(dim=[1, 2])
            layer.weight.grad_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()

    else:

        # --- materialize gradients ---
        # compute validation gradient [d_out, d_in]
        grad_val = torch.matmul(B_val.T, A_val)

        # Reshape for sum-over-T contraction
        A_train_3d = A_train.view(train_bs, seq_len, d_in)
        B_train_3d = B_train.view(train_bs, seq_len, d_out)

        B_train_T = B_train_3d.transpose(1, 2).contiguous()

        # grad_train: collection of per-sample train gradients [train_bs, d_out, d_in]
        grad_train = torch.bmm(B_train_T, A_train_3d)

        layer.weight.grad_dot_prod = torch.matmul(grad_train.view(train_bs, -1), grad_val.view(-1))
        _maybe_store_grad_val(layer.weight, grad_val)

        if log_grad_norms:
            layer.weight.grad_train_norm = (grad_train.to(accum_dtype) ** 2).sum(dim=[1, 2])
            layer.weight.grad_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()


def _compute_linear_train_grad(
    layer: nn.Linear,
    A: Float[torch.Tensor, "batch ... in_features"],
    B: Float[torch.Tensor, "batch ... out_features"],
    val_batch_size: int,
):
    """
    Computes the training gradient for an nn.Linear layer's weight.
    This version always computes the average gradient to match PyTorch's default behavior.
    """
    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return None # Return None if there's nothing to compute

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Ensure consistent dtype for einsum (e.g., when activations are bf16 and backprops are fp32)
    param_dtype = layer.weight.dtype
    if A_train.dtype != param_dtype:
        A_train = A_train.to(param_dtype)
    if B_train.dtype != param_dtype:
        B_train = B_train.to(param_dtype)

    # Compute the SUM of gradients over the training batch
    grad_weight = torch.einsum('b...p,b...d->pd', B_train, A_train)

    # Always divide by the number of training samples to get the AVERAGE gradient
    grad_weight /= train_batch_size
    
    return grad_weight




# Embedding Layer Implementation
# #############################################################################

def _compute_embedding_dot_product(
    layer: nn.Embedding,
    A: Int[torch.Tensor, "batch ..."],
    B: Float[torch.Tensor, "batch ... embed_dim"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
):
    """Computes the gradient dot-product for an nn.Embedding layer."""

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Ensure the index tensors are of integer type (long) before using them.
    A_train_long = A_train.long()
    A_val_long = A_val.long()
    B_train_c = B_train.to(compute_dtype)
    B_val_c = B_val.to(compute_dtype)

    vocab_size, d_f = layer.weight.shape
    grad_val = torch.zeros((vocab_size, d_f), dtype=compute_dtype, device=B_val.device)
    grad_val.index_add_(
        0,
        A_val_long.reshape(-1),                     # indices  [val_batch * seq]
        B_val_c.reshape(-1, d_f)                    # vectors  [val_batch * seq, d_f]
    )

    dot_products = (B_train_c * grad_val[A_train_long]).to(accum_dtype).sum(dim=[1, 2])
    layer.weight.grad_dot_prod = dot_products
    _maybe_store_grad_val(layer.weight, grad_val)

    if log_grad_norms:
        # Compute per-sample train grad norm using unique tokens per sample
        train_norms = []
        for sample_tokens, sample_B in zip(A_train_long, B_train_c):
            unique_tok, inverse = torch.unique(sample_tokens, return_inverse=True)
            agg = torch.zeros((unique_tok.numel(), sample_B.size(-1)), device=sample_B.device, dtype=sample_B.dtype)
            agg.index_add_(0, inverse, sample_B)
            train_norms.append((agg.to(accum_dtype) ** 2).sum())
        layer.weight.grad_train_norm = torch.stack(train_norms)

        # Validation aggregated gradient norm
        layer.weight.grad_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()


def _compute_embedding_train_grad(
    layer: nn.Embedding,
    A: Int[torch.Tensor, "batch ..."],
    B: Float[torch.Tensor, "batch ... embed_dim"],
    val_batch_size: int,
):
    """
    Computes the training gradient for an nn.Embedding layer.
    This version always computes the average gradient.
    """

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return None

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Match dtype with embedding weights to avoid mixed-type index_add
    param_dtype = layer.weight.dtype
    if B_train.dtype != param_dtype:
        B_train = B_train.to(param_dtype)

    A_train_long = A_train.long()

    grad_weight = torch.zeros_like(layer.weight)
    grad_weight.index_add_(0, A_train_long.reshape(-1), B_train.reshape(-1, B_train.shape[-1]))

    # Always divide by the number of training samples to get the AVERAGE gradient
    grad_weight /= train_batch_size
    
    return grad_weight



# LayerNorm Layer Implementation
# #############################################################################

def _compute_layernorm_dot_product(
    layer: nn.LayerNorm,
    A: Float[torch.Tensor, "batch ... features"],
    B: Float[torch.Tensor, "batch ... features"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
):
    """Computes the gradient dot-product for an nn.LayerNorm layer."""

    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return

    A_train, A_val = torch.split(A.to(compute_dtype), [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B.to(compute_dtype), [train_batch_size, val_batch_size], dim=0)
    
    # --- Weight dot product ---
    # The gradient for the weight is B * normalized_A
    normalized_A_train = F.layer_norm(A_train.to(accum_dtype), layer.normalized_shape, eps=layer.eps)
    normalized_A_val = F.layer_norm(A_val.to(accum_dtype), layer.normalized_shape, eps=layer.eps)

    B_train_accum = B_train.to(accum_dtype)
    B_val_accum = B_val.to(accum_dtype)

    grad_weight_train = B_train_accum * normalized_A_train
    grad_weight_val = B_val_accum * normalized_A_val

    # Reduce training per-sample gradients over non-feature dims only → [B_train, F]
    # Keep the last dim (feature) intact for a correct dot with validation vector.
    if grad_weight_train.dim() >= 2:
        sum_dims_train = list(range(1, grad_weight_train.dim() - 1))
        per_sample_grad_weight = grad_weight_train.sum(dim=sum_dims_train) if sum_dims_train else grad_weight_train
    else:
        per_sample_grad_weight = grad_weight_train

    # Aggregate validation gradient over batch and token dims only → [F]
    sum_dims_val = list(range(grad_weight_val.dim() - 1))
    total_grad_weight_val = grad_weight_val.sum(dim=sum_dims_val)

    # Feature-wise inner product to obtain per-sample scalars → [B_train]
    layer.weight.grad_dot_prod = torch.einsum(
        'bf,f->b', per_sample_grad_weight.to(accum_dtype), total_grad_weight_val.to(accum_dtype)
    )
    # subtract-val: stash the validation weight-gradient for post-backward train-grad recovery.
    _maybe_store_grad_val(layer.weight, total_grad_weight_val)
    weight_train_norm = None
    weight_val_norm_sq = None
    if log_grad_norms:
        weight_train_norm = (per_sample_grad_weight.to(accum_dtype) ** 2).sum(dim=1)
        weight_val_norm_sq = (total_grad_weight_val.to(accum_dtype) ** 2).sum()

    # --- Bias dot product ---
    if layer.bias is not None:
        # Bias gradient is B; reduce non-feature dims for train → [B_train, F]
        if B_train_accum.dim() >= 2:
            sum_dims_train = list(range(1, B_train_accum.dim() - 1))
            per_sample_grad_bias = B_train_accum.sum(dim=sum_dims_train) if sum_dims_train else B_train_accum
        else:
            per_sample_grad_bias = B_train_accum

        # Validation aggregate over batch and token dims only → [F]
        sum_dims_val = list(range(B_val_accum.dim() - 1))
        total_grad_bias_val = B_val_accum.sum(dim=sum_dims_val)

        layer.bias.grad_dot_prod = torch.einsum(
            'bf,f->b', per_sample_grad_bias.to(accum_dtype), total_grad_bias_val.to(accum_dtype)
        )
        # subtract-val: stash the validation bias-gradient for train-grad recovery.
        _maybe_store_grad_val(layer.bias, total_grad_bias_val)
        bias_train_norm = None
        bias_val_norm_sq = None
        if log_grad_norms:
            bias_train_norm = (per_sample_grad_bias.to(accum_dtype) ** 2).sum(dim=1)
            bias_val_norm_sq = (total_grad_bias_val.to(accum_dtype) ** 2).sum()
    else:
        bias_train_norm = None
        bias_val_norm_sq = None

    if log_grad_norms:
        layer.weight.grad_train_norm = weight_train_norm
        layer.weight.grad_val_norm_sq = weight_val_norm_sq
        if layer.bias is not None:
            layer.bias.grad_train_norm = bias_train_norm
            layer.bias.grad_val_norm_sq = bias_val_norm_sq


def _compute_layernorm_train_grad(
    layer: nn.LayerNorm,
    A: Float[torch.Tensor, "batch ... features"],
    B: Float[torch.Tensor, "batch ... features"],
    val_batch_size: int
) -> None:
    """
    Computes and directly applies the training gradient for an nn.LayerNorm layer.

    --- Design Reasoning ---
    This function is handled differently from `_compute_linear_train_grad` because
    LayerNorm's gradient calculations are self-contained. The gradients for both
    its parameters (weight and bias) are computed from the same inputs (A and B).

    - Weight Gradient: Depends on both the input activations `A` (specifically, the
      normalized version of `A`) and the backpropagated gradients `B`.
    - Bias Gradient: Depends only on the backpropagated gradients `B`.

    Because both computations use the same `A` and `B` tensors and are closely
    related, it is cleaner and more efficient to calculate and apply them within
    this single function. This approach avoids passing intermediate results (like
    `normalized_A`) out to a dispatcher.

    Therefore, this function computes and applies the gradients for both weight
    and bias internally and returns `None` to signal to the dispatcher that its
    work for this layer is complete.
    """

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute gradients, check batch sizes.")
    
    # debug: print out the shapes of A and B & the first few elements
    # print(f"[Grad] Layer Name: {layer.__class__.__name__}")
    # print(f"[Grad] A shape: {A.shape}, B shape: {B.shape}, val_batch_size: {val_batch_size}")
    # print(f"[Grad] A (first 5): {A[:5]}")
    # print(f"B (first 5): {B[:5]}")

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # --- Weight (gamma) gradient ---
    # The gradient is B * normalized_A. We sum over all dimensions except the
    # final feature dimension to match the shape of the weight parameter.
    normalized_A_train = F.layer_norm(A_train, layer.normalized_shape, eps=layer.eps)
    grad_weight = (B_train * normalized_A_train).sum(dim=list(range(A_train.dim() - 1)))

    grad_weight /= train_batch_size

    # Apply the computed weight gradient directly.
    _create_or_accumulate_train_grad(layer.weight, grad_weight)

    # --- Bias (beta) gradient ---
    if layer.bias is not None:
        # The gradient is just B, summed over all dimensions except features.
        grad_bias = B_train.sum(dim=list(range(B_train.dim() - 1)))

        grad_bias /= train_batch_size

        # Apply the computed bias gradient directly.
        _create_or_accumulate_train_grad(layer.bias, grad_bias)

    # Return None because this function handles its own gradient application.
    return None


def _compute_rmsnorm_dot_product(
    layer: nn.RMSNorm,
    A: Float[torch.Tensor, "batch ... features"],
    B: Float[torch.Tensor, "batch ... features"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
):
    """Compute gradient dot-product for nn.RMSNorm (weight-only)."""
    """
    A: [batch, seq, dim] (normalized by RMSNorm)
    B: [batch, seq, dim] (backpropagated gradients)
    """

    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    A = A.to(compute_dtype)
    B = B.to(compute_dtype)

    train_batch_size = A.size(0) - val_batch_size

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    eps = getattr(layer, "eps", 1e-5)
    rms_train = torch.sqrt((A_train.to(accum_dtype) ** 2).mean(dim=-1, keepdim=True) + eps)
    rms_val = torch.sqrt((A_val.to(accum_dtype) ** 2).mean(dim=-1, keepdim=True) + eps)

    norm_A_train = (A_train.to(accum_dtype) / rms_train)
    norm_A_val = (A_val.to(accum_dtype) / rms_val)

    # Accumulate in fp32 for numerical stability.
    grad_weight_train = B_train.to(accum_dtype) * norm_A_train
    grad_weight_val = B_val.to(accum_dtype) * norm_A_val

    sum_dims_train = list(range(1, grad_weight_train.dim() - 1))
    per_sample_grad_weight = grad_weight_train.sum(dim=sum_dims_train) if sum_dims_train else grad_weight_train

    sum_dims_val = list(range(grad_weight_val.dim() - 1))
    total_grad_weight_val = grad_weight_val.sum(dim=sum_dims_val)

    layer.weight.grad_dot_prod = torch.einsum(
        "bf,f->b", per_sample_grad_weight.to(accum_dtype), total_grad_weight_val.to(accum_dtype)
    )
    _maybe_store_grad_val(layer.weight, total_grad_weight_val)

    if log_grad_norms:
        layer.weight.grad_train_norm = (per_sample_grad_weight.to(accum_dtype) ** 2).sum(dim=1)
        layer.weight.grad_val_norm_sq = (total_grad_weight_val.to(accum_dtype) ** 2).sum()


def _compute_rmsnorm_train_grad(
    layer: nn.RMSNorm,
    A: Float[torch.Tensor, "batch ... features"],
    B: Float[torch.Tensor, "batch ... features"],
    val_batch_size: int,
):
    """Compute and apply averaged training gradient for nn.RMSNorm weight."""
    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute gradients, check batch sizes.")

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    eps = getattr(layer, "eps", 1e-5)
    rms_train = torch.sqrt((A_train.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    norm_A_train = (A_train.float() / rms_train).to(B_train.dtype)

    grad_weight = (B_train * norm_A_train).sum(dim=list(range(A_train.dim() - 1)))
    grad_weight = grad_weight.to(layer.weight.dtype)
    grad_weight /= train_batch_size

    return grad_weight


def _compute_Conv1D_dot_product(
    layer: nn.Linear,
    A: Float[torch.Tensor, "batch seq in_features"],
    B: Float[torch.Tensor, "batch seq out_features"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
) -> None:
    """
    Computes the gradient dot-product between the validation gradient and each of the
    training data's per-sample gradients for a Conv1D layer.

    Args:
        layer: The Conv1D layer.
        A: The input activations from the combined (train + val) batch.
        B: The output gradients from the combined (train + val) batch.
        val_batch_size: The number of samples in the validation batch.
    """

    if A is None:
        raise ValueError("Input activations A cannot be None.")
    if B is None:
        raise ValueError("Output gradients B cannot be None.")

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    A_c = A.to(compute_dtype)
    B_c = B.to(compute_dtype)

    train_batch_size = A_c.size(0) - val_batch_size

    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    A_train, A_val = torch.split(A_c, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B_c, [train_batch_size, val_batch_size], dim=0)
    A_train_f = A_train.to(accum_dtype)
    A_val_f = A_val.to(accum_dtype)
    B_train_f = B_train.to(accum_dtype)
    B_val_f = B_val.to(accum_dtype)

    # Always check if ghost computation should be used.
    _should_use_ghost_computation(layer, A_c, B_c)

    if A_c.dim() > 3:
        raise ValueError("Currently we only expect 3D input tensors (batch, seq_len, features). Extending this to 4D or higher requires additional handling.")

    weight_train_norm = None
    weight_val_norm_sq = None

    # --- Compute weight gradient dot product ---
    # The validation gradient is the sum of the validation samples' per-sample
    # gradients: grad_val[d, p] = sum_j sum_t A_val[j, t, d] * B_val[j, t, p].
    # Each sample j is contracted with itself, so there are no cross-terms between
    # distinct validation samples. (Summing A_val / B_val over the batch dimension
    # *before* the outer product would introduce spurious j != k cross-terms.)
    if layer.use_ghost_computation:
        # Materialize the (single) validation gradient, then project it through the
        # per-sample train output grads and contract with the train inputs. No
        # per-sample train gradient is ever materialized.
        grad_val = torch.einsum('jtd,jtp->dp', A_val_f, B_val_f)
        grad_val_proj = torch.einsum('bsp,dp->bsd', B_train_f, grad_val)
        token_scores = torch.einsum('bsd,bsd->bs', A_train_f, grad_val_proj)
        layer.weight.grad_dot_prod = token_scores.sum(dim=1).to(accum_dtype)

        if log_grad_norms:
            grad_train = torch.einsum('btd,btp->bpd', A_train_f, B_train_f)
            weight_train_norm = (grad_train.to(accum_dtype) ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()

    else:
        # Materialize gradients to compute the dot product
        grad_train = torch.einsum('b...d, b...p->bpd', A_train_f, B_train_f).detach()
        grad_val = torch.einsum('jtd,jtp->pd', A_val_f, B_val_f).detach()
        layer.weight.grad_dot_prod = torch.einsum('pd,bpd->b', grad_val, grad_train)
        if log_grad_norms:
            weight_train_norm = (grad_train.to(accum_dtype) ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()

    # --- Compute bias gradient dot product ---
    if layer.bias is not None:

        # For a tensor B of shape [batch, seq_len, features], the per-sample
        # bias grad is B.sum(dim=1). The total validation grad is B_val.sum(dim=[0, 1]).
        if B.dim() >= 2:
            # Sum over all dimensions except the last one (the feature dimension)
            sum_dims_val = list(range(B_val_f.dim() - 1))
            grad_bias_val = B_val_f.sum(dim=sum_dims_val)

            # For the training batch, we want per-sample gradients, so we only
            # sum over the sequence/spatial dimensions, not the batch dimension.
            sum_dims_train = list(range(1, B_train_f.dim() - 1))
            grad_bias_train = B_train_f.sum(dim=sum_dims_train)
        else: # Should not happen, but as a safeguard
            grad_bias_train = B_train_f
            grad_bias_val = B_val_f.sum(dim=0)

        # The dot product is a simple einsum between the total validation bias grad
        # and the per-sample training bias grads.
        # grad_bias_val shape: [features]
        # grad_bias_train shape: [batch, features]
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)

        if log_grad_norms:
            layer.bias.grad_train_norm = (grad_bias_train.to(accum_dtype) ** 2).sum(dim=1)
            layer.bias.grad_val_norm_sq = (grad_bias_val.to(accum_dtype) ** 2).sum()

    if log_grad_norms:
        layer.weight.grad_train_norm = weight_train_norm
        layer.weight.grad_val_norm_sq = weight_val_norm_sq


def _compute_Conv1D_train_grad(
    layer: transformers.pytorch_utils.Conv1D,
    A: Float[torch.Tensor, "batch seq in_features"],
    B: Float[torch.Tensor, "batch seq out_features"],
    val_batch_size: int,
) -> torch.Tensor:
    """
    Compute the training gradient for a Conv1D layer's weight and return the
    averaged gradient across the training batch.
    
    Args:
        layer: The Conv1D layer.
        A: The input activations from the combined (train + val) batch.
        B: The output gradients from the combined (train + val) batch.
        val_batch_size: The number of samples in the validation batch.
    
    Returns:
        The averaged gradient for the layer's weight parameter.
    """

    # Determine training batch size from the total size and validation size.
    train_batch_size = A.size(0) - val_batch_size

    # Isolate the activations and backpropagated gradients for the training data.
    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    param_dtype = layer.weight.dtype
    if A_train.dtype != param_dtype:
        A_train = A_train.to(param_dtype)
    if B_train.dtype != param_dtype:
        B_train = B_train.to(param_dtype)

    # Compute the summed gradient for the training batch.
    grad_weight = torch.einsum('b...d,b...p->dp', A_train, B_train)

    grad_weight /= train_batch_size

    return grad_weight






def _compute_conv2d_dot_product(
    layer: nn.Conv2d,
    A: Float[torch.Tensor, "batch in_channels height width"],
    B: Float[torch.Tensor, "batch out_channels out_height out_width"],
    val_batch_size: int,
    log_grad_norms: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    accum_dtype: torch.dtype = torch.float32,
) -> None:
    """Compute gradient dot-products for nn.Conv2d."""

    A = A.detach()
    B = B.detach()

    if compute_dtype is None:
        compute_dtype = B.dtype if B.is_floating_point() else layer.weight.dtype
    if accum_dtype is None:
        accum_dtype = torch.float32

    A_c = A.to(compute_dtype)
    B_c = B.to(compute_dtype)

    train_batch_size = A_c.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product")

    A_train, A_val = torch.split(A_c, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B_c, [train_batch_size, val_batch_size], dim=0)

    unfold_params = dict(
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )

    A_train_u = F.unfold(A_train, **unfold_params)
    A_val_u = F.unfold(A_val, **unfold_params)

    B_train_r = B_train.reshape(B_train.size(0), B_train.size(1), -1)
    B_val_r = B_val.reshape(B_val.size(0), B_val.size(1), -1)

    _should_use_ghost_computation(layer, A_train_u, B_train_r, conv=True)

    weight_train_norm = None
    weight_val_norm_sq = None

    # The validation gradient sums each validation sample's per-sample gradient,
    # contracting every sample with itself over the spatial locations:
    # grad_val[p, i] = sum_j sum_k B_val[j, p, k] * A_val[j, i, k]. Summing the
    # unfolded activations over the batch dimension before the outer product would
    # introduce spurious cross-terms between distinct validation samples.
    if layer.use_ghost_computation:
        A_train_u_f = A_train_u.to(accum_dtype)
        B_train_r_f = B_train_r.to(accum_dtype)
        A_val_u_f = A_val_u.to(accum_dtype)
        B_val_r_f = B_val_r.to(accum_dtype)
        # Materialize the (single) validation gradient, then project it through the
        # per-sample train output grads and contract with the train inputs.
        grad_val = torch.einsum('jik,jpk->pi', A_val_u_f, B_val_r_f)
        grad_val_proj = torch.einsum('bpk,pi->bik', B_train_r_f, grad_val)
        layer.weight.grad_dot_prod = torch.einsum('bik,bik->b', A_train_u_f, grad_val_proj).to(accum_dtype)
        if log_grad_norms:
            grad_train = torch.einsum('bik,bpk->bpi', A_train_u_f, B_train_r_f)
            weight_train_norm = (grad_train.to(accum_dtype) ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()
    else:
        A_train_u_f = A_train_u.to(accum_dtype)
        B_train_r_f = B_train_r.to(accum_dtype)
        A_val_u_f = A_val_u.to(accum_dtype)
        B_val_r_f = B_val_r.to(accum_dtype)
        grad_train = torch.einsum('bik,bpk->bpi', A_train_u_f, B_train_r_f)
        grad_val = torch.einsum('jik,jpk->pi', A_val_u_f, B_val_r_f)
        layer.weight.grad_dot_prod = torch.einsum('pi,bpi->b', grad_val, grad_train)
        if log_grad_norms:
            weight_train_norm = (grad_train.to(accum_dtype) ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.to(accum_dtype) ** 2).sum()

    if layer.bias is not None:
        grad_bias_val = B_val_r.to(accum_dtype).sum(dim=[0, 2])
        grad_bias_train = B_train_r.to(accum_dtype).sum(dim=2)
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)
        if log_grad_norms:
            layer.bias.grad_train_norm = (grad_bias_train.to(accum_dtype) ** 2).sum(dim=1)
            layer.bias.grad_val_norm_sq = (grad_bias_val.to(accum_dtype) ** 2).sum()

    if log_grad_norms:
        layer.weight.grad_train_norm = weight_train_norm
        layer.weight.grad_val_norm_sq = weight_val_norm_sq


def _compute_conv2d_train_grad(
    layer: nn.Conv2d,
    A: Float[torch.Tensor, "batch in_channels height width"],
    B: Float[torch.Tensor, "batch out_channels out_height out_width"],
    val_batch_size: int,
) -> torch.Tensor:
    """Compute averaged training gradients for nn.Conv2d weight."""

    train_batch_size = A.size(0) - val_batch_size
    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    param_dtype = layer.weight.dtype
    if A_train.dtype != param_dtype:
        A_train = A_train.to(param_dtype)
    if B_train.dtype != param_dtype:
        B_train = B_train.to(param_dtype)

    unfold_params = dict(
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )

    A_u = F.unfold(A_train, **unfold_params)
    B_r = B_train.reshape(B_train.size(0), B_train.size(1), -1)

    grad_weight = torch.einsum('bik,bpk->pi', A_u, B_r)
    grad_weight = grad_weight.view_as(layer.weight)
    grad_weight /= train_batch_size

    return grad_weight



_supported_layers_dotprod = {
    nn.Linear: (_compute_linear_dot_product, _compute_linear_train_grad),
    nn.Embedding: (_compute_embedding_dot_product, _compute_embedding_train_grad),
    nn.LayerNorm: (_compute_layernorm_dot_product, _compute_layernorm_train_grad),
    nn.RMSNorm: (_compute_rmsnorm_dot_product, _compute_rmsnorm_train_grad),
    transformers.pytorch_utils.Conv1D: (_compute_Conv1D_dot_product, _compute_Conv1D_train_grad),
    nn.Conv2d: (_compute_conv2d_dot_product, _compute_conv2d_train_grad),
}
