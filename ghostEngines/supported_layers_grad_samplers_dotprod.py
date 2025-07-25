import torch
import transformers.pytorch_utils
from torch import nn
from torch.functional import F
from transformers.models.t5.modeling_t5 import T5LayerNorm
import time

def _should_use_ghost_computation(layer: nn.Module, A: torch.Tensor, B: torch.Tensor, conv: bool = False):
    """
    Determines whether to use the efficient "ghost" computation method based
    on the dimensions of the activation and backpropagation tensors.

    This check is based on the heuristic described in the literature, comparing
    the computational cost of materializing the full gradient versus using the
    ghost computation trick.

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
        T = torch.prod(torch.tensor(A.shape[1:-1])).item() if A.dim() > 2 else 1
        d = A.shape[-1]
        p = B.shape[-1]
    else:
        # For convolutional layers (after unfolding)
        T = A.shape[-1]
        d = A.shape[1]
        p = B.shape[1]

    # The total number of parameters in the weight matrix
    num_weight_params = layer.weight.numel()

    # Test: 2*T^2 <= d*p  (or num_weight_params)
    layer.use_ghost_computation = bool(2 * T**2 <= num_weight_params)


def _create_or_accumulate_train_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
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


# Linear Layer Implementation
# #############################################################################

def _compute_linear_dot_product(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """Computes the gradient dot-product for an nn.Linear layer."""

    if A is None:
        raise ValueError("Input activations A cannot be None.")
    if B is None:
        raise ValueError("Output gradients B cannot be None.")

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    # start_time = time.time()

    # Cast A and B to bfloat16 for efficiency.
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    _should_use_ghost_computation(layer, A, B)

    # --- Weight dot product ---
    if layer.use_ghost_computation:

        # mm_start_time = time.time()

        val_bs = val_batch_size
        B_total, T, d = A.shape
        p             = B.shape[-1]
        train_bs      = B_total - val_bs

        # ------------------------------------------------------------------
        # 1.  Flatten (batch, token) → 'token index' so that each example‑
        #     token pair becomes one row.  This lets us use a single GEMM.
        # ------------------------------------------------------------------
        A_train = A[:train_bs].reshape(-1, d).contiguous()  # [(train_bs·T), d]
        A_val   = A[train_bs:].reshape(-1, d).contiguous()  # [(val_bs  ·T), d]
        B_train = B[:train_bs].reshape(-1, p).contiguous()  # [(train_bs·T), p]
        B_val   = B[train_bs:].reshape(-1, p).contiguous()  # [(val_bs  ·T), p]

        # Optionally cast once (assumes you run AMP/bfloat16 in the forward)
        A_train = A_train.to(torch.bfloat16)
        A_val   = A_val  .to(torch.bfloat16)
        B_train = B_train.to(torch.bfloat16)
        B_val   = B_val  .to(torch.bfloat16)

        # ------------------------------------------------------------------
        # 2.  Two GEMMs →  (train_tokens × val_tokens)  matrices
        # ------------------------------------------------------------------
        #    Shapes:
        #       A_train : (N_tr_tok, d)  @  A_val.T : (d, N_val_tok)  → (N_tr_tok, N_val_tok)
        #       B_train : (N_tr_tok, p)  @  B_val.T : (p, N_val_tok)  → (N_tr_tok, N_val_tok)
        a_dot = torch.matmul(A_train, A_val.T)        # 𝐀   (bf16)
        b_dot = torch.matmul(B_train, B_val.T)        # 𝐁   (bf16)

        # ------------------------------------------------------------------
        # 3.  Hadamard product + reduction over *all* validation tokens
        # ------------------------------------------------------------------
        #    (N_tr_tok, N_val_tok)  →  (N_tr_tok,)  (float32 accumulator)
        token_contrib = (a_dot * b_dot).sum(dim=1, dtype=torch.float32)

        # ------------------------------------------------------------------
        # 4.  Fold back token dimension → one scalar per training example
        # ------------------------------------------------------------------
        layer.weight.grad_dot_prod = token_contrib.reshape(train_bs, T).sum(dim=1)


        # A_train, A_val = A[:-val_batch_size].contiguous(), A[-val_batch_size:].contiguous()
        # B_train, B_val = B[:-val_batch_size].contiguous(), B[-val_batch_size:].contiguous()
        # A_val_sum = torch.sum(A_val, dim=0)
        # B_val_sum = torch.sum(B_val, dim=0)
        # AA = torch.matmul(A_val_sum.unsqueeze(0), A_train.transpose(-1, -2))
        # BB = torch.matmul(B_val_sum.unsqueeze(0), B_train.transpose(-1, -2))
        # layer.weight.grad_dot_prod = torch.sum(AA * BB, dim=[1, 2])

        # torch.cuda.synchronize()  # Ensure all operations are complete
        # print(f"Debug: Prepare Dotprod time for lm_head: {(time.time() - mm_start_time)*1000:.4f}ms")

    else:
        grad_train = torch.einsum('b...d,b...p->bpd', A_train, B_train)
        grad_val = torch.einsum('...d,...p->dp', torch.sum(A_val, dim=0), torch.sum(B_val, dim=0))
        layer.weight.grad_dot_prod = torch.einsum('dp,bpd->b', grad_val, grad_train)

    # --- Bias dot product ---
    if layer.bias is not None:
        sum_dims_val = list(range(B_val.dim() - 1))
        grad_bias_val = B_val.sum(dim=sum_dims_val)
        sum_dims_train = list(range(1, B_train.dim() - 1))
        grad_bias_train = B_train.sum(dim=sum_dims_train) if B_train.dim() > 2 else B_train
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)

    # torch.cuda.synchronize()  # Ensure all operations are complete
    # end_time = time.time()
    # print(f"Debug: Dot product computation time for lm_head: {(end_time - start_time) * 1000:.4f} ms")
    # print(f"Debug: Check grad dot product value for Linear layer: {layer.weight.grad_dot_prod}")



def _compute_linear_train_grad(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """
    Computes the training gradient for an nn.Linear layer's weight.
    This version always computes the average gradient to match PyTorch's default behavior.
    """
    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return None # Return None if there's nothing to compute

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Compute the SUM of gradients over the training batch
    grad_weight = torch.einsum('b...p,b...d->pd', B_train, A_train)

    # Always divide by the number of training samples to get the AVERAGE gradient
    grad_weight /= train_batch_size
    
    return grad_weight




# Embedding Layer Implementation
# #############################################################################

def _compute_embedding_dot_product(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """Computes the gradient dot-product for an nn.Embedding layer."""

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    # Cast B to bfloat16 for efficiency.
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Ensure the index tensors are of integer type (long) before using them.
    A_train_long = A_train.long()
    A_val_long = A_val.long()

    vocab_size, d_f = layer.weight.shape
    grad_val = torch.zeros((vocab_size, d_f), dtype=torch.bfloat16, device=B_val.device)
    grad_val.index_add_(
        0,
        A_val_long.reshape(-1),                     # indices  [val_batch * seq]
        B_val.reshape(-1, d_f)                      # vectors  [val_batch * seq, d_f]
    )

    dot_products = (B_train * grad_val[A_train_long]).float().sum(dim=[1, 2])
    layer.weight.grad_dot_prod = dot_products


def _compute_embedding_train_grad(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """
    Computes the training gradient for an nn.Embedding layer.
    This version always computes the average gradient.
    """

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return None

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    A_train_long = A_train.long()

    grad_weight = torch.zeros_like(layer.weight)
    grad_weight.index_add_(0, A_train_long.reshape(-1), B_train.reshape(-1, B_train.shape[-1]))

    # Always divide by the number of training samples to get the AVERAGE gradient
    grad_weight /= train_batch_size
    
    return grad_weight



# LayerNorm Layer Implementation
# #############################################################################

def _compute_layernorm_dot_product(layer: nn.LayerNorm, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """Computes the gradient dot-product for an nn.LayerNorm layer."""

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    # Cast A and B to bfloat16 for efficiency.
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)
    
    # --- Weight dot product ---
    # The gradient for the weight is B * normalized_A
    normalized_A_train = F.layer_norm(A_train, layer.normalized_shape, eps=layer.eps)
    normalized_A_val = F.layer_norm(A_val, layer.normalized_shape, eps=layer.eps)

    grad_weight_train = B_train * normalized_A_train
    grad_weight_val = B_val * normalized_A_val

    # Sum over all dimensions except batch to get per-sample grad
    sum_dims = list(range(1, grad_weight_train.dim()))
    per_sample_grad_weight = grad_weight_train.sum(dim=sum_dims)
    total_grad_weight_val = grad_weight_val.sum(dim=list(range(grad_weight_val.dim())))
    
    layer.weight.grad_dot_prod = per_sample_grad_weight * total_grad_weight_val

    # --- Bias dot product ---
    if layer.bias is not None:
        # The gradient for the bias is just B
        sum_dims = list(range(1, B_train.dim()))
        per_sample_grad_bias = B_train.sum(dim=sum_dims)
        total_grad_bias_val = B_val.sum(dim=list(range(B_val.dim())))
        layer.bias.grad_dot_prod = per_sample_grad_bias * total_grad_bias_val


def _compute_layernorm_train_grad(
    layer: nn.LayerNorm,
    A: torch.Tensor,
    B: torch.Tensor,
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


def _compute_Conv1D_dot_product(
    layer: nn.Linear,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int
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

    # Cast A and B to bfloat16 for efficiency.
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size

    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Always check if ghost computation should be used.
    _should_use_ghost_computation(layer, A, B)

    if A.dim() > 3:
        raise ValueError("Currently we only expect 3D input tensors (batch, seq_len, features). Extending this to 4D or higher requires additional handling.")

    # --- Compute weight gradient dot product ---
    if layer.use_ghost_computation:
        # Sum over the validation batch dimension to get the validation gradient components
        A_val_sum = torch.sum(A_val, dim=0)
        B_val_sum = torch.sum(B_val, dim=0)

        # The dot product of gradients G1 and G2 is trace((A1 @ A2.T) * (B1 @ B2.T))
        # We use torch.matmul which correctly broadcasts the validation tensors across the training batch dimension.

        # start_time = time.time()

        # A_val_sum: [t, d] -> unsqueezed to [1, t, d] for broadcasting
        # A_train.transpose(-1, -2): [b, d, t]
        # Result AA: [b, t, t]
        AA = torch.matmul(A_val_sum.unsqueeze(0), A_train.transpose(-1, -2))

        # B_val_sum: [t, p] -> unsqueezed to [1, t, p]
        # B_train.transpose(-1, -2): [b, p, t]
        # Result BB: [b, t, t]
        BB = torch.matmul(B_val_sum.unsqueeze(0), B_train.transpose(-1, -2))

        # Element-wise product and sum over the two `t` dimensions to get the trace
        # The result is a tensor of shape [b], with one value per training sample.
        # layer.weight.grad_dot_prod = torch.sum(AA * BB, dim=[1, 2])
        layer.weight.grad_dot_prod = torch.sum((AA * BB).float(), dim=[1, 2])

        # torch.cuda.synchronize()  # Ensure all operations are complete
        # print(f"Prepare Dotprod time for {layer.name}: {(time.time() - start_time)*1000:.4f}ms")
        # print(f"Debug: Check grad dot product value for Conv1D layer: {layer.weight.grad_dot_prod}")

    else:
        # Materialize gradients to compute the dot product
        grad_train = torch.einsum('b...d, b...p->bpd', A_train, B_train).detach()
        grad_val = torch.einsum('...d, ...p->pd', torch.sum(A_val, dim=0), torch.sum(B_val, dim=0)).detach()
        layer.weight.grad_dot_prod = torch.einsum('pd,bpd->b', grad_val, grad_train)

    # --- Compute bias gradient dot product ---
    if layer.bias is not None:

        # For a tensor B of shape [batch, seq_len, features], the per-sample
        # bias grad is B.sum(dim=1). The total validation grad is B_val.sum(dim=[0, 1]).
        if B.dim() >= 2:
            # Sum over all dimensions except the last one (the feature dimension)
            sum_dims_val = list(range(B_val.dim() - 1))
            grad_bias_val = B_val.sum(dim=sum_dims_val)

            # For the training batch, we want per-sample gradients, so we only
            # sum over the sequence/spatial dimensions, not the batch dimension.
            sum_dims_train = list(range(1, B_train.dim() - 1))
            grad_bias_train = B_train.sum(dim=sum_dims_train)
        else: # Should not happen, but as a safeguard
            grad_bias_train = B_train
            grad_bias_val = B_val.sum(dim=0)

        # The dot product is a simple einsum between the total validation bias grad
        # and the per-sample training bias grads.
        # grad_bias_val shape: [features]
        # grad_bias_train shape: [batch, features]
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)


def _compute_Conv1D_train_grad(
    layer: transformers.pytorch_utils.Conv1D,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    average_grad: bool = False
) -> torch.Tensor:
    """
    Computes the sum or average of gradients across the training data for a
    Conv1D layer's weight.

    Args:
        layer: The Conv1D layer.
        A: The input activations from the combined (train + val) batch.
        B: The output gradients from the combined (train + val) batch.
        val_batch_size: The number of samples in the validation batch.
        average_grad: If True, computes the average gradient.
                      If False (default), computes the summed gradient.

    Returns:
        The summed or averaged gradient for the layer's weight parameter.
    """

    # Determine training batch size from the total size and validation size.
    train_batch_size = A.size(0) - val_batch_size

    # Isolate the activations and backpropagated gradients for the training data.
    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Compute the summed gradient for the training batch.
    grad_weight = torch.einsum('b...d,b...p->dp', A_train, B_train)

    grad_weight /= train_batch_size

    # # If requested, compute the average gradient instead of the sum.
    # if average_grad:
    #     # Avoid division by zero if for some reason the training batch size is 0.
    #     if train_batch_size > 0:
    #         # grad_weight /= train_batch_size

    return grad_weight


_supported_layers_dotprod = {
    nn.Linear: (_compute_linear_dot_product, _compute_linear_train_grad),
    nn.Embedding: (_compute_embedding_dot_product, _compute_embedding_train_grad),
    nn.LayerNorm: (_compute_layernorm_dot_product, _compute_layernorm_train_grad),
    transformers.pytorch_utils.Conv1D: (_compute_Conv1D_dot_product, _compute_Conv1D_train_grad),
}


