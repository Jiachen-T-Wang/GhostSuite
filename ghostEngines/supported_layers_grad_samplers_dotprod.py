import torch
import transformers.pytorch_utils
from torch import nn
import torch.nn.functional as F


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

def _compute_linear_dot_product(
    layer: nn.Linear,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False
):
    # ... [Input checks and Setup remain the same] ...
    
    # Detach and Cast
    A = A.detach()
    B = B.detach()
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)

    total_bs = A.size(0)
    train_bs = total_bs - val_batch_size
    
    # Setup Dimensions
    d = A_bf16.size(-1)
    p = B_bf16.size(-1)
    A_flat = A_bf16.reshape(-1, d)
    B_flat = B_bf16.reshape(-1, p)
    
    tokens_per_sample = A_bf16.numel() // (total_bs * d)
    split_idx = train_bs * tokens_per_sample

    A_train = A_flat[:split_idx]  # [N_train, d]
    A_val = A_flat[split_idx:]    # [N_val, d]
    B_train = B_flat[:split_idx]  # [N_train, p]
    B_val = B_flat[split_idx:]    # [N_val, p]

    # Pre-declare variables for logging reuse
    grad_val_for_norm = None
    
    # Decide whether to use ghost computation
    _should_use_ghost_computation(layer, A, B)

    if layer.use_ghost_computation:
        # --- ghost computation with associativity trick ---
        
        # 1. Compute Val Gradient Summary (d x p)
        # Note: This is effectively Grad_Val.T
        val_interaction = torch.matmul(A_val.T, B_val)
        
        # Save for logging (Transposed to match standard shape [d, p])
        if log_grad_norms:
             grad_val_for_norm = val_interaction.T

        # 2. Project Training Gradients (N_train x d)
        B_projected = torch.matmul(B_train, val_interaction.T)

        # 3. Element-wise Interaction
        token_contrib = (A_train * B_projected).sum(dim=1)

        # 4. Fold back and reduce
        layer.weight.grad_dot_prod = token_contrib.view(train_bs, tokens_per_sample).sum(dim=1)
        
    else:
        # --- materialize gradients ---
        
        # 1. Compute Val Gradient (d x p)
        # Correctly contracts batch/time dims together
        grad_val = torch.einsum('nd,np->dp', A_val, B_val)
        
        if log_grad_norms:
            grad_val_for_norm = grad_val

        # 2. Compute Per-Sample Training Gradients (Batch x p x d) or (Batch x d x p)
        # To match your einsum 'bpd' later, we need PxD. 
        # But for 'grad_train' calculation, let's stick to your working logic.
        
        # Reshape for sum-over-T contraction
        A_train_3d = A_train.view(train_bs, tokens_per_sample, d)
        B_train_3d = B_train.view(train_bs, tokens_per_sample, p)
        
        # Contract T (Ghost logic for 'else' branch)
        grad_train = torch.einsum('btd,btp->bpd', A_train_3d, B_train_3d)

        # 3. Frobenius Inner Product
        layer.weight.grad_dot_prod = torch.einsum('dp,bpd->b', grad_val, grad_train)

    if log_grad_norms:
        raise NotImplementedError("Gradient norm logging not implemented for linear layer")


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
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False
):
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

    if log_grad_norms:
        # Compute per-sample train grad norm using unique tokens per sample
        train_norms = []
        for sample_tokens, sample_B in zip(A_train_long, B_train):
            unique_tok, inverse = torch.unique(sample_tokens, return_inverse=True)
            agg = torch.zeros((unique_tok.numel(), sample_B.size(-1)), device=sample_B.device, dtype=sample_B.dtype)
            agg.index_add_(0, inverse, sample_B)
            train_norms.append((agg.float() ** 2).sum())
        layer.weight.grad_train_norm = torch.stack(train_norms)

        # Validation aggregated gradient norm
        layer.weight.grad_val_norm_sq = (grad_val.float() ** 2).sum()


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
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False
):
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
        'bf,f->b', per_sample_grad_weight.float(), total_grad_weight_val.float()
    )
    weight_train_norm = None
    weight_val_norm_sq = None
    if log_grad_norms:
        weight_train_norm = (per_sample_grad_weight.float() ** 2).sum(dim=1)
        weight_val_norm_sq = (total_grad_weight_val.float() ** 2).sum()

    # --- Bias dot product ---
    if layer.bias is not None:
        # Bias gradient is B; reduce non-feature dims for train → [B_train, F]
        if B_train.dim() >= 2:
            sum_dims_train = list(range(1, B_train.dim() - 1))
            per_sample_grad_bias = B_train.sum(dim=sum_dims_train) if sum_dims_train else B_train
        else:
            per_sample_grad_bias = B_train

        # Validation aggregate over batch and token dims only → [F]
        sum_dims_val = list(range(B_val.dim() - 1))
        total_grad_bias_val = B_val.sum(dim=sum_dims_val)

        layer.bias.grad_dot_prod = torch.einsum(
            'bf,f->b', per_sample_grad_bias.float(), total_grad_bias_val.float()
        )
        bias_train_norm = None
        bias_val_norm_sq = None
        if log_grad_norms:
            bias_train_norm = (per_sample_grad_bias.float() ** 2).sum(dim=1)
            bias_val_norm_sq = (total_grad_bias_val.float() ** 2).sum()
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


def _compute_rmsnorm_dot_product(
    layer: nn.RMSNorm,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False,
):
    """Compute gradient dot-product for nn.RMSNorm (weight-only)."""
    """
    A: [batch, seq, dim] (normalized by RMSNorm)
    B: [batch, seq, dim] (backpropagated gradients)
    """
    A = A.detach()
    B = B.detach()

    train_batch_size = A.size(0) - val_batch_size

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    eps = getattr(layer, "eps", 1e-5)
    rms_train = torch.sqrt((A_train.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    rms_val = torch.sqrt((A_val.float() ** 2).mean(dim=-1, keepdim=True) + eps)

    norm_A_train = (A_train.float() / rms_train)
    norm_A_val = (A_val.float() / rms_val)

    grad_weight_train = B_train * norm_A_train
    grad_weight_val = B_val * norm_A_val

    sum_dims_train = list(range(1, grad_weight_train.dim() - 1))
    per_sample_grad_weight = grad_weight_train.sum(dim=sum_dims_train) if sum_dims_train else grad_weight_train

    sum_dims_val = list(range(grad_weight_val.dim() - 1))
    total_grad_weight_val = grad_weight_val.sum(dim=sum_dims_val)

    layer.weight.grad_dot_prod = torch.einsum(
        "bf,f->b", per_sample_grad_weight.float(), total_grad_weight_val.float()
    )

    if log_grad_norms:
        layer.weight.grad_train_norm = (per_sample_grad_weight.float() ** 2).sum(dim=1)
        layer.weight.grad_val_norm_sq = (total_grad_weight_val.float() ** 2).sum()


def _compute_rmsnorm_train_grad(
    layer: nn.RMSNorm,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
):
    """Compute and apply averaged training gradient for nn.RMSNorm weight."""
    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute gradients, check batch sizes.")

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    eps = getattr(layer, "eps", 1e-6)
    rms_train = torch.sqrt((A_train.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    norm_A_train = (A_train.float() / rms_train).to(B_train.dtype)

    grad_weight = (B_train * norm_A_train).sum(dim=list(range(A_train.dim() - 1)))
    grad_weight = grad_weight.to(layer.weight.dtype)
    grad_weight /= train_batch_size

    return grad_weight


def _compute_Conv1D_dot_product(
    layer: nn.Linear,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False
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

    weight_train_norm = None
    weight_val_norm_sq = None

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

        if log_grad_norms:
            grad_train = torch.einsum('btd,btp->bpd', A_train, B_train)
            weight_train_norm = (grad_train.float() ** 2).sum(dim=[1, 2])
            grad_val = torch.einsum('td,tp->pd', A_val_sum, B_val_sum)
            weight_val_norm_sq = (grad_val.float() ** 2).sum()

        # torch.cuda.synchronize()  # Ensure all operations are complete
        # print(f"Prepare Dotprod time for {layer.name}: {(time.time() - start_time)*1000:.4f}ms")
        # print(f"Debug: Check grad dot product value for Conv1D layer: {layer.weight.grad_dot_prod}")

    else:
        # Materialize gradients to compute the dot product
        grad_train = torch.einsum('b...d, b...p->bpd', A_train, B_train).detach()
        grad_val = torch.einsum('...d, ...p->pd', torch.sum(A_val, dim=0), torch.sum(B_val, dim=0)).detach()
        layer.weight.grad_dot_prod = torch.einsum('pd,bpd->b', grad_val, grad_train)
        if log_grad_norms:
            weight_train_norm = (grad_train.float() ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.float() ** 2).sum()

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

        if log_grad_norms:
            layer.bias.grad_train_norm = (grad_bias_train.float() ** 2).sum(dim=1)
            layer.bias.grad_val_norm_sq = (grad_bias_val.float() ** 2).sum()

    if log_grad_norms:
        layer.weight.grad_train_norm = weight_train_norm
        layer.weight.grad_val_norm_sq = weight_val_norm_sq


def _compute_Conv1D_train_grad(
    layer: transformers.pytorch_utils.Conv1D,
    A: torch.Tensor,
    B: torch.Tensor,
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
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int,
    log_grad_norms: bool = False,
) -> None:
    """Compute gradient dot-products for nn.Conv2d."""

    A = A.detach()
    B = B.detach()

    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute dot product")

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)

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

    if layer.use_ghost_computation:
        A_val_sum = torch.sum(A_val_u, dim=0)
        B_val_sum = torch.sum(B_val_r, dim=0)
        AA = torch.matmul(A_val_sum.unsqueeze(0), A_train_u.transpose(1, 2))
        BB = torch.matmul(B_val_sum.unsqueeze(0), B_train_r.transpose(1, 2))
        layer.weight.grad_dot_prod = torch.sum((AA * BB).float(), dim=[1, 2])
        if log_grad_norms:
            grad_train = torch.einsum('bik,bpk->bpi', A_train_u, B_train_r)
            weight_train_norm = (grad_train.float() ** 2).sum(dim=[1, 2])
            grad_val = torch.einsum('ik,pk->pi', A_val_sum, B_val_sum)
            weight_val_norm_sq = (grad_val.float() ** 2).sum()
    else:
        grad_train = torch.einsum('bik,bpk->bpi', A_train_u, B_train_r)
        grad_val = torch.einsum('ik,pk->pi', A_val_u.sum(dim=0), B_val_r.sum(dim=0))
        layer.weight.grad_dot_prod = torch.einsum('pi,bpi->b', grad_val, grad_train)
        if log_grad_norms:
            weight_train_norm = (grad_train.float() ** 2).sum(dim=[1, 2])
            weight_val_norm_sq = (grad_val.float() ** 2).sum()

    if layer.bias is not None:
        grad_bias_val = B_val_r.sum(dim=[0, 2])
        grad_bias_train = B_train_r.sum(dim=2)
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)
        if log_grad_norms:
            layer.bias.grad_train_norm = (grad_bias_train.float() ** 2).sum(dim=1)
            layer.bias.grad_val_norm_sq = (grad_bias_val.float() ** 2).sum()

    if log_grad_norms:
        layer.weight.grad_train_norm = weight_train_norm
        layer.weight.grad_val_norm_sq = weight_val_norm_sq


def _compute_conv2d_train_grad(
    layer: nn.Conv2d,
    A: torch.Tensor,
    B: torch.Tensor,
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
