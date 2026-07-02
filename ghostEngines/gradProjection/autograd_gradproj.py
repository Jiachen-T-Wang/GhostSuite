"""
Autograd hook utilities for gradient projection.
Handles forward/backward hooks for computing projected per-sample gradients.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any


def project_dense(A: torch.Tensor, B_out: torch.Tensor,
                  P_i: torch.Tensor, P_o: torch.Tensor) -> torch.Tensor:
    """Per-sample projected dense (Linear/Conv1D) gradient.

    With layer input ``A`` and output grad ``B``, the per-sample gradient is
    ``dL/dW = Σ_t B_t A_tᵀ``, projected without materializing it:
    ``P_o dL/dW P_iᵀ = Σ_t (P_o B_t)(P_i A_t)ᵀ``.

    Shared by the eager hook path and the decoupled in-graph Function so any
    scaling/masking fix applies to both at once.

    Args:
        A: input activations ``[B, ..., n_i]`` (middle dims flattened as tokens)
        B_out: output grads ``[B, ..., n_o]``
        P_i: ``[k_i, n_i]`` input projection
        P_o: ``[k_o, n_o]`` output projection

    Returns:
        ``[B, k_o, k_i]`` float32 per-sample projected gradients, scaled by the
        batch size (undoes the 1/B a mean-reduced loss puts on the output grads,
        matching a reduction='sum' naive per-sample gradient).
    """
    batch_size = A.shape[0]
    # The projection runs outside autocast, so A/B may carry the model dtype
    # (e.g. bf16) while P is float32: cast to the projection dtype so the matmuls
    # don't raise and the projection accumulates in full precision.
    A2 = A.reshape(batch_size, -1, A.shape[-1]).to(P_i.dtype)      # [B, T, n_i]
    B2 = B_out.reshape(batch_size, -1, B_out.shape[-1]).to(P_o.dtype)  # [B, T, n_o]
    A_proj = torch.matmul(A2, P_i.t())                             # [B, T, k_i]
    B_proj = torch.matmul(B2, P_o.t())                             # [B, T, k_o]
    gradG = torch.einsum('bti,btj->bij', B_proj, A_proj)           # [B, k_o, k_i]
    return (gradG * batch_size).to(torch.float32)


def project_embedding(idx: torch.Tensor, B_out: torch.Tensor,
                      P_i: torch.Tensor, P_o: torch.Tensor,
                      padding_idx: Optional[int] = None) -> torch.Tensor:
    """Per-sample projected nn.Embedding gradient (accumulated directly in projected space).

    For token index ``idx_t`` with output grad ``g_t``, accumulates the outer product
    ``(P_o g_t)(P_i[:, idx_t])ᵀ`` over tokens. Shared by the eager hook path and the
    decoupled in-graph Function.

    ``padding_idx``: the native embedding backward zeroes ``dL/dW[padding_idx]``, and
    all of that row's mass comes exactly from the pad-token positions — so masking
    those positions out of the accumulation reproduces the native behavior in
    projected space.

    Args:
        idx: token indices ``[B, T]`` (or ``[B, ...]``, flattened)
        B_out: output grads ``[B, ..., D]``
        P_i: ``[k_i, vocab]`` input projection
        P_o: ``[k_o, D]`` output projection
        padding_idx: the layer's ``padding_idx`` (or None)

    Returns:
        ``[B, k_o, k_i]`` float32, scaled by batch size (see :func:`project_dense`).
    """
    batch_size = idx.shape[0]
    idx_flat = idx.reshape(batch_size, -1)                                  # [B, T]
    grad_flat = B_out.reshape(batch_size, -1, B_out.shape[-1]).to(P_o.dtype)  # [B, T, D]
    if padding_idx is not None:
        grad_flat = grad_flat * (idx_flat != padding_idx).unsqueeze(-1).to(grad_flat.dtype)
    B_proj = torch.matmul(grad_flat, P_o.t())                               # [B, T, k_o]
    A_proj = P_i.t()[idx_flat]                                              # [B, T, k_i]
    gradG = torch.einsum('bto,bti->boi', B_proj, A_proj)                    # [B, k_o, k_i]
    return (gradG * batch_size).to(torch.float32)


def check_embedding_supported(module: nn.Embedding, layer_name: str) -> None:
    """Reject nn.Embedding options the projection math does not model (fail loud at attach).

    ``max_norm`` renormalizes weight rows in-place during forward, ``scale_grad_by_freq``
    rescales gradient rows by token frequency, and ``sparse`` yields sparse grads — none
    of which the projected per-sample gradient accounts for, so the capture would silently
    diverge from the true gradient. ``padding_idx`` IS supported (masked in
    :func:`project_embedding`).
    """
    unsupported = []
    if module.max_norm is not None:
        unsupported.append(f"max_norm={module.max_norm}")
    if module.scale_grad_by_freq:
        unsupported.append("scale_grad_by_freq=True")
    if module.sparse:
        unsupported.append("sparse=True")
    if unsupported:
        raise NotImplementedError(
            f"Gradient projection does not support nn.Embedding option(s) "
            f"[{', '.join(unsupported)}] on layer '{layer_name}': the captured projected "
            "gradient would silently diverge from the true per-sample gradient. Use a "
            "default-configured embedding or exclude this layer from proj_layers.")


class GradProjHooks:
    """
    Container for forward and backward hooks used in gradient projection.
    Stores projection matrices and provides hook functions.
    """

    def __init__(self, P_i: torch.Tensor, P_o: torch.Tensor,
                 layer_name: str, layer_type: str):
        """
        Initialize hooks with projection matrices.

        Args:
            P_i: Input projection matrix [k_i, n_i]
            P_o: Output projection matrix [k_o, n_o]
            layer_name: Name of the layer for debugging
            layer_type: Type of layer (Linear, Conv1D, Embedding, etc.)
        """
        self.P_i = P_i
        self.P_o = P_o
        self.layer_name = layer_name
        self.layer_type = layer_type
        self._module = None
        self._handle_forward = None
        self._handle_backward = None

    def forward_hook_store_inputs(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...],
                                 output: torch.Tensor) -> None:
        """
        Forward hook to store input activations.

        Args:
            module: The layer being hooked
            inputs: Input tuple (typically contains single tensor)
            output: Output from the layer (unused)
        """
        # Skip no-grad forwards (e.g. an eval pass between a train forward and its
        # backward): capturing here would overwrite the train activations and pair
        # the eval input with the train output grads — silently wrong projections.
        if not torch.is_grad_enabled():
            return
        # Store detached input for later use in backward.
        # Limitation: a single cache slot per module, so a projected module
        # invoked more than once per forward pass (e.g. a shared/tied module) only
        # retains the last call's activations. The engine targets distinct
        # Linear/Embedding/Conv1D layers, which are each called once per step.
        module._ghost_A_raw = inputs[0].detach()

    def backward_hook_compute_proj(self, module: nn.Module, grad_input: Tuple[Optional[torch.Tensor], ...],
                                  grad_output: Tuple[torch.Tensor, ...]) -> None:
        """
        Backward hook to compute projected gradients.

        Args:
            module: The layer being hooked
            grad_input: Gradients w.r.t. inputs (unused)
            grad_output: Gradients w.r.t. outputs
        """
        # Get cached activations
        A_raw = getattr(module, '_ghost_A_raw', None)
        if A_raw is None:
            raise RuntimeError(f'Missing cached activations for GradProjection in {self.layer_name}')

        # Get output gradients
        B_out = grad_output[0]
        if B_out is None:
            # No gradient flowing through this layer
            module._ghost_grad_proj = None
            delattr(module, '_ghost_A_raw')
            return

        B_out = B_out.detach()

        # Handle different layer types
        if self.layer_type == 'Embedding':
            # For embedding, we need special handling
            self._compute_embedding_proj(module, A_raw, B_out)
        else:
            # For Linear/Conv1D layers
            self._compute_dense_proj(module, A_raw, B_out)

        # Clean up cached activations
        delattr(module, '_ghost_A_raw')

    def _compute_dense_proj(self, module: nn.Module, A_raw: torch.Tensor,
                           B_out: torch.Tensor) -> None:
        """
        Compute projected gradients for dense layers (Linear, Conv1D).

        Delegates the math to the shared :func:`project_dense` kernel (also used by
        the decoupled in-graph path). The kernel is defined purely in terms of
        (input activations, output grads), so the same path is correct for
        nn.Linear and transformers Conv1D alike once get_layer_dimensions reports
        the true (n_i = in_features, n_o = out_features) for each.
        """
        module._ghost_grad_proj = project_dense(A_raw, B_out, self.P_i, self.P_o)

    def _compute_embedding_proj(self, module: nn.Module, indices: torch.Tensor,
                               grad_output: torch.Tensor) -> None:
        """
        Compute projected gradients for embedding layers.

        Delegates to the shared :func:`project_embedding` kernel (also used by the
        decoupled in-graph path), which accumulates directly in projected space and
        masks the layer's ``padding_idx`` positions to match the native backward.
        """
        module._ghost_grad_proj = project_embedding(
            indices, grad_output, self.P_i, self.P_o, padding_idx=module.padding_idx)

    def attach(self, module: nn.Module) -> None:
        """Attach hooks to the module."""
        self._module = module
        self._handle_forward = module.register_forward_hook(self.forward_hook_store_inputs)
        self._handle_backward = module.register_full_backward_hook(self.backward_hook_compute_proj)

    def detach(self) -> None:
        """Remove hooks from the module and clear its capture scratch."""
        if self._handle_forward is not None:
            self._handle_forward.remove()
            self._handle_forward = None
        if self._handle_backward is not None:
            self._handle_backward.remove()
            self._handle_backward = None
        # Drop any stray capture (e.g. a forward that was never followed by a
        # backward) so detached modules cannot leak activations.
        if self._module is not None:
            for attr in ('_ghost_A_raw', '_ghost_grad_proj'):
                if hasattr(self._module, attr):
                    delattr(self._module, attr)
            self._module = None


def create_projection_hooks(module: nn.Module, layer_name: str,
                          P_i: torch.Tensor, P_o: torch.Tensor) -> GradProjHooks:
    """
    Create and return hooks for a specific layer.

    Args:
        module: The layer to hook
        layer_name: Name of the layer
        P_i: Input projection matrix
        P_o: Output projection matrix

    Returns:
        GradProjHooks instance (not yet attached)
    """
    # Determine layer type
    layer_type = module.__class__.__name__

    # Handle special cases
    if layer_type == 'Conv1D':
        # Transformers Conv1D is like Linear with transposed weight
        layer_type = 'Linear'
    elif isinstance(module, nn.Conv1d):
        # Conv1d requires proper unfolding - not yet implemented
        raise NotImplementedError(
            f"Conv1d layers are not yet supported for gradient projection. "
            f"Layer '{layer_name}' is a Conv1d layer. "
            f"Proper im2col unfolding is required for correct gradient computation. "
            f"Please exclude Conv1d layers from proj_layers pattern."
        )
    elif isinstance(module, nn.Conv2d):
        # Conv2d requires proper unfolding - not yet implemented
        raise NotImplementedError(
            f"Conv2d layers are not yet supported for gradient projection. "
            f"Layer '{layer_name}' is a Conv2d layer. "
            f"Proper im2col unfolding is required for correct gradient computation. "
            f"Please exclude Conv2d layers from proj_layers pattern or set include_conv2d=False."
        )
    elif isinstance(module, nn.Linear):
        layer_type = 'Linear'
    elif isinstance(module, nn.Embedding):
        check_embedding_supported(module, layer_name)
        layer_type = 'Embedding'
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

    return GradProjHooks(P_i, P_o, layer_name, layer_type)
