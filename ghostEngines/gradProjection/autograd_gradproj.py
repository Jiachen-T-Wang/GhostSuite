"""
Autograd hook utilities for gradient projection.
Handles forward/backward hooks for computing projected per-sample gradients.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten middle dimensions to get [batch, tokens, features] shape.
    
    Args:
        x: Input tensor of shape [B, ...] or [B, ..., D]
        
    Returns:
        Tensor of shape [B, T, D] where T is the product of middle dimensions
    """
    if x.dim() <= 2:
        # [B, D] -> [B, 1, D]
        return x.unsqueeze(1)
    
    B = x.shape[0]
    *mid, D = x.shape[1:]
    
    if len(mid) == 0:
        # Already [B, D]
        return x.unsqueeze(1)
    
    # Compute total tokens
    T = 1
    for dim in mid:
        T *= dim
    
    return x.reshape(B, T, D)


def _extract_batch_size(x: torch.Tensor) -> int:
    """Extract batch size from tensor, handling different input formats."""
    if x.dim() == 0:
        return 1
    return x.shape[0]


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
        # Store detached input for later use in backward
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
        
        The gradient for weight W is: dL/dW = sum_t B_t @ A_t^T
        We project this as: P_o @ dL/dW @ P_i^T = sum_t (P_o @ B_t) @ (P_i @ A_t)^T
        """
        # Flatten to [B, T, D] format
        A = _flatten_tokens(A_raw)  # [B, T, n_i]
        B = _flatten_tokens(B_out)  # [B, T, n_o]
        
        batch_size = A.shape[0]
        
        # Project activations and gradients
        # A_proj: [B, T, n_i] @ [n_i, k_i] -> [B, T, k_i]
        A_proj = torch.matmul(A, self.P_i.t())
        
        # B_proj: [B, T, n_o] @ [n_o, k_o] -> [B, T, k_o]  
        B_proj = torch.matmul(B, self.P_o.t())
        
        # Compute per-sample projected gradients
        # Align with naive reference: [B, k_o, k_i]
        # Note: grad_output from CrossEntropyLoss(mean) carries a 1/B factor;
        # multiply by batch_size to match reduction='sum' naive computation.
        gradG = torch.einsum('bti,btj->bij', B_proj, A_proj)
        gradG = gradG * batch_size
        
        # Store in float32 for precision
        module._ghost_grad_proj = gradG.to(torch.float32)
        
    def _compute_embedding_proj(self, module: nn.Module, indices: torch.Tensor,
                               grad_output: torch.Tensor) -> None:
        """
        Compute projected gradients for embedding layers.
        
        For embeddings, the gradient is sparse - only touched indices get gradients.
        We accumulate the full gradient then project it.
        """
        # indices: [B, T] or [B, ..., T]
        # grad_output: [B, T, embedding_dim] or [B, ..., T, embedding_dim]
        
        # Flatten inputs
        if indices.dim() > 2:
            batch_size = indices.shape[0]
            indices_flat = indices.reshape(batch_size, -1)  # [B, T_total]
            grad_flat = grad_output.reshape(batch_size, -1, grad_output.shape[-1])  # [B, T_total, D]
        else:
            indices_flat = indices  # [B, T]
            grad_flat = grad_output  # [B, T, D]
            batch_size = indices.shape[0]
            
        vocab_size = module.num_embeddings
        embed_dim = module.embedding_dim
        
        # Compute per-sample gradients
        per_sample_grads = []
        
        for b in range(batch_size):
            # Initialize gradient accumulator
            grad_weight = torch.zeros(vocab_size, embed_dim, 
                                     dtype=grad_flat.dtype, device=grad_flat.device)
            
            # Accumulate gradients for this sample
            idx_b = indices_flat[b]  # [T]
            grad_b = grad_flat[b]    # [T, D]
            
            # Use index_add to accumulate gradients
            grad_weight.index_add_(0, idx_b, grad_b)
            
            # Project the gradient
            # grad_weight: [vocab_size, embed_dim]
            # P_i: [k_i, vocab_size], P_o: [k_o, embed_dim]
            # Result: [k_o, k_i]
            grad_proj = self.P_o @ grad_weight.t() @ self.P_i.t()
            per_sample_grads.append(grad_proj)
            
        # Stack all per-sample gradients: [B, k_o, k_i]
        gradG = torch.stack(per_sample_grads, dim=0)
        module._ghost_grad_proj = gradG.to(torch.float32)
        
    def attach(self, module: nn.Module) -> None:
        """Attach hooks to the module."""
        self._handle_forward = module.register_forward_hook(self.forward_hook_store_inputs)
        self._handle_backward = module.register_full_backward_hook(self.backward_hook_compute_proj)
        
    def detach(self) -> None:
        """Remove hooks from the module."""
        if self._handle_forward is not None:
            self._handle_forward.remove()
            self._handle_forward = None
        if self._handle_backward is not None:
            self._handle_backward.remove()
            self._handle_backward = None


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
        layer_type = 'Conv1d'
    elif isinstance(module, nn.Linear):
        layer_type = 'Linear'
    elif isinstance(module, nn.Embedding):
        layer_type = 'Embedding'
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    
    return GradProjHooks(P_i, P_o, layer_name, layer_type)
