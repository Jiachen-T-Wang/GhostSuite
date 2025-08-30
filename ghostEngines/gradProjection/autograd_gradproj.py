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
        
        Special handling for Conv1D: The transformers Conv1D layer has transposed weight,
        so we need to swap the projections accordingly.
        """
        # Check if this is a Conv1D layer from transformers
        is_conv1d = (module.__class__.__name__ == 'Conv1D')
        
        # Flatten to [B, T, D] format
        A = _flatten_tokens(A_raw)  # [B, T, n_i]
        B = _flatten_tokens(B_out)  # [B, T, n_o]
        
        batch_size = A.shape[0]
        
        # For Conv1D, the weight is transposed, so we need to swap projections
        if is_conv1d:
            # Conv1D: weight is [n_out, n_in], gradient is B^T @ A (transposed)
            # So we need to swap A and B for projection
            A_proj = torch.matmul(B, self.P_i.t())  # Use B with P_i
            B_proj = torch.matmul(A, self.P_o.t())  # Use A with P_o
            # Compute gradG with swapped order
            gradG = torch.einsum('bti,btj->bji', B_proj, A_proj)  # Note: bji instead of bij
        else:
            # Regular Linear layer
            # A_proj: [B, T, n_i] @ [n_i, k_i] -> [B, T, k_i]
            A_proj = torch.matmul(A, self.P_i.t())
            
            # B_proj: [B, T, n_o] @ [n_o, k_o] -> [B, T, k_o]  
            B_proj = torch.matmul(B, self.P_o.t())
            
            # Compute per-sample projected gradients
            # Align with naive reference: [B, k_o, k_i]
            gradG = torch.einsum('bti,btj->bij', B_proj, A_proj)
        
        # Note: grad_output from CrossEntropyLoss(mean) carries a 1/B factor;
        # multiply by batch_size to match reduction='sum' naive computation.
        gradG = gradG * batch_size
        
        # Store in float32 for precision
        module._ghost_grad_proj = gradG.to(torch.float32)
        
    def _compute_embedding_proj(self, module: nn.Module, indices: torch.Tensor,
                               grad_output: torch.Tensor) -> None:
        """
        Compute projected gradients for embedding layers.
        
        Memory-efficient implementation that accumulates directly in projected space.
        For each token index j with gradient g_t, we compute:
        - P_o @ g_t (k_o-dimensional)
        - P_i[:, j] (k_i-dimensional)
        Then accumulate their outer product into [k_o, k_i] matrix.
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
            
        k_o, _ = self.P_o.shape  # [k_o, embed_dim]
        k_i, _ = self.P_i.shape  # [k_i, vocab_size]
        
        # Compute per-sample projected gradients efficiently
        per_sample_grads = []
        
        for b in range(batch_size):
            # Initialize projected gradient accumulator in low-dim space
            grad_proj = torch.zeros(k_o, k_i, dtype=torch.float32, device=grad_flat.device)
            
            idx_b = indices_flat[b]  # [T]
            grad_b = grad_flat[b]    # [T, D]
            
            # Process each token
            for t in range(idx_b.shape[0]):
                token_idx = idx_b[t].item()
                token_grad = grad_b[t]  # [D]
                
                # Project gradient: P_o @ g_t -> [k_o]
                grad_o_proj = self.P_o @ token_grad  # [k_o]
                
                # Get projection for this token index: P_i[:, j] -> [k_i]
                grad_i_proj = self.P_i[:, token_idx]  # [k_i]
                
                # Accumulate outer product: [k_o] x [k_i] -> [k_o, k_i]
                grad_proj += grad_o_proj.unsqueeze(1) @ grad_i_proj.unsqueeze(0)
            
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
        layer_type = 'Embedding'
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    
    return GradProjHooks(P_i, P_o, layer_name, layer_type)
