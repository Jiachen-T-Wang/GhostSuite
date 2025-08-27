"""
Main gradient projection engine using LoRA-style architecture.
Computes and stores per-sample projected gradients efficiently.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from collections import OrderedDict

from .projection_utils import (
    choose_ki_ko, 
    get_projection_initializer,
    compute_projection_metadata
)
from .autograd_gradproj import create_projection_hooks
from .supported_layers_gradproj import (
    find_matching_layers,
    validate_layer_selection,
    get_layer_dimensions,
    compute_total_projection_size,
    get_layer_slice_ranges
)


class GradProjLoraEngine:
    """
    Gradient Projection Engine using LoRA-style side branches.
    
    This engine computes per-sample projected gradients without modifying
    the model's forward pass or training dynamics. It uses low-rank projections
    to reduce gradient dimensionality while preserving similarity structure.
    """
    
    def __init__(self, 
                 module: nn.Module,
                 proj_layers: Union[str, List[str]],
                 proj_rank_total: int,
                 proj_rank_min: int,
                 proj_seed: int,
                 proj_dtype: str,
                 proj_dir: str,
                 proj_row_orthonormal: bool = False,
                 include_embeddings: bool = False,
                 include_conv2d: bool = False,
                 proj_save_interval: int = 1,
                 **kwargs):
        """
        Initialize the gradient projection engine.
        
        Args:
            module: The model to attach projections to
            proj_layers: Comma-separated patterns for layers to project
            proj_rank_total: Target total projection dimension per layer
            proj_rank_min: Minimum dimension for k_i and k_o
            proj_seed: Random seed for projection matrices
            proj_dtype: Data type for storage (float16, bfloat16, float32)
            proj_dir: Directory to save projected gradients
            proj_row_orthonormal: Whether to use row-orthonormal projections
            include_embeddings: Whether to include embedding layers
            include_conv2d: Whether to include Conv2d layers
            proj_save_interval: How often to save projections (in iterations)
            **kwargs: Additional unused arguments for compatibility
        """
        self.module = module
        self.proj_layers = proj_layers
        self.proj_rank_total = proj_rank_total
        self.proj_rank_min = proj_rank_min
        self.proj_seed = proj_seed
        self.proj_dir = Path(proj_dir)
        self.proj_save_interval = proj_save_interval
        
        # Parse dtype
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        if proj_dtype not in dtype_map:
            raise ValueError(f"proj_dtype must be one of {list(dtype_map.keys())}, got {proj_dtype}")
        self.proj_dtype = dtype_map[proj_dtype]
        
        # Projection method
        self.proj_method = 'orthonormal' if proj_row_orthonormal else 'gaussian'
        
        # Options
        self.include_embeddings = include_embeddings
        self.include_conv2d = include_conv2d
        
        # State
        self.matched_layers = OrderedDict()
        self.projection_matrices = {}
        self.projection_dims = {}
        self.hooks = {}
        self.slice_ranges = {}
        self.total_proj_dim = 0
        self.metadata = {}
        self.is_attached = False
        
        # Counters
        self.iteration = 0
        self.batch_count = 0
        
        # Initialize projections
        self._initialize_projections()
        
    def _initialize_projections(self):
        """Initialize projection matrices for selected layers."""
        # Find matching layers
        self.matched_layers = find_matching_layers(
            self.module, 
            self.proj_layers,
            self.include_embeddings,
            self.include_conv2d
        )
        
        # Validate selection
        validate_layer_selection(self.matched_layers, self.proj_layers)
        
        # Get device from first parameter
        device = next(self.module.parameters()).device
        
        # Create projection matrices for each layer
        init_fn = get_projection_initializer(self.proj_method)
        
        # Use deterministic seed for each layer
        torch.manual_seed(self.proj_seed)
        
        for layer_idx, (layer_name, layer) in enumerate(self.matched_layers.items()):
            # Get layer dimensions
            n_i, n_o = get_layer_dimensions(layer)
            
            # Choose optimal projection dimensions
            k_i, k_o = choose_ki_ko(n_i, n_o, self.proj_rank_total, self.proj_rank_min)
            self.projection_dims[layer_name] = (k_i, k_o)
            
            # Create projection matrices with layer-specific seed
            layer_seed = self.proj_seed + layer_idx
            
            P_i = init_fn(k_i, n_i, dtype=torch.float32, device=device, seed=layer_seed)
            P_o = init_fn(k_o, n_o, dtype=torch.float32, device=device, seed=layer_seed + 1000)
            
            self.projection_matrices[layer_name] = (P_i, P_o)
            
            print(f"  Projection dims: k_i={k_i}, k_o={k_o} (k_total={k_i*k_o})")
            
        # Compute slice ranges for concatenation
        self.slice_ranges = get_layer_slice_ranges(self.matched_layers, self.projection_dims)
        self.total_proj_dim = compute_total_projection_size(self.matched_layers, self.projection_dims)
        
        print(f"Total projection dimension: {self.total_proj_dim}")
        
        # Prepare metadata
        self._prepare_metadata()
        
    def _prepare_metadata(self):
        """Prepare metadata for saving."""
        self.metadata = {
            'engine': 'GradProjLora',
            'proj_seed': self.proj_seed,
            'proj_dtype': str(self.proj_dtype).split('.')[-1],
            'proj_method': self.proj_method,
            'proj_rank_total': self.proj_rank_total,
            'proj_rank_min': self.proj_rank_min,
            'total_proj_dim': self.total_proj_dim,
            'layers': []
        }
        
        # Add per-layer metadata
        for layer_name in sorted(self.matched_layers.keys()):
            layer = self.matched_layers[layer_name]
            k_i, k_o = self.projection_dims[layer_name]
            start, end = self.slice_ranges[layer_name]
            
            layer_meta = compute_projection_metadata(layer_name, layer, k_i, k_o)
            layer_meta['slice_start'] = start
            layer_meta['slice_end'] = end
            
            self.metadata['layers'].append(layer_meta)
            
    def attach(self):
        """Attach hooks to selected layers."""
        if self.is_attached:
            return
            
        for layer_name, layer in self.matched_layers.items():
            P_i, P_o = self.projection_matrices[layer_name]
            
            # Create and attach hooks
            hooks = create_projection_hooks(layer, layer_name, P_i, P_o)
            hooks.attach(layer)
            self.hooks[layer_name] = hooks
            
        self.is_attached = True
        print(f"Attached projection hooks to {len(self.matched_layers)} layers")
        
    def detach(self):
        """Remove hooks from layers and clean up."""
        if not self.is_attached:
            return
            
        # Remove hooks
        for layer_name, hooks in self.hooks.items():
            hooks.detach()
            
            # Clean up any cached data
            layer = self.matched_layers[layer_name]
            if hasattr(layer, '_ghost_A_raw'):
                delattr(layer, '_ghost_A_raw')
            if hasattr(layer, '_ghost_grad_proj'):
                delattr(layer, '_ghost_grad_proj')
                
        self.hooks.clear()
        self.is_attached = False
        print(f"Detached projection hooks from {len(self.matched_layers)} layers")
        
    def collect_batch(self, batch_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Collect projected gradients from all layers and optionally save.
        
        Args:
            batch_indices: Optional list of sample indices in the batch
            
        Returns:
            Concatenated projection tensor of shape [B, total_proj_dim]
            
        Raises:
            RuntimeError: If no gradients are available
        """
        if not self.is_attached:
            raise RuntimeError("Engine is not attached. Call attach() first.")
            
        # Collect per-layer projections
        layer_projections = []
        batch_size = None
        
        for layer_name in sorted(self.matched_layers.keys()):
            layer = self.matched_layers[layer_name]
            
            # Get projected gradient
            grad_proj = getattr(layer, '_ghost_grad_proj', None)
            if grad_proj is None:
                raise RuntimeError(f"No projected gradient found for layer {layer_name}")
                
            # Flatten to [B, k_i * k_o]
            B, k_o, k_i = grad_proj.shape
            grad_flat = grad_proj.reshape(B, k_i * k_o)
            
            if batch_size is None:
                batch_size = B
            elif B != batch_size:
                raise RuntimeError(f"Batch size mismatch: expected {batch_size}, got {B} for {layer_name}")
                
            layer_projections.append(grad_flat)
            
        # Concatenate all layers
        full_projection = torch.cat(layer_projections, dim=1)  # [B, total_proj_dim]
        
        # Convert to storage dtype
        full_projection = full_projection.to(self.proj_dtype)
        
        # Save if needed
        if self.iteration % self.proj_save_interval == 0:
            self._save_projection(full_projection, batch_indices)
            
        self.iteration += 1
        self.batch_count += batch_size
        
        return full_projection
        
    def _save_projection(self, projection: torch.Tensor, batch_indices: Optional[List[int]] = None):
        """Save projection to disk."""
        # Create directory if needed
        self.proj_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata on first save
        metadata_path = self.proj_dir / 'metadata.json'
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            
        # Prepare save dict
        save_dict = {
            'proj': projection.cpu(),
            'iter': self.iteration,
            'batch_size': projection.shape[0],
        }
        
        if batch_indices is not None:
            save_dict['batch_idx'] = batch_indices
            
        # Save projection
        filename = f'proj_iter_{self.iteration:06d}.pt'
        save_path = self.proj_dir / filename
        torch.save(save_dict, save_path)
        
        print(f"Saved projection [{projection.shape}] to {save_path}")
        
    def aggregate_and_log(self, result_dict: Optional[dict] = None):
        """
        Compatibility method for integration with training loop.
        Can be called after collect_batch() for any additional logging.
        
        Args:
            result_dict: Optional dictionary to add metrics to
        """
        if result_dict is not None:
            result_dict['proj_iteration'] = self.iteration
            result_dict['proj_batch_count'] = self.batch_count
            result_dict['proj_total_dim'] = self.total_proj_dim
            
    def get_projection_metadata(self) -> dict:
        """Get metadata about the projection configuration."""
        return self.metadata.copy()
        
    # === Compatibility methods for engine_manager integration ===
    
    def attach_train_batch(self, X_train, Y_train, iter_num, batch_idx=None):
        """
        Store training batch information (compatibility method).
        
        For GradProjLora, we track iteration number for saving purposes.
        """
        self.current_iter_num = iter_num
        self.current_batch_idx = batch_idx
        
    def prepare_gradients(self):
        """
        Prepare gradients after backward pass (compatibility method).
        
        For GradProjLora, projections are computed during backward hooks,
        so this is a no-op.
        """
        pass
        
    def aggregate_and_log(self):
        """
        Aggregate and log metrics after optimizer step (compatibility method).
        
        This calls collect_batch() to compute and save projections.
        """
        if hasattr(self, 'current_batch_idx') and self.current_batch_idx is not None:
            # Collect with batch indices if available
            self.collect_batch(batch_indices=self.current_batch_idx)
        else:
            # Collect without batch indices
            self.collect_batch()
            
    def clear_gradients(self):
        """
        Clear gradients and cached data after optimizer step (compatibility method).
        
        This cleans up any cached activations or gradients.
        """
        # Clean up cached data in layers
        for layer_name, layer in self.matched_layers.items():
            if hasattr(layer, '_ghost_A_raw'):
                delattr(layer, '_ghost_A_raw')
            if hasattr(layer, '_ghost_grad_proj'):
                delattr(layer, '_ghost_grad_proj')
                
    def attach_with_optimizer(self, optimizer):
        """
        Attach to optimizer (compatibility method for engine_manager).
        
        For GradProjLora, we don't need the optimizer, so this just
        calls the regular attach() method.
        """
        self.attach()
            
    def detach_for_evaluation(self):
        """
        Detach during evaluation (compatibility method).
        
        Alias for the existing detach() method.
        """
        self.detach()
        
    def reattach_after_evaluation(self):
        """
        Reattach after evaluation (compatibility method).
        
        Re-attaches projection hooks after evaluation.
        """
        if not self.is_attached:
            # Re-attach hooks
            for layer_name, layer in self.matched_layers.items():
                P_i, P_o = self.projection_matrices[layer_name]
                
                # Create and attach hooks
                from .autograd_gradproj import create_projection_hooks
                hooks = create_projection_hooks(layer, layer_name, P_i, P_o)
                hooks.attach(layer)
                self.hooks[layer_name] = hooks
                
            self.is_attached = True
            
    def cleanup(self):
        """
        Cleanup and save any remaining data (compatibility method).
        
        Ensures all projections are saved and cleans up resources.
        """
        # Save any pending projections
        if hasattr(self, '_pending_projections'):
            # Implementation depends on whether we buffer projections
            pass
            
        # Detach all hooks
        self.detach()
        
        # Clear projection matrices to free memory
        self.projection_matrices.clear()
        
    def save_projections(self, iter_num: int):
        """
        Save projections at the given iteration (compatibility method).
        
        This is called by engine_manager's save_metrics().
        For GradProjLora, projections are saved in collect_batch(),
        so this can be a no-op or trigger a forced save.
        """
        # Projections are saved automatically in collect_batch()
        # This method exists for compatibility
        pass
        
    def __repr__(self):
        return (f"GradProjLoraEngine(layers={len(self.matched_layers)}, "
                f"total_dim={self.total_proj_dim}, "
                f"attached={self.is_attached})")


def create_gradproj_engine(model: nn.Module, config: dict) -> GradProjLoraEngine:
    """
    Factory function to create engine from config dict.
    
    Args:
        model: Model to attach engine to
        config: Configuration dictionary
        
    Returns:
        Configured GradProjLoraEngine instance
    """
    # Extract required parameters
    required_params = [
        'proj_layers', 'proj_rank_total', 'proj_rank_min',
        'proj_seed', 'proj_dtype', 'proj_dir'
    ]
    
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
            
    return GradProjLoraEngine(model, **config)