"""
Layer-specific implementations for gradient projection.
Provides utilities for identifying and configuring supported layer types.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


# Supported layer types for projection
SUPPORTED_DENSE_LAYERS = (nn.Linear, nn.Conv1d)
SUPPORTED_EMBEDDING_LAYERS = (nn.Embedding,)
IGNORED_LAYERS = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)


def is_transformers_conv1d(module: nn.Module) -> bool:
    """
    Check if module is a transformers.pytorch_utils.Conv1D layer.
    
    Args:
        module: Module to check
        
    Returns:
        True if it's a Conv1D from transformers
    """
    return module.__class__.__name__ == 'Conv1D'


def is_supported_layer(module: nn.Module, include_embeddings: bool = True,
                      include_conv2d: bool = False) -> bool:
    """
    Check if a layer is supported for gradient projection.
    
    Args:
        module: The layer to check
        include_embeddings: Whether to include embedding layers
        include_conv2d: Whether to include Conv2d layers
        
    Returns:
        True if the layer is supported
    """
    # Check standard supported types
    if isinstance(module, SUPPORTED_DENSE_LAYERS):
        return True
    
    # Check transformers Conv1D
    if is_transformers_conv1d(module):
        return True
        
    # Check embeddings
    if include_embeddings and isinstance(module, SUPPORTED_EMBEDDING_LAYERS):
        return True
        
    # Check Conv2d if requested
    if include_conv2d and isinstance(module, nn.Conv2d):
        return True
        
    return False


def get_layer_dimensions(module: nn.Module) -> Tuple[int, int]:
    """
    Get input and output dimensions for a layer.
    
    Args:
        module: The layer module
        
    Returns:
        (n_i, n_o): Input and output dimensions
        
    Raises:
        ValueError: If layer type is not supported
    """
    if isinstance(module, nn.Linear):
        # Linear: weight is [out_features, in_features]
        return module.in_features, module.out_features
        
    elif is_transformers_conv1d(module):
        # Transformers Conv1D: weight is [out_features, in_features]  
        weight_shape = module.weight.shape
        return weight_shape[1], weight_shape[0]
        
    elif isinstance(module, nn.Conv1d):
        # Conv1d: weight is [out_channels, in_channels, kernel_size]
        n_o = module.out_channels
        n_i = module.in_channels * module.kernel_size[0]
        return n_i, n_o
        
    elif isinstance(module, nn.Conv2d):
        # Conv2d: weight is [out_channels, in_channels, kernel_h, kernel_w]
        n_o = module.out_channels
        n_i = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        return n_i, n_o
        
    elif isinstance(module, nn.Embedding):
        # Embedding: weight is [num_embeddings, embedding_dim]
        return module.num_embeddings, module.embedding_dim
        
    else:
        raise ValueError(f"Unsupported layer type: {module.__class__.__name__}")


def find_matching_layers(module: nn.Module, layer_patterns: Union[str, List[str]],
                        include_embeddings: bool = True,
                        include_conv2d: bool = False) -> Dict[str, nn.Module]:
    """
    Find all layers in a module that match the given patterns and are supported.
    
    Args:
        module: Root module to search
        layer_patterns: Comma-separated string or list of layer name patterns
        include_embeddings: Whether to include embedding layers
        include_conv2d: Whether to include Conv2d layers
        
    Returns:
        Dictionary mapping layer names to modules
    """
    # Parse patterns
    if isinstance(layer_patterns, str):
        patterns = [p.strip() for p in layer_patterns.split(',')]
    else:
        patterns = layer_patterns
        
    matched_layers = {}
    
    # Walk through all modules
    for name, submodule in module.named_modules():
        # Skip if not supported
        if not is_supported_layer(submodule, include_embeddings, include_conv2d):
            continue
            
        # Check if name matches any pattern
        for pattern in patterns:
            if pattern in name:
                matched_layers[name] = submodule
                break
                
    return matched_layers


def validate_layer_selection(matched_layers: Dict[str, nn.Module],
                            layer_patterns: Union[str, List[str]]) -> None:
    """
    Validate that layer selection found appropriate layers.
    
    Args:
        matched_layers: Dictionary of matched layers
        layer_patterns: Original patterns used for matching
        
    Raises:
        ValueError: If no layers were matched or other issues
    """
    if len(matched_layers) == 0:
        raise ValueError(f"No layers matched patterns: {layer_patterns}")
        
    # Log matched layers
    print(f"Found {len(matched_layers)} layers for projection:")
    for name in sorted(matched_layers.keys()):
        layer = matched_layers[name]
        layer_type = layer.__class__.__name__
        try:
            n_i, n_o = get_layer_dimensions(layer)
            print(f"  - {name} ({layer_type}): {n_i} -> {n_o}")
        except Exception:
            print(f"  - {name} ({layer_type})")


def compute_total_projection_size(matched_layers: Dict[str, nn.Module],
                                 projection_dims: Dict[str, Tuple[int, int]]) -> int:
    """
    Compute total size of concatenated projection vector.
    
    Args:
        matched_layers: Dictionary of matched layers
        projection_dims: Dictionary mapping layer names to (k_i, k_o) tuples
        
    Returns:
        Total projection dimension
    """
    total_size = 0
    for layer_name in matched_layers:
        k_i, k_o = projection_dims[layer_name]
        total_size += k_i * k_o
    return total_size


def get_layer_slice_ranges(matched_layers: Dict[str, nn.Module],
                          projection_dims: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """
    Compute slice ranges for each layer in the concatenated projection vector.
    
    Args:
        matched_layers: Dictionary of matched layers (ordered)
        projection_dims: Dictionary mapping layer names to (k_i, k_o) tuples
        
    Returns:
        Dictionary mapping layer names to (start, end) indices
    """
    slice_ranges = {}
    current_idx = 0
    
    for layer_name in sorted(matched_layers.keys()):
        k_i, k_o = projection_dims[layer_name]
        layer_size = k_i * k_o
        slice_ranges[layer_name] = (current_idx, current_idx + layer_size)
        current_idx += layer_size
        
    return slice_ranges