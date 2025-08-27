"""
Projection utilities for gradient projection engine.
Handles optimal dimension selection and projection matrix initialization.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


def choose_ki_ko(n_i: int, n_o: int, k_total: int, k_min: int = 1) -> Tuple[int, int]:
    """
    Choose optimal projection dimensions k_i and k_o to minimize computational cost.
    
    Following the formula from the plan:
    - k_i/k_o ≈ sqrt(n_o/n_i)
    - k_i * k_o ≈ k_total
    - Both k_i and k_o must be at least k_min
    
    Args:
        n_i: Input dimension of the layer
        n_o: Output dimension of the layer  
        k_total: Target total projection dimension (k_i * k_o)
        k_min: Minimum dimension for k_i and k_o
        
    Returns:
        (k_i, k_o): Optimal projection dimensions
        
    Raises:
        ValueError: If constraints cannot be satisfied
    """
    if min(n_i, n_o) <= 0 or k_total <= 0:
        raise ValueError(f"Invalid dimensions for projection: n_i={n_i}, n_o={n_o}, k_total={k_total}")
    
    if k_min > min(n_i, n_o):
        raise ValueError(f"k_min={k_min} exceeds layer dimensions (n_i={n_i}, n_o={n_o})")
    
    # Start with the ratio rule
    root = math.sqrt(k_total)
    r = math.sqrt(n_o / max(1, n_i))
    
    # Initial guess
    k_i = max(k_min, min(n_i, max(1, int(round(root * r)))))
    k_o = max(k_min, min(n_o, max(1, k_total // k_i)))
    
    # Fine-tune to get closer to k_total while respecting bounds
    best = (k_i, k_o)
    best_err = abs(k_i * k_o - k_total)
    
    # Try small adjustments
    for di in (-2, -1, 0, 1, 2):
        for dj in (-2, -1, 0, 1, 2):
            ki, ko = k_i + di, k_o + dj
            if k_min <= ki <= n_i and k_min <= ko <= n_o:
                err = abs(ki * ko - k_total)
                if err < best_err:
                    best, best_err = (ki, ko), err
    
    ki, ko = best
    
    # Final validation
    if ki < k_min or ko < k_min:
        raise ValueError(f"Cannot satisfy k_min={k_min} constraints with k_total={k_total}")
    
    return ki, ko


def init_projection_matrix_gaussian(rows: int, cols: int, dtype: torch.dtype = torch.float32, 
                                   device: torch.device = torch.device('cpu'),
                                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize projection matrix using Gaussian JL (Johnson-Lindenstrauss).
    Each entry ~ N(0, 1/rows) so E[P^T P] ≈ I.
    
    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)
        dtype: Data type for the matrix
        device: Device to create the matrix on
        seed: Random seed for reproducibility
        
    Returns:
        Projection matrix of shape [rows, cols]
    """
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Standard deviation = 1/sqrt(rows) for JL property
    std = 1.0 / math.sqrt(rows)
    P = torch.randn(rows, cols, dtype=dtype, device=device, generator=generator) * std
    P.requires_grad_(False)
    
    return P


def init_projection_matrix_rademacher(rows: int, cols: int, dtype: torch.dtype = torch.float32,
                                     device: torch.device = torch.device('cpu'), 
                                     seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize projection matrix using Rademacher distribution.
    Each entry is ±1/sqrt(rows) with probability 1/2.
    
    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)  
        dtype: Data type for the matrix
        device: Device to create the matrix on
        seed: Random seed for reproducibility
        
    Returns:
        Projection matrix of shape [rows, cols]
    """
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Generate random signs as integers first
    signs = torch.randint(0, 2, (rows, cols), dtype=torch.int8, device=device, generator=generator)
    # Convert to ±1 floats
    signs = (signs * 2 - 1).to(dtype)
    scale = 1.0 / math.sqrt(rows)
    P = signs * scale
    P.requires_grad_(False)
    
    return P


def init_projection_matrix_orthonormal(rows: int, cols: int, dtype: torch.dtype = torch.float32,
                                      device: torch.device = torch.device('cpu'),
                                      seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize row-orthonormal projection matrix using economy QR decomposition.
    Satisfies P @ P^T = I_rows for exact energy preservation.
    
    Uses economy approach: Generate [cols x rows] Gaussian, compute QR, 
    then transpose Q to get row-orthonormal [rows x cols] matrix.
    This is more memory-efficient than full [cols x cols] QR.
    
    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)
        dtype: Data type for the matrix
        device: Device to create the matrix on  
        seed: Random seed for reproducibility
        
    Returns:
        Projection matrix of shape [rows, cols] with orthonormal rows
    """
    if rows > cols:
        raise ValueError(f"Cannot create {rows} orthonormal rows in {cols} dimensions")
    
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Economy approach: Generate [cols x rows] matrix and compute QR
    # This gives us Q with shape [cols x rows] where columns are orthonormal
    M = torch.randn(cols, rows, dtype=torch.float64, device=device, generator=generator)
    Q, _ = torch.linalg.qr(M, mode='reduced')  # Q is [cols x rows] with orthonormal columns
    
    # Transpose to get row-orthonormal matrix [rows x cols]
    # Since Q has orthonormal columns, Q^T has orthonormal rows
    P = Q.t().to(dtype)  # [rows x cols]
    P.requires_grad_(False)
    
    return P


def get_projection_initializer(method: str = 'gaussian'):
    """
    Get projection matrix initializer function by method name.
    
    Args:
        method: One of 'gaussian', 'rademacher', or 'orthonormal'
        
    Returns:
        Initialization function
        
    Raises:
        ValueError: If method is not recognized
    """
    initializers = {
        'gaussian': init_projection_matrix_gaussian,
        'rademacher': init_projection_matrix_rademacher,
        'orthonormal': init_projection_matrix_orthonormal,
    }
    
    if method not in initializers:
        raise ValueError(f"Unknown projection method: {method}. Choose from {list(initializers.keys())}")
    
    return initializers[method]


def compute_projection_metadata(layer_name: str, layer: nn.Module, 
                               k_i: int, k_o: int) -> dict:
    """
    Compute metadata for a projected layer.
    
    Args:
        layer_name: Name/path of the layer in the model
        layer: The layer module
        k_i: Input projection dimension
        k_o: Output projection dimension
        
    Returns:
        Dictionary with layer metadata
    """
    metadata = {
        'name': layer_name,
        'type': layer.__class__.__name__,
        'k_i': k_i,
        'k_o': k_o,
        'k_total': k_i * k_o,
    }
    
    # Add original dimensions based on layer type
    if hasattr(layer, 'weight'):
        weight_shape = layer.weight.shape
        if isinstance(layer, nn.Linear):
            metadata['n_o'], metadata['n_i'] = weight_shape
        elif isinstance(layer, nn.Conv1d):
            metadata['n_o'] = weight_shape[0]
            metadata['n_i'] = weight_shape[1] * weight_shape[2]
        elif layer.__class__.__name__ == 'Conv1D':  # transformers Conv1D
            metadata['n_o'], metadata['n_i'] = weight_shape
    elif isinstance(layer, nn.Embedding):
        metadata['vocab_size'] = layer.num_embeddings
        metadata['n_i'] = layer.num_embeddings
        metadata['n_o'] = layer.embedding_dim
        
    return metadata