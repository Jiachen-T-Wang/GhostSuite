"""
Projection utilities for gradient projection engine.
Handles optimal dimension selection and projection matrix initialization.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


# Version of the projection metadata schema written by GradProjLoraEngine.
# Version 2: projection matrices are generated with a CPU generator (so the recorded
# per-layer seeds reconstruct P identically on any device) and the orthonormal init
# carries the sqrt(cols/rows) calibration making E[P^T P] = I like the Gaussian init.
# Version-1 captures (no 'metadata_version' field) recorded neither property: their P
# was generated with a device-bound generator (CPU and CUDA streams differ for the
# same seed) and their orthonormal P was uncalibrated.
PROJECTION_METADATA_VERSION = 2


def check_projection_metadata_reconstructible(metadata: dict) -> None:
    """Raise unless this metadata's projection matrices can be rebuilt by the current code.

    Call before reconstructing P from saved metadata (seeds + dims). Old (version-1)
    captures cannot be reconstructed reliably: the generator device was part of P's
    identity but was never recorded, and orthonormal P lacked the sqrt(cols/rows)
    calibration — rebuilding would silently produce a *different* P with matching shapes.

    Raises:
        ValueError: for version-1 captures, unknown versions, or a non-CPU rng_device.
    """
    version = metadata.get('metadata_version')
    if version is None:
        raise ValueError(
            "Projection metadata has no 'metadata_version': this is a version-1 capture. "
            "Its projection matrices were generated with a device-bound RNG (CPU and CUDA "
            "generators produce different streams for the same seed) and, if orthonormal, "
            "without the sqrt(cols/rows) calibration — so P cannot be reconstructed from "
            "the recorded seeds. Re-run the capture with the current code "
            f"(metadata_version {PROJECTION_METADATA_VERSION}, CPU-generated P).")
    if version != PROJECTION_METADATA_VERSION:
        raise ValueError(
            f"Unsupported projection metadata_version {version}; this code supports "
            f"version {PROJECTION_METADATA_VERSION} only.")
    rng_device = metadata.get('rng_device')
    if rng_device != 'cpu':
        raise ValueError(
            f"Projection metadata records rng_device={rng_device!r}; only 'cpu'-generated "
            "projection matrices can be reconstructed device-independently.")


def choose_ki_ko(n_i: int, n_o: int, k_total: int, k_min: int = 1) -> Tuple[int, int]:
    """
    Choose optimal projection dimensions k_i and k_o to minimize computational cost.

    Targets:
    - k_i/k_o ≈ n_o/n_i (the cost-minimizing aspect ratio)
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

    # Ratio rule k_i/k_o ≈ n_o/n_i: k_i ≈ sqrt(k_total * n_o/n_i), k_o ≈ sqrt(k_total * n_i/n_o).
    root = math.sqrt(k_total)
    r = math.sqrt(n_o / max(1, n_i))

    def _clamp_i(v: int) -> int:
        return max(k_min, min(n_i, max(1, v)))

    def _clamp_o(v: int) -> int:
        return max(k_min, min(n_o, max(1, v)))

    def _derive(i_first: bool) -> Tuple[int, int]:
        # One dim from the ratio rule, the other from the budget — then re-derive the
        # first from the (possibly clamped) second, so a k_min/n clamp on the second dim
        # cannot leave the product far over k_total (e.g. 768x50257 @ k_total=256, k_min=8
        # previously returned (127, 8) -> 1016).
        if i_first:
            k_i = _clamp_i(int(round(root * r)))
            k_o = _clamp_o(k_total // k_i)
            k_i = _clamp_i(k_total // k_o)
        else:
            k_o = _clamp_o(int(round(root / r)))
            k_i = _clamp_i(k_total // k_o)
            k_o = _clamp_o(k_total // k_i)
        return k_i, k_o

    # Evaluate both derivation orders; keep the one whose product is closest to k_total.
    k_i, k_o = min((_derive(True), _derive(False)),
                   key=lambda c: abs(c[0] * c[1] - k_total))

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

    Generation always uses a CPU generator (then moves to `device`): CPU and CUDA
    generators produce different streams for the same seed, so a device-bound
    generator would make the recorded seeds insufficient to reconstruct P across
    devices (see metadata_version 2).

    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)
        dtype: Data type for the matrix
        device: Device the returned matrix lives on
        seed: Random seed for reproducibility

    Returns:
        Projection matrix of shape [rows, cols]
    """
    if seed is not None:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    else:
        generator = None

    # Standard deviation = 1/sqrt(rows) for JL property
    std = 1.0 / math.sqrt(rows)
    P = torch.randn(rows, cols, dtype=dtype, device='cpu', generator=generator) * std
    P = P.to(device)
    P.requires_grad_(False)

    return P


def init_projection_matrix_rademacher(rows: int, cols: int, dtype: torch.dtype = torch.float32,
                                     device: torch.device = torch.device('cpu'),
                                     seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize projection matrix using Rademacher distribution.
    Each entry is ±1/sqrt(rows) with probability 1/2.

    Generated with a CPU generator and moved to `device` so the seed reconstructs P
    device-independently (see init_projection_matrix_gaussian).

    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)
        dtype: Data type for the matrix
        device: Device the returned matrix lives on
        seed: Random seed for reproducibility

    Returns:
        Projection matrix of shape [rows, cols]
    """
    if seed is not None:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    else:
        generator = None

    # Generate random signs as integers first
    signs = torch.randint(0, 2, (rows, cols), dtype=torch.int8, device='cpu', generator=generator)
    # Convert to ±1 floats
    signs = (signs * 2 - 1).to(dtype)
    scale = 1.0 / math.sqrt(rows)
    P = (signs * scale).to(device)
    P.requires_grad_(False)

    return P


def init_projection_matrix_orthonormal(rows: int, cols: int, dtype: torch.dtype = torch.float32,
                                      device: torch.device = torch.device('cpu'),
                                      seed: Optional[int] = None) -> torch.Tensor:
    """
    Initialize a row-orthogonal projection matrix using economy QR decomposition,
    calibrated so E[P^T P] = I like the Gaussian init.

    The QR step yields orthonormal rows (Q^T Q^T^T = I_rows), but a bare orthonormal P
    has E[P^T P] = (rows/cols) I — two-sided projection would then shrink inner products
    by a layer-dependent (k_i k_o)/(n_i n_o) factor. We therefore scale by
    sqrt(cols/rows): rows stay mutually orthogonal (P P^T = (cols/rows) I_rows) and
    E[P^T P] = I, so projected dot products are on the same scale as the Gaussian init.

    Uses economy approach: Generate [cols x rows] Gaussian, compute QR,
    then transpose Q to get the row-orthogonal [rows x cols] matrix.
    This is more memory-efficient than full [cols x cols] QR.

    Generated with a CPU generator and moved to `device` so the seed reconstructs P
    device-independently (see init_projection_matrix_gaussian).

    Args:
        rows: Number of rows (projection dimension)
        cols: Number of columns (original dimension)
        dtype: Data type for the matrix
        device: Device the returned matrix lives on
        seed: Random seed for reproducibility

    Returns:
        Projection matrix of shape [rows, cols] with orthogonal rows of norm sqrt(cols/rows)
    """
    if rows > cols:
        raise ValueError(f"Cannot create {rows} orthonormal rows in {cols} dimensions")

    if seed is not None:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
    else:
        generator = None

    # Economy approach: Generate [cols x rows] matrix and compute QR
    # This gives us Q with shape [cols x rows] where columns are orthonormal
    M = torch.randn(cols, rows, dtype=torch.float64, device='cpu', generator=generator)
    Q, _ = torch.linalg.qr(M, mode='reduced')  # Q is [cols x rows] with orthonormal columns

    # Transpose to row-orthogonal [rows x cols] and apply the sqrt(cols/rows)
    # calibration so E[P^T P] = I (unbiased inner-product estimator, like Gaussian).
    P = (Q.t() * math.sqrt(cols / rows)).to(dtype).to(device)
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

    # Add original dimensions based on layer type. Dispatch on the concrete type
    # (not hasattr('weight'), which is also true for Embedding and would shadow
    # the Embedding branch, leaving it without n_i/n_o).
    if isinstance(layer, nn.Linear):
        metadata['n_o'], metadata['n_i'] = layer.weight.shape
    elif isinstance(layer, nn.Conv1d):
        weight_shape = layer.weight.shape
        metadata['n_o'] = weight_shape[0]
        metadata['n_i'] = weight_shape[1] * weight_shape[2]
    elif layer.__class__.__name__ == 'Conv1D':  # transformers Conv1D: weight [in, out]
        metadata['n_i'], metadata['n_o'] = layer.weight.shape
    elif isinstance(layer, nn.Embedding):
        metadata['vocab_size'] = layer.num_embeddings
        metadata['n_i'] = layer.num_embeddings
        metadata['n_o'] = layer.embedding_dim

    return metadata