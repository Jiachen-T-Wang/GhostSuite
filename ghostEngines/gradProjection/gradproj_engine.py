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
    PROJECTION_METADATA_VERSION,
    choose_ki_ko,
    get_projection_initializer,
    compute_projection_metadata
)
from .autograd_gradproj import check_embedding_supported, create_projection_hooks
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
            proj_save_interval: Save every Nth collect_batch call's projections to
                disk. Batches in between are computed and returned to the caller but
                are NOT buffered — they are discarded, not batched up for a later
                save. Use 1 (the default) to persist every batch.
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
        self.projection_seeds = {}
        self.hooks = {}
        self.slice_ranges = {}
        self.total_proj_dim = 0
        self.metadata = {}
        self.is_attached = False

        # Counters
        self.iteration = 0
        self.batch_count = 0

        # One-time notice that proj_save_interval > 1 discards (not buffers) skipped batches.
        self._save_interval_notice_printed = False

        # Gradient-accumulation state. None when no accumulation step is in
        # progress (the legacy single-microbatch path). begin_step() sets it to a
        # per-layer list of microbatch projections that collect_batch() concatenates.
        self._micro_buffers = None

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

        # Seed each layer by its position in sorted-name order. Every other
        # structure (slice ranges, concatenation, metadata) is built in sorted
        # order, so seeding by sorted index keeps the saved metadata sufficient
        # to reconstruct P (named_modules() order is not sorted, e.g. h.10 < h.2).
        for layer_idx, layer_name in enumerate(sorted(self.matched_layers.keys())):
            layer = self.matched_layers[layer_name]
            # Get layer dimensions
            n_i, n_o = get_layer_dimensions(layer)

            # Choose optimal projection dimensions
            k_i, k_o = choose_ki_ko(n_i, n_o, self.proj_rank_total, self.proj_rank_min)
            self.projection_dims[layer_name] = (k_i, k_o)

            # Create projection matrices with layer-specific seed
            seed_i = self.proj_seed + layer_idx
            seed_o = seed_i + 1000
            self.projection_seeds[layer_name] = (seed_i, seed_o)

            P_i = init_fn(k_i, n_i, dtype=torch.float32, device=device, seed=seed_i)
            P_o = init_fn(k_o, n_o, dtype=torch.float32, device=device, seed=seed_o)

            self.projection_matrices[layer_name] = (P_i, P_o)

            print(f"  Projection dims: k_i={k_i}, k_o={k_o} (k_total={k_i*k_o})")

        # Compute slice ranges for concatenation
        self.slice_ranges = get_layer_slice_ranges(self.matched_layers, self.projection_dims)
        self.total_proj_dim = compute_total_projection_size(self.matched_layers, self.projection_dims)

        print(f"[INFO] Total projection dimension: {self.total_proj_dim}")

        # Prepare metadata
        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare metadata for saving."""
        self.metadata = {
            'engine': 'GradProjLora',
            # Version 2: P generated with a CPU generator (device-independent for a
            # given seed) and orthonormal P calibrated by sqrt(cols/rows) so
            # E[P^T P] = I. Reconstruction paths must check these fields (see
            # projection_utils.check_projection_metadata_reconstructible).
            'metadata_version': PROJECTION_METADATA_VERSION,
            'rng_device': 'cpu',
            'proj_seed': self.proj_seed,
            'proj_dtype': str(self.proj_dtype).split('.')[-1],
            'proj_method': self.proj_method,
            'proj_rank_total': self.proj_rank_total,
            'proj_rank_min': self.proj_rank_min,
            'total_proj_dim': self.total_proj_dim,
            'layers': []
        }
        if self.proj_method == 'orthonormal':
            # Records that this capture's orthonormal P includes the sqrt(cols/rows)
            # factor (uncalibrated version-1 orthonormal captures shrink dot products
            # by a layer-dependent (k_i*k_o)/(n_i*n_o) factor).
            self.metadata['orthonormal_calibration'] = 'sqrt(cols/rows)'

        # Add per-layer metadata
        for layer_name in sorted(self.matched_layers.keys()):
            layer = self.matched_layers[layer_name]
            k_i, k_o = self.projection_dims[layer_name]
            start, end = self.slice_ranges[layer_name]

            layer_meta = compute_projection_metadata(layer_name, layer, k_i, k_o)
            layer_meta['slice_start'] = start
            layer_meta['slice_end'] = end
            # Persist the exact per-layer seeds so P can be reconstructed
            # independently of the seeding scheme.
            seed_i, seed_o = self.projection_seeds[layer_name]
            layer_meta['seed_i'] = seed_i
            layer_meta['seed_o'] = seed_o

            self.metadata['layers'].append(layer_meta)

    def attach(self, optimizer=None):
        """Attach hooks to selected layers.

        ``optimizer`` is accepted (and ignored) so the signature matches the GhostEngine
        protocol — GradProjLora only observes gradients, it does not update them.
        """
        if self.is_attached:
            return

        # Validate every embedding before attaching any hook, so an unsupported
        # option fails atomically instead of leaving earlier layers hooked.
        for layer_name, layer in self.matched_layers.items():
            if isinstance(layer, nn.Embedding):
                check_embedding_supported(layer, layer_name)

        for layer_name, layer in self.matched_layers.items():
            P_i, P_o = self.projection_matrices[layer_name]

            # Create and attach hooks
            hooks = create_projection_hooks(layer, layer_name, P_i, P_o)
            hooks.attach(layer)
            self.hooks[layer_name] = hooks

        self.is_attached = True
        print(f"[INFO] Attached projection hooks to {len(self.matched_layers)} layers")

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
        print(f"[INFO] Detached projection hooks from {len(self.matched_layers)} layers")

    def begin_step(self):
        """Begin a gradient-accumulation step: reset the per-layer microbatch buffers.

        Call once before the microbatch loop. Each microbatch's per-sample projections
        are then appended by ``collect_microbatch``; the end-of-step ``collect_batch``
        concatenates them over the batch dimension into ``[N*microbatch, ...]`` and saves
        once per optimizer step. A single-microbatch step
        (``gradient_accumulation_steps == 1``) does not need this — the legacy
        ``backward -> collect_batch`` path still works unchanged.
        """
        self._micro_buffers = {name: [] for name in self.matched_layers}

    def collect_microbatch(self):
        """Append this microbatch's per-sample projections to the per-step buffers.

        Call after each microbatch's ``backward()``, inside the accumulation loop, before
        the next backward overwrites the per-layer ``_ghost_grad_proj`` slot. Reads each
        matched layer's projection, stashes it, and clears the per-layer scratch so the
        next microbatch starts clean.

        Scaling: the projection magnitude follows the loss reduction of *this*
        microbatch's backward. To make the pooled result identical to a single
        ``gradient_accumulation_steps == 1`` pass over the concatenated batch, do **not**
        pre-divide the per-microbatch loss by the number of accumulation steps — each
        microbatch carries its own mean reduction, which the ``* batch_size`` factor in
        the hooks already undoes with the local microbatch size.
        """
        if self._micro_buffers is None:
            raise RuntimeError(
                "collect_microbatch() called without begin_step(). Call begin_step() once "
                "before the gradient-accumulation microbatch loop.")
        for layer_name in sorted(self.matched_layers.keys()):
            layer = self.matched_layers[layer_name]
            grad_proj = getattr(layer, '_ghost_grad_proj', None)
            if grad_proj is None:
                raise RuntimeError(f"No projected gradient found for layer {layer_name}")
            self._micro_buffers[layer_name].append(grad_proj)
            # Clear the single-slot scratch so the next microbatch's backward starts
            # clean (the data now lives in the per-step buffer, not the layer).
            if hasattr(layer, '_ghost_grad_proj'):
                delattr(layer, '_ghost_grad_proj')
            if hasattr(layer, '_ghost_A_raw'):
                delattr(layer, '_ghost_A_raw')

    # Keys written by _save_projection itself; callers may not override them via `extra`.
    _RESERVED_EXTRA_KEYS = frozenset({'proj', 'iter', 'batch_size', 'batch_idx'})

    def collect_batch(self, batch_indices: Optional[List[int]] = None,
                      extra: Optional[dict] = None,
                      save: bool = True) -> torch.Tensor:
        """
        Collect projected gradients from all layers and optionally save.

        Two modes:
        - Single-microbatch (legacy): reads each layer's current ``_ghost_grad_proj``
          slot (set by the most recent backward).
        - Gradient accumulation: if ``begin_step()`` opened a step, concatenates each
          layer's per-microbatch buffers over the batch dimension into ``[N*mb, ...]``
          (so every microbatch's samples are captured, not just the last), then resets
          the buffers for the next step.

        Args:
            batch_indices: Optional list of sample indices in the (possibly pooled) batch
            extra: Optional dict of extra per-step metadata to store in the saved
                file (e.g. {'lr': ..., 'order': ...} for Data Value Embedding). Keys
                'proj', 'iter', 'batch_size', 'batch_idx' are reserved.
            save: When False, never write a file for this call regardless of
                proj_save_interval. Use for transient passes (e.g. capturing test
                gradients) where only the returned tensor is needed.

        Returns:
            Concatenated projection tensor of shape [B, total_proj_dim]

        Raises:
            RuntimeError: If no gradients are available
            ValueError: If `extra` contains a reserved key
        """
        if not self.is_attached:
            raise RuntimeError("Engine is not attached. Call attach() first.")

        # Validate up front so a bad key fails fast even on non-save steps.
        if extra is not None:
            bad = self._RESERVED_EXTRA_KEYS & extra.keys()
            if bad:
                raise ValueError(f"extra keys {sorted(bad)} are reserved")

        accumulating = self._micro_buffers is not None

        # Collect per-layer projections
        layer_projections = []
        batch_size = None

        for layer_name in sorted(self.matched_layers.keys()):
            layer = self.matched_layers[layer_name]

            if accumulating:
                # Concatenate this layer's microbatch projections over the batch dim.
                micro = self._micro_buffers.get(layer_name)
                if not micro:
                    raise RuntimeError(
                        f"No accumulated projections for layer {layer_name}; call "
                        f"collect_microbatch() after each microbatch backward.")
                grad_proj = torch.cat(micro, dim=0)  # [N*mb, k_o, k_i]
            else:
                # Get projected gradient from the single-slot scratch.
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
        if save and self.proj_save_interval > 1 and not self._save_interval_notice_printed:
            print(f"[INFO] GradProjLora: proj_save_interval={self.proj_save_interval} — saving "
                  f"every {self.proj_save_interval}th batch's projections; the batches in "
                  "between are returned to the caller but NOT buffered — they are discarded.")
            self._save_interval_notice_printed = True
        if save and self.iteration % self.proj_save_interval == 0:
            self._save_projection(full_projection, batch_indices, extra)

        self.iteration += 1
        self.batch_count += batch_size

        # Close the accumulation step so the next begin_step() starts fresh.
        if accumulating:
            self._micro_buffers = None

        return full_projection

    def _save_projection(self, projection: torch.Tensor,
                         batch_indices: Optional[List[int]] = None,
                         extra: Optional[dict] = None):
        """Save projection to disk."""
        # Create directory if needed
        self.proj_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata on first save
        metadata_path = self.proj_dir / 'metadata.json'
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"[INFO] Saved metadata to {metadata_path}")

        # Prepare save dict
        save_dict = {
            'proj': projection.cpu(),
            'iter': self.iteration,
            'batch_size': projection.shape[0],
        }

        if batch_indices is not None:
            save_dict['batch_idx'] = batch_indices

        if extra is not None:
            for key, val in extra.items():
                if key in self._RESERVED_EXTRA_KEYS:
                    raise ValueError(f"extra key '{key}' is reserved")
                save_dict[key] = val

        # Save projection
        filename = f'proj_iter_{self.iteration:06d}.pt'
        save_path = self.proj_dir / filename
        torch.save(save_dict, save_path)

        print(f"[INFO] Saved projection [{projection.shape}] to {save_path}")

    def get_projection_metadata(self) -> dict:
        """Get metadata about the projection configuration."""
        return self.metadata.copy()

    # === Lifecycle helpers shared with the direct-driving loops (gradproj_lm / dvemb_lm) ===

    def prepare_gradients(self):
        """No-op for GradProjLora: projections are computed during the backward hooks."""
        pass

    def clear_gradients(self):
        """
        Clear gradients and cached data after optimizer step.

        This cleans up any cached activations or gradients.
        """
        # Clean up cached data in layers
        for layer_name, layer in self.matched_layers.items():
            if hasattr(layer, '_ghost_A_raw'):
                delattr(layer, '_ghost_A_raw')
            if hasattr(layer, '_ghost_grad_proj'):
                delattr(layer, '_ghost_grad_proj')

    def cleanup(self):
        """
        Cleanup and free resources.

        Projections are saved in collect_batch(); this just detaches hooks and
        releases the projection matrices.
        """
        # Detach all hooks
        self.detach()

        # Clear projection matrices to free memory
        self.projection_matrices.clear()

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