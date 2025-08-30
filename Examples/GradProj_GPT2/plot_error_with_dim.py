"""
Plot L2 distance of gradient dot products vs projection dimension.

Adds a CLI and flexible reference selection.

What it does:
- Discovers result subfolders in --results_dir by --results_pattern (regex or glob via --pattern_type)
- Validates subfolder naming only differs by rank_total_{K}
- Loads projections per rank and computes multi-reference dot-products
- Compares errors to a reference specified by --reference:
  * "rank=NNN"              -> use projections at that rank as reference
  * "full"                  -> use exact full-model gradients saved by compute_full_gradients.py
  * "full_layers"           -> use exact gradients restricted to GradProj layers
  * "naive_proj_layers=NNN" -> rebuild P using metadata+seed, apply to exact per-layer grads
- Saves .pdf plots and a JSON with settings+metrics under Plots/
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Pattern, Optional
import re
import argparse
import fnmatch

# Local imports for rebuilding projections when needed
# Ensure project root is on path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ghostEngines.gradProjection.projection_utils import get_projection_initializer


def load_gradient_projections(result_dir: Path, max_iters: Optional[int] = None) -> torch.Tensor:
    """
    Load all gradient projections from a result directory.
    
    Args:
        result_dir: Directory containing proj_iter_*.pt files
        max_iters: Maximum number of iterations to load (None for all)
    
    Returns:
        Tensor of shape [num_samples, proj_dim]
    """
    proj_files = sorted(result_dir.glob("proj_iter_*.pt"))
    
    if max_iters is not None:
        proj_files = proj_files[:max_iters]
    
    all_projections = []
    
    print(f"Loading {len(proj_files)} files from {result_dir.name}...")
    
    for proj_file in tqdm(proj_files, desc="Loading projections"):
        data = torch.load(proj_file, map_location='cpu')
        proj = data['proj'].float()  # Convert from bfloat16 to float32
        all_projections.append(proj)
    
    # Stack all projections: [num_iters, batch_size, proj_dim] -> [num_samples, proj_dim]
    all_projections = torch.cat(all_projections, dim=0)
    
    return all_projections


def compute_dot_products_multi_ref(projections: torch.Tensor, num_ref: int = 10) -> torch.Tensor:
    """
    Compute dot products between first num_ref samples and all others.
    
    Args:
        projections: Tensor of shape [num_samples, proj_dim]
        num_ref: Number of reference samples to use
    
    Returns:
        Tensor of shape [num_ref, num_samples - num_ref] containing dot products
    """
    ref_projs = projections[:num_ref]  # [num_ref, proj_dim]
    rest_proj = projections[num_ref:]  # [num_samples - num_ref, proj_dim]
    
    # Compute all dot products: [num_ref, proj_dim] @ [proj_dim, num_samples - num_ref]
    dot_products = torch.matmul(ref_projs, rest_proj.T)
    
    return dot_products


def discover_rank_dirs(results_dir: Path, pattern: Pattern[str]) -> Dict[int, Path]:
    """
    Find result subdirectories whose names match the regex and extract rank_total.
    Ensures all matched names differ only in the 'rank_total_\d+' token.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    matches = []
    for child in results_dir.iterdir():
        if child.is_dir() and pattern.search(child.name):
            matches.append(child)
    if not matches:
        raise ValueError(f"No subdirectories in {results_dir} matched pattern: {pattern.pattern}")

    # Extract ranks and validate naming
    rank_re = re.compile(r"rank_total_(\d+)")
    template_key = None
    rank_dirs: Dict[int, Path] = {}
    for d in matches:
        m = rank_re.search(d.name)
        if not m:
            raise ValueError(f"Matched directory lacks rank_total token: {d.name}")
        rank = int(m.group(1))
        # Normalize name by replacing rank value with a placeholder
        normalized = rank_re.sub("rank_total_{R}", d.name)
        if template_key is None:
            template_key = normalized
        elif normalized != template_key:
            raise ValueError(
                f"Directories differ beyond rank_total: '{d.name}' vs template '{template_key}'"
            )
        rank_dirs[rank] = d
    if len(rank_dirs) < 2:
        print("[WARN] Fewer than 2 rank directories matched; plots may be uninformative.")
    return dict(sorted(rank_dirs.items()))


def load_full_gradients(fullgrad_dir: Path, max_iters: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
    """
    Load exact full-model gradients saved by compute_full_gradients.py.

    Returns:
        grads: [num_samples, total_param_dim] float32
        meta:  metadata dict from fullgrad_meta.json
    """
    grad_files = sorted(fullgrad_dir.glob("fullgrad_iter_*.pt"))
    if max_iters is not None:
        grad_files = grad_files[:max_iters]
    if not grad_files:
        raise ValueError(f"No full gradient files found in {fullgrad_dir}")

    grads = []
    for gf in tqdm(grad_files, desc="Loading full grads"):
        data = torch.load(gf, map_location='cpu')
        g = data['grad'].float()
        grads.append(g)
    grads = torch.stack(grads, dim=0)

    meta_path = fullgrad_dir / 'fullgrad_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return grads, meta


def extract_layer_restricted(grads: torch.Tensor, full_meta: dict, proj_meta: dict) -> torch.Tensor:
    """
    Restrict full gradient vectors to only the weight tensors of GradProj layers.

    Uses full_meta['param_slices'] mapping {name: {start, end}} and
    proj_meta['layers'][i]['name'] (.weight appended) to build an index mask.
    """
    param_slices = full_meta['param_slices']
    indices: List[Tuple[int, int]] = []
    for layer in proj_meta['layers']:
        pname = f"{layer['name']}.weight"
        if pname not in param_slices:
            raise KeyError(f"Param slice not found for {pname}")
        s = param_slices[pname]
        indices.append((int(s['start']), int(s['end'])))
    # Build a view by concatenating slices for each sample
    parts = []
    for start, end in indices:
        parts.append(grads[:, start:end])
    return torch.cat(parts, dim=1)


def rebuild_naive_projection_from_full(grads_layer_restricted: torch.Tensor,
                                       proj_meta: dict,
                                       proj_seed: int,
                                       method: str) -> torch.Tensor:
    """
    For each layer: reshape full dW, apply P_o @ dW @ P_i^T using metadata dims and seed,
    then flatten and concatenate across layers. grads_layer_restricted concatenation order
    must match proj_meta['layers'] order with only .weight tensors.
    """
    init_fn = get_projection_initializer(method)
    device = torch.device('cpu')

    outputs = []
    offset = 0
    for li, layer in enumerate(proj_meta['layers']):
        n_o = int(layer['n_o'])
        n_i = int(layer['n_i'])
        k_i = int(layer['k_i'])
        k_o = int(layer['k_o'])
        numel = n_o * n_i
        g_slice = grads_layer_restricted[:, offset:offset + numel]
        offset += numel
        # Reshape to [B, n_o, n_i]
        B = g_slice.shape[0]
        gW = g_slice.reshape(B, n_o, n_i)

        # Rebuild P matrices with same seeding scheme as engine
        layer_seed = proj_seed + li
        P_i = init_fn(k_i, n_i, dtype=torch.float32, device=device, seed=layer_seed)
        P_o = init_fn(k_o, n_o, dtype=torch.float32, device=device, seed=layer_seed + 1000)

        # Apply naive projection per-sample: [B, k_o, k_i]
        # (P_o @ gW @ P_i^T)
        # Compute via einsum: first left multiply, then right multiply
        left = torch.einsum('oi,bjk->boj', P_o, gW)        # [B, k_o, n_i]
        proj = torch.einsum('boj,ij->boi', left, P_i)      # [B, k_o, k_i]
        outputs.append(proj.reshape(B, k_o * k_i))

    return torch.cat(outputs, dim=1)


def build_output_basename(rank_dirs: Dict[int, Path], reference_label: str) -> str:
    # Use the common normalized template as a base name
    sample_name = next(iter(rank_dirs.values())).name
    base = re.sub(r"rank_total_\d+", "rank_total_ALL", sample_name)
    return f"{base}__ref_{reference_label}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze gradient projection error vs dimension")
    p.add_argument('--results_dir', type=str, required=True,
                  help='Directory containing result subfolders')
    p.add_argument('--results_pattern', type=str, required=True,
                  help='Pattern to match result subfolder names that only differ by rank_total_K')
    p.add_argument('--pattern_type', type=str, default='glob', choices=['regex', 'glob'],
                  help='Interpretation of --results_pattern: regex (default) or shell-style glob')
    p.add_argument('--num_ref', type=int, default=50, help='Number of reference samples')
    p.add_argument('--max_iters', type=int, default=100, help='Max iteration files to load per dir')
    p.add_argument('--reference', type=str, default='rank=1024',
                  help='Reference mode: rank=NNN | full | full_layers | naive_proj_layers=NNN')
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.results_dir)
    # Build matcher pattern (regex or translated glob)
    if args.pattern_type == 'glob':
        translated = fnmatch.translate(args.results_pattern)
        pattern = re.compile(translated)
    else:
        try:
            pattern = re.compile(args.results_pattern)
        except re.error as e:
            # Provide a clearer error if the user likely provided a glob by mistake
            if any(ch in args.results_pattern for ch in ['*', '?', '[', ']']):
                raise ValueError(
                    "Invalid regex in --results_pattern. It looks like a shell-style glob. "
                    "Either escape regex metacharacters or pass --pattern_type glob.\n"
                    f"Pattern: {args.results_pattern}\nOriginal regex error: {e}"
                )
            raise

    # Discover rank dirs and extract ranks
    rank_dirs = discover_rank_dirs(base_dir, pattern)
    ranks = list(rank_dirs.keys())

    # Load projections and compute dot products per rank
    dot_products_by_rank: Dict[int, torch.Tensor] = {}
    for rank, result_dir in rank_dirs.items():
        print(f"\n{'='*60}\nProcessing rank {rank}\n{'='*60}")
        projections = load_gradient_projections(result_dir, max_iters=args.max_iters)
        print(f"Loaded projections shape: {projections.shape}")
        print(f"Computing dot products with {args.num_ref} reference samples...")
        dots = compute_dot_products_multi_ref(projections, num_ref=args.num_ref)
        dot_products_by_rank[rank] = dots
        print(f"Dot products shape: {dots.shape}")
        print(f"Dot products mean stats - Mean: {dots.mean():.4f}, Std: {dots.std():.4f}")


    num_data_loaded = projections.shape[0]

    # Prepare reference
    ref_mode = args.reference
    ref_label = ref_mode
    reference_dots = None

    if ref_mode.startswith('rank='):
        reference_rank = int(ref_mode.split('=', 1)[1])
        if reference_rank not in dot_products_by_rank:
            raise ValueError(f"Reference rank {reference_rank} not among discovered ranks: {list(dot_products_by_rank)}")
        reference_dots = dot_products_by_rank[reference_rank]

    else:
        # Need full gradients and metadata from one of the rank dirs (to get layer list)
        # Infer fullgrad directory name convention: sibling folder starting with 'fullgrads'
        fullgrad_cands = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith('fullgrads')]
        if not fullgrad_cands:
            raise FileNotFoundError("No 'fullgrads*' directory found in results_dir; run compute_full_gradients.py first.")
        fullgrad_dir = sorted(fullgrad_cands)[0]

        # We need to load num_data_loaded full gradients because the fullgrads are saved with batch size = 1. 
        full_grads, full_meta = load_full_gradients(fullgrad_dir, max_iters=num_data_loaded)

        print(f"Loaded full gradients shape: {full_grads.shape}")

        # Load projection metadata from any rank dir (assume constant except rank)
        any_rank_dir = next(iter(rank_dirs.values()))
        meta_path = any_rank_dir / 'metadata.json'
        with open(meta_path, 'r') as f:
            proj_meta = json.load(f)

        if ref_mode == 'full':
            reference_dots = compute_dot_products_multi_ref(full_grads, num_ref=args.num_ref)
        elif ref_mode == 'full_layers':
            full_lr = extract_layer_restricted(full_grads, full_meta, proj_meta)
            reference_dots = compute_dot_products_multi_ref(full_lr, num_ref=args.num_ref)
        elif ref_mode.startswith('naive_proj_layers='):
            # Rebuild naive projections at the specified rank using full gradients
            # Note: proj_meta already contains k_i,k_o per layer for its own rank.
            # To match a different rank, require that rank_dirs includes that rank so we can reuse its metadata.
            target_rank = int(ref_mode.split('=', 1)[1])
            if target_rank not in rank_dirs:
                raise ValueError(f"naive_proj_layers requires a discovered rank directory for rank={target_rank}")
            with open(rank_dirs[target_rank] / 'metadata.json', 'r') as f:
                target_meta = json.load(f)
            # Restrict full grads to layer weights
            full_lr = extract_layer_restricted(full_grads, full_meta, target_meta)
            proj_seed = int(target_meta['proj_seed'])
            method = target_meta.get('proj_method', 'gaussian')
            naive_proj = rebuild_naive_projection_from_full(full_lr, target_meta, proj_seed, method)
            reference_dots = compute_dot_products_multi_ref(naive_proj, num_ref=args.num_ref)
        else:
            raise ValueError(f"Unknown --reference option: {ref_mode}")

    # Compute RMSE and relative errors from reference
    rmse_values: List[float] = []
    relative_errors: List[float] = []
    plot_ranks = [r for r in ranks if not (ref_mode.startswith('rank=') and r == int(ref_mode.split('=')[1]))]

    print(f"\n{'='*60}")
    print(f"Computing errors from reference ({ref_label})")
    print(f"Using {reference_dots.shape[0]} reference samples")
    print(f"{'='*60}")


    print(f"reference_dots shape: {reference_dots.shape}")
    print(f"dot_products_by_rank[1024] shape: {dot_products_by_rank[1024].shape}")

    for rank in plot_ranks:
        dots = dot_products_by_rank[rank]
        # Align lengths
        min_samples = min(dots.shape[1], reference_dots.shape[1])
        dots = dots[:, :min_samples]
        ref = reference_dots[:, :min_samples]

        # RMSE per reference point
        mse = torch.mean((dots - ref) ** 2, dim=1)
        rmse = torch.sqrt(mse).cpu().numpy()  # [num_ref]
        avg_rmse = float(np.mean(rmse))
        std_rmse = float(np.std(rmse))
        rmse_values.append(avg_rmse)

        ref_rms = float(torch.sqrt(torch.mean(ref ** 2)).item())
        rel_error = avg_rmse / ref_rms if ref_rms > 0 else float('inf')
        relative_errors.append(rel_error)

        print(f"Rank {rank:4d}: Avg RMSE = {avg_rmse:.4f} (±{std_rmse:.4f}), Relative error = {rel_error:.4%}")

    # Prepare output dir and filenames
    output_dir = base_dir.parent / 'Plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_output_basename(rank_dirs, reference_label=ref_label.replace('=', '_'))

    # Plot 1: RMSE vs Dimension
    plt.figure(figsize=(10, 6))
    plt.plot(plot_ranks, rmse_values, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Projection Dimension', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Gradient Dot Product RMSE vs Projection Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks(plot_ranks, [str(r) for r in plot_ranks])
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}__rmse_vs_dimension.pdf", bbox_inches='tight')

    # Plot 2: Relative Error vs Dimension
    plt.figure(figsize=(10, 6))
    plt.plot(plot_ranks, [re * 100 for re in relative_errors], 'g-s', linewidth=2, markersize=8)
    plt.xlabel('Projection Dimension', fontsize=12)
    plt.ylabel('Relative Error (%) of Average Dot-Product', fontsize=12)
    plt.title('Gradient Dot Product Relative Error vs Projection Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks(plot_ranks, [str(r) for r in plot_ranks])
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}__relative_error_vs_dimension.pdf", bbox_inches='tight')

    # Plot 3: Combined
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.loglog(plot_ranks, rmse_values, 'b-o', linewidth=2, markersize=8, base=2)
    ax1.set_xlabel('Projection Dimension', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE (Log-Log Scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(plot_ranks)
    ax1.set_xticklabels([str(r) for r in plot_ranks])

    ax2.semilogx(plot_ranks, [re * 100 for re in relative_errors], 'g-s', linewidth=2, markersize=8, base=2)
    ax2.set_xlabel('Projection Dimension', fontsize=12)
    ax2.set_ylabel('Relative Error (%) of Average Dot-Product', fontsize=12)
    ax2.set_title('Relative Error (Log X-Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(plot_ranks)
    ax2.set_xticklabels([str(r) for r in plot_ranks])
    plt.suptitle('Gradient Dot Product Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}__error_analysis_loglog.pdf", bbox_inches='tight')

    # Save numerical results
    results = {
        'ranks': plot_ranks,
        'rmse_values': rmse_values,
        'relative_errors_percent': [re * 100 for re in relative_errors],
        'reference': ref_mode,
        'num_ref_samples': int(reference_dots.shape[0]),
        'num_test_samples': int(reference_dots.shape[1]),
        'results_pattern': args.results_pattern,
        'max_iters': args.max_iters,
    }
    with open(output_dir / f"{base_name}__error_vs_dimension_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved outputs under {output_dir} with base '{base_name}' (PDF + JSON)")
    print(f"\n{'='*60}\nAnalysis complete!\n{'='*60}")


if __name__ == "__main__":
    main()
