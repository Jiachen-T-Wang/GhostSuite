"""
Plot L2 distance of gradient dot products vs projection dimension.

This script:
1. Loads gradient projections for different dimensions (16 to 2048)
2. Computes dot products between first sample and all others
3. Calculates L2 distance from the 2048-dimensional reference
4. Plots error vs dimensionality
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def load_gradient_projections(result_dir: Path, max_iters: int = None) -> torch.Tensor:
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


def main():
    # Configuration
    base_dir = Path("/scratch/gpfs/tw8948/GhostPub/GhostSuite/Examples/GradProj_GPT2/Results")
    ranks = [16, 32, 64, 128, 256, 512, 1024]  # Excluding 2048 due to memory constraints
    reference_rank = 1024  # Use 1024 as reference instead
    
    # For testing, use a smaller subset
    max_iters = 100  # Set to e.g., 100 for testing, None for full analysis
    
    # Storage for results
    dot_products_by_rank = {}
    num_ref = 50  # Number of reference samples
    
    # Load projections and compute dot products for each rank
    for rank in ranks:
        dir_name = f"proj_layers_mlp_rank_total_{rank}_rank_min_4_seed_9_dtype_bfloat16_row_on_False_emb_False"
        result_dir = base_dir / dir_name
        
        if not result_dir.exists():
            print(f"Warning: Directory not found for rank {rank}: {result_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing rank {rank}")
        print(f"{'='*60}")
        
        # Load projections
        projections = load_gradient_projections(result_dir, max_iters=max_iters)
        print(f"Loaded projections shape: {projections.shape}")
        
        # Compute dot products with multiple references
        print(f"Computing dot products with {num_ref} reference samples...")
        dot_products = compute_dot_products_multi_ref(projections, num_ref=num_ref)
        dot_products_by_rank[rank] = dot_products
        
        print(f"Dot products shape: {dot_products.shape}")
        print(f"Dot products mean stats - Mean: {dot_products.mean():.4f}, Std: {dot_products.std():.4f}")
    
    # Compute RMSE and relative errors from reference
    if reference_rank not in dot_products_by_rank:
        raise ValueError(f"Reference rank {reference_rank} not found in results")
    
    reference_dots = dot_products_by_rank[reference_rank]  # [num_ref, num_samples - num_ref]
    
    rmse_values = []
    relative_errors = []
    rmse_per_ref = []  # Store RMSE for each reference point
    plot_ranks = [r for r in ranks if r != reference_rank]
    
    print(f"\n{'='*60}")
    print(f"Computing errors from reference (rank {reference_rank})")
    print(f"Using {reference_dots.shape[0]} reference samples")
    print(f"{'='*60}")
    
    for rank in plot_ranks:
        if rank not in dot_products_by_rank:
            continue
        
        dots = dot_products_by_rank[rank]  # [num_ref, num_samples - num_ref]
        
        # Ensure same dimensions
        min_samples = min(dots.shape[1], reference_dots.shape[1])
        dots = dots[:, :min_samples]
        ref = reference_dots[:, :min_samples]
        
        # Compute RMSE for each reference point
        rmse_per_ref_point = []
        for i in range(dots.shape[0]):
            mse_i = torch.mean((dots[i] - ref[i]) ** 2)
            rmse_i = torch.sqrt(mse_i).item()
            rmse_per_ref_point.append(rmse_i)
        
        # Average RMSE across all reference points
        avg_rmse = np.mean(rmse_per_ref_point)
        std_rmse = np.std(rmse_per_ref_point)
        rmse_values.append(avg_rmse)
        rmse_per_ref.append(rmse_per_ref_point)
        
        # Compute relative error (normalized by reference RMS)
        ref_rms = torch.sqrt(torch.mean(ref ** 2)).item()
        rel_error = avg_rmse / ref_rms if ref_rms > 0 else float('inf')
        relative_errors.append(rel_error)
        
        print(f"Rank {rank:4d}: Avg RMSE = {avg_rmse:.4f} (±{std_rmse:.4f}), Relative error = {rel_error:.4%}")
    
    # Create the plot
    print(f"\n{'='*60}")
    print("Creating plots...")
    print(f"{'='*60}")
    
    # Plot 1: RMSE vs Dimension
    plt.figure(figsize=(10, 6))
    plt.plot(plot_ranks, rmse_values, 'b-o', linewidth=2, markersize=8)
    
    # Formatting
    plt.xlabel('Projection Dimension', fontsize=12)
    plt.ylabel(f'RMSE', fontsize=12)
    plt.title('Gradient Dot Product RMSE vs Projection Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Use log scale for x-axis
    plt.xscale('log', base=2)
    plt.xticks(plot_ranks, [str(r) for r in plot_ranks])
    
    # Plot 2: Relative Error vs Dimension
    plt.figure(figsize=(10, 6))
    plt.plot(plot_ranks, [re * 100 for re in relative_errors], 'g-s', linewidth=2, markersize=8)
    plt.xlabel('Projection Dimension', fontsize=12)
    plt.ylabel(f'Relative Error (%) of Average Dot-Product', fontsize=12)
    plt.title('Gradient Dot Product Relative Error vs Projection Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.xticks(plot_ranks, [str(r) for r in plot_ranks])
    
    # Plot 3: Combined plot with RMSE log-log and Relative Error with log x-axis only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE log-log
    ax1.loglog(plot_ranks, rmse_values, 'b-o', linewidth=2, markersize=8, base=2)
    ax1.set_xlabel('Projection Dimension', fontsize=12)
    ax1.set_ylabel(f'RMSE', fontsize=12)
    ax1.set_title('RMSE (Log-Log Scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(plot_ranks)
    ax1.set_xticklabels([str(r) for r in plot_ranks])
    
    # Relative Error with log x-axis only (linear y-axis)
    ax2.semilogx(plot_ranks, [re * 100 for re in relative_errors], 'g-s', linewidth=2, markersize=8, base=2)
    ax2.set_xlabel('Projection Dimension', fontsize=12)
    ax2.set_ylabel(f'Relative Error (%) of Average Dot-Product', fontsize=12)
    ax2.set_title('Relative Error (Log X-Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(plot_ranks)
    ax2.set_xticklabels([str(r) for r in plot_ranks])
    
    plt.suptitle('Gradient Dot Product Error Analysis', fontsize=14, fontweight='bold')
    
    # Save plots
    output_dir = Path("/scratch/gpfs/tw8948/GhostPub/GhostSuite/Examples/GradProj_GPT2/Plots")
    
    # Save RMSE plot
    plt.figure(1)
    plt.tight_layout()
    # plt.savefig(output_dir / "rmse_vs_dimension.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "rmse_vs_dimension.pdf", bbox_inches='tight')
    print(f"Saved RMSE plot to {output_dir}/rmse_vs_dimension.{{png,pdf}}")
    
    # Save Relative Error plot
    plt.figure(2)
    plt.tight_layout()
    # plt.savefig(output_dir / "relative_error_vs_dimension.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "relative_error_vs_dimension.pdf", bbox_inches='tight')
    print(f"Saved relative error plot to {output_dir}/relative_error_vs_dimension.{{png,pdf}}")
    
    # Save combined log-log plot
    plt.figure(3)
    plt.tight_layout()
    # plt.savefig(output_dir / "error_analysis_loglog.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "error_analysis_loglog.pdf", bbox_inches='tight')
    print(f"Saved combined log-log plot to {output_dir}/error_analysis_loglog.{{png,pdf}}")
    
    # Show plots if in interactive mode
    plt.show()
    
    # Save numerical results
    results = {
        'ranks': plot_ranks,
        'rmse_values': rmse_values,
        'relative_errors': [re * 100 for re in relative_errors],  # Save as percentages
        'reference_rank': reference_rank,
        'num_ref_samples': reference_dots.shape[0],
        'num_test_samples': reference_dots.shape[1]
    }
    
    import json
    with open(output_dir / "error_vs_dimension_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved numerical results to {output_dir}/error_vs_dimension_results.json")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()