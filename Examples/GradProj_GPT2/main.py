"""Main entry point for computing gradient projections on Pile dataset."""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from local config
from config_file import parse_arguments, ProjectionConfig

# Import from main codebase
from dataloader import load_all_data
from model_setup import create_GPT_model
from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine


def main():
    """Main function for gradient projection computation."""
    print("=" * 80)
    print("Gradient Projection Computation for Pile Dataset")
    print("=" * 80)
    
    # Parse arguments and create config
    args = parse_arguments()
    config = ProjectionConfig(args)
    
    print(f"\nConfiguration:")
    print(f"  Architecture: {config.architecture}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Projection layers: {config.proj_layers}")
    print(f"  Projection rank: {config.proj_rank_total}")
    print(f"  Output directory: {config.proj_dir}")
    print(f"  Max samples: {config.max_samples if config.max_samples else 'All'}")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Set device
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Set precision
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_map[config.model_dtype]
    train_dtype = dtype_map[config.train_dtype]
    
    # Check bfloat16 support
    if config.model_dtype == 'bfloat16' or config.train_dtype == 'bfloat16':
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("Warning: bfloat16 not supported, falling back to float32")
            model_dtype = torch.float32
            train_dtype = torch.float32
    
    # Create autocast context
    ctx = torch.amp.autocast(device_type='cuda', dtype=train_dtype, enabled=(train_dtype != torch.float32))
    
    # Load dataset
    print("\n" + "=" * 40)
    print("Loading Pile dataset...")
    dataset = load_all_data()
    train_data = dataset['train']
    print(f"Loaded dataset with {len(train_data)} tokens")
    
    # Calculate number of samples
    total_samples = (len(train_data) - config.block_size) // config.block_size
    if config.max_samples:
        num_samples = min(config.max_samples, total_samples)
    else:
        num_samples = total_samples
    num_iterations = (num_samples + config.batch_size - 1) // config.batch_size
    
    print(f"Will process {num_samples} samples in {num_iterations} iterations")
    
    # Create model
    print("\n" + "=" * 40)
    print(f"Creating {config.architecture} model...")
    model = create_GPT_model(config)
    model = model.to(device)
    
    # Apply dtype conversion
    if model_dtype != torch.float32:
        model = model.to(model_dtype)
    
    model.eval()  # Use eval mode to disable dropout
    
    # Disable caching for gradient computation
    model.config.use_cache = False
    
    # Apply transformers support if available
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
        print("Applied transformers forward swapper")
    except ImportError:
        print("Warning: transformers_support not available, continuing without it")
    
    print(f"Model created with dtype {next(model.parameters()).dtype}")
    
    # Create projection engine
    print("\n" + "=" * 40)
    print("Initializing projection engine...")
    engine_config = {
        'proj_layers': config.proj_layers,
        'proj_rank_total': config.proj_rank_total,
        'proj_rank_min': config.proj_rank_min,
        'proj_seed': config.proj_seed,
        'proj_dtype': config.proj_dtype,
        'proj_dir': config.proj_dir,
        'proj_row_orthonormal': config.proj_row_orthonormal,
        'proj_save_interval': config.proj_save_interval,
        'include_embeddings': config.include_embeddings,
    }
    
    engine = GradProjLoraEngine(model, **engine_config)
    engine.attach()
    
    # Save run configuration
    run_config = {
        'timestamp': datetime.now().isoformat(),
        'architecture': config.architecture,
        'batch_size': config.batch_size,
        'block_size': config.block_size,
        'num_samples': num_samples,
        'num_iterations': num_iterations,
        'proj_layers': config.proj_layers,
        'proj_rank_total': config.proj_rank_total,
        'proj_rank_min': config.proj_rank_min,
        'proj_seed': config.proj_seed,
        'proj_dtype': config.proj_dtype,
        'model_dtype': config.model_dtype,
        'train_dtype': config.train_dtype,
        'device': str(device),
    }
    
    config_path = os.path.join(config.proj_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run configuration to {config_path}")
    
    # Import and run the projection loop
    from gradproj_loop import compute_projections
    
    print("\n" + "=" * 40)
    print("Starting projection computation...")
    compute_projections(
        model=model,
        engine=engine,
        dataset=dataset,
        config=config,
        device=device,
        ctx=ctx,
        num_iterations=num_iterations
    )
    
    # Cleanup
    engine.detach()
    print("\n" + "=" * 80)
    print(f"✓ Projection computation complete!")
    print(f"✓ Results saved to: {config.proj_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()