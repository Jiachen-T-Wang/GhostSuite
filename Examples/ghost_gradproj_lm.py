"""
Example demonstrating gradient projection on GPT-2 language model.
Computes and stores per-sample projected gradients for transformer layers.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine


def load_gpt2_model(model_name='gpt2'):
    """Load GPT-2 model and tokenizer."""
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        raise ImportError("transformers library required. Install with: pip install transformers")
        
    print(f"Loading {model_name} model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def prepare_sample_data(tokenizer, batch_size=4, seq_length=128):
    """Prepare sample text data for gradient computation."""
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have achieved remarkable results in various tasks.",
        "Transformer architectures have revolutionized NLP applications.",
        "Gradient descent is a fundamental optimization algorithm.",
        "Neural networks can learn complex patterns from data.",
        "Artificial intelligence is advancing rapidly across domains.",
    ]
    
    # Take batch_size texts
    texts = texts[:batch_size]
    
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=seq_length
    )
    
    # Create labels (shifted input ids for language modeling)
    labels = inputs['input_ids'].clone()
    labels[inputs['attention_mask'] == 0] = -100  # Ignore padding in loss
    
    return inputs, labels


def run_gpt2_projection(args):
    """Run gradient projection on GPT-2 model."""
    print("\n=== GPT-2 Gradient Projection ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_gpt2_model(args.model_name)
    model = model.to(device)
    model.eval()  # Use eval mode to avoid dropout

    # Disable caching for gradient computation
    model.config.use_cache = False

    # Apply transformers support if available
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
        print("Applied transformers forward swapper")
    except ImportError:
        print("Warning: transformers_support not available, continuing without it")
        
    # Create projection engine
    engine_config = {
        'proj_layers': args.proj_layers,
        'proj_rank_total': args.proj_rank_total,
        'proj_rank_min': args.proj_rank_min,
        'proj_seed': args.proj_seed,
        'proj_dtype': args.proj_dtype,
        'proj_dir': args.proj_dir,
        'proj_save_interval': args.proj_save_interval,
        'include_embeddings': args.include_embeddings,
    }
    
    print("\nInitializing projection engine...")
    engine = GradProjLoraEngine(model, **engine_config)
    engine.attach()
    
    # Process batches
    for batch_idx in range(args.num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{args.num_batches}")
        
        # Prepare data
        inputs, labels = prepare_sample_data(tokenizer, args.batch_size)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Collect projected gradients
        batch_indices = list(range(batch_idx * args.batch_size, 
                                 (batch_idx + 1) * args.batch_size))
        projections = engine.collect_batch(batch_indices)
        
        print(f"  Projection shape: {projections.shape}")
        print(f"  Projection dtype: {projections.dtype}")
        
    # Detach engine
    engine.detach()
    
    print(f"\n✓ Projections saved to: {args.proj_dir}")
    
    # Show how to load and use projections
    print("\n=== Example: Loading saved projections ===")
    show_projection_usage(args.proj_dir)


def show_projection_usage(proj_dir):
    """Show example of loading and using saved projections."""
    import json
    import glob
    
    proj_path = Path(proj_dir)
    
    # Load metadata
    metadata_path = proj_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Projection metadata:")
        print(f"  Total dimension: {metadata['total_proj_dim']}")
        print(f"  Number of layers: {len(metadata['layers'])}")
        print(f"  Projection method: {metadata['proj_method']}")
        print(f"  Data type: {metadata['proj_dtype']}")
        
        # Show first few layers
        print("\n  Layer configurations:")
        for layer_info in metadata['layers'][:3]:
            print(f"    - {layer_info['name']}: k_i={layer_info['k_i']}, k_o={layer_info['k_o']}")
            
    # Load projection files
    proj_files = sorted(glob.glob(str(proj_path / 'proj_iter_*.pt')))
    if proj_files:
        print(f"\nFound {len(proj_files)} projection files")
        
        # Load first file as example
        first_proj = torch.load(proj_files[0], map_location='cpu')
        print(f"\nFirst projection file content:")
        print(f"  Shape: {first_proj['proj'].shape}")
        print(f"  Iteration: {first_proj['iter']}")
        print(f"  Batch size: {first_proj['batch_size']}")
        
        if len(proj_files) > 1:
            # Show how to compute similarities
            print("\n=== Computing pairwise similarities ===")
            
            # Load first two batches
            proj1 = torch.load(proj_files[0], map_location='cpu')['proj']
            proj2 = torch.load(proj_files[1], map_location='cpu')['proj'] if len(proj_files) > 1 else proj1
            
            # Compute similarities
            sims = proj1 @ proj2.t()
            print(f"Similarity matrix shape: {sims.shape}")
            print(f"Similarity range: [{sims.min():.4f}, {sims.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='GPT-2 Gradient Projection Example')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='GPT-2 model name (gpt2, gpt2-medium, etc.)')
    
    # Projection configuration (required)
    parser.add_argument('--proj_layers', type=str, default='attn.c_attn,mlp.c_fc',
                       help='Comma-separated layer name patterns')
    parser.add_argument('--proj_rank_total', type=int, default=256,
                       help='Target total projection dimension per layer')
    parser.add_argument('--proj_rank_min', type=int, default=8,
                       help='Minimum dimension for k_i and k_o')
    parser.add_argument('--proj_seed', type=int, default=42,
                       help='Random seed for projection matrices')
    parser.add_argument('--proj_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Data type for storing projections')
    parser.add_argument('--proj_dir', type=str, default='./Examples/grad_proj_lm',
                       help='Directory to save projections')
    parser.add_argument('--proj_save_interval', type=int, default=1,
                       help='Save projections every N iterations')
    parser.add_argument('--include_embeddings', action='store_true',
                       help='Include embedding layers in projection')
    
    # Data configuration
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches to process')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.proj_layers:
        raise ValueError("--proj_layers is required")
    if args.proj_rank_total <= 0:
        raise ValueError("--proj_rank_total must be positive")
    if args.proj_rank_min <= 0:
        raise ValueError("--proj_rank_min must be positive")
        
    # Run projection
    run_gpt2_projection(args)


if __name__ == '__main__':
    main()