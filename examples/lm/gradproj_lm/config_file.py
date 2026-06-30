"""Configuration for standalone gradient projection computation on Pile dataset."""

import argparse
import os
import sys

# Add parent directories to path to import from main codebase
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)

# Anchored to this file's directory so each git worktree writes to its own
# results/ tree instead of a CWD-relative path. Mirrors graddotprod_lm.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def parse_arguments():
    """Parse command line arguments for gradient projection."""
    parser = argparse.ArgumentParser(description='Compute gradient projections for Pile dataset')
    
    # Model parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Tiny', 'GPT2-Small', 'GPT2-Medium', 'GPT2-Large'],
                       help='GPT2 model architecture')
    
    # Projection parameters
    parser.add_argument('--proj_layers', type=str, default='mlp,attn',
                       help='Comma-separated patterns for layers to project')
    parser.add_argument('--proj_rank_total', type=int, default=256,
                       help='Target total projection dimension per layer')
    parser.add_argument('--proj_rank_min', type=int, default=8,
                       help='Minimum dimension for k_i and k_o')
    parser.add_argument('--proj_seed', type=int, default=42,
                       help='Random seed for projection matrices')
    parser.add_argument('--proj_dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Data type for storing projections')
    parser.add_argument('--proj_row_orthonormal', action='store_true',
                       help='Use row-orthonormal projections')
    parser.add_argument('--include_embeddings', action='store_true',
                       help='Include embedding layers in projections')
    parser.add_argument('--proj_save_interval', type=int, default=1,
                       help='Save projections every N iterations')
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per-microbatch batch size for processing (small due to GPU memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of microbatches pooled per saved projection. The engine '
                            'buffers each microbatch and concatenates into a [N*batch_size, dim] '
                            'projection so all samples are captured, not just the last microbatch. '
                            'Do not pre-divide the loss by N (see GradProjLoraEngine.collect_microbatch).')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--block_size', type=int, default=1024,
                       help='Sequence length for GPT2')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data sampling')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                       help="Directory to save projections (default: this example's results/)")
    
    # Precision parameters
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Model data type')
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Training/gradient data type')
    
    # Data source
    parser.add_argument('--data_source', type=str, default='pile',
                       choices=['pile', 'synthetic'],
                       help="Data source: 'pile' (tokenized corpus on disk) or "
                            "'synthetic' (random tokens, for smoke tests)")

    # Misc parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    
    return parser.parse_args()


class ProjectionConfig:
    """Configuration class for gradient projection."""
    
    def __init__(self, args):
        self.args = args
        
        # Model configuration
        self.architecture = args.architecture
        
        # Projection configuration
        self.proj_layers = args.proj_layers
        self.proj_rank_total = args.proj_rank_total
        self.proj_rank_min = args.proj_rank_min
        self.proj_seed = args.proj_seed
        self.proj_dtype = args.proj_dtype
        self.proj_row_orthonormal = args.proj_row_orthonormal
        self.include_embeddings = args.include_embeddings
        self.proj_save_interval = args.proj_save_interval
        self.proj_dir = args.output_dir
        
        # Data source
        self.data_source = args.data_source

        # Processing configuration
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_samples = args.max_samples
        # Cap the sampling window to the model's block size (e.g. GPT2-Tiny=64)
        # so synthetic/real windows never exceed what the model can forward.
        self.block_size = args.block_size
        try:
            from shared.GPT2_configs import get_model_config
            model_block = get_model_config(self.architecture)['block_size']
            self.block_size = min(self.block_size, model_block)
        except (ImportError, ValueError, KeyError):
            pass
        self.seed = args.seed
        
        # Precision settings
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype
        
        # System settings
        self.device = args.device
        self.verbose = args.verbose

        # Setup output directory
        folder_name = f"proj_layers_{self.proj_layers}_rank_total_{self.proj_rank_total}_rank_min_{self.proj_rank_min}_seed_{self.proj_seed}_dtype_{self.proj_dtype}_row_on_{self.proj_row_orthonormal}_emb_{self.include_embeddings}"
        self.proj_dir = os.path.join(self.proj_dir, folder_name)
        
        # Create output directory
        os.makedirs(self.proj_dir, exist_ok=True)

    def __repr__(self):
        return f"ProjectionConfig(architecture={self.architecture}, batch_size={self.batch_size})"