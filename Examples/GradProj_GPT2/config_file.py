"""Configuration for standalone gradient projection computation on Pile dataset."""

import argparse
import os
import sys

# Add parent directories to path to import from main codebase
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Import from main config (avoiding name conflict)
import importlib.util
spec = importlib.util.spec_from_file_location("main_config", os.path.join(parent_dir, "config_file.py"))
main_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_config)
PILE_DATA_DIR = main_config.PILE_DATA_DIR


def parse_arguments():
    """Parse command line arguments for gradient projection."""
    parser = argparse.ArgumentParser(description='Compute gradient projections for Pile dataset')
    
    # Model parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large'],
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
                       help='Batch size for processing (small due to GPU memory)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--block_size', type=int, default=1024,
                       help='Sequence length for GPT2')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data sampling')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./Results',
                       help='Directory to save projections')
    
    # Precision parameters
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Model data type')
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Training/gradient data type')
    
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
        
        # Processing configuration
        self.batch_size = args.batch_size
        self.max_samples = args.max_samples
        self.block_size = args.block_size
        self.seed = args.seed
        
        # Precision settings
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype
        
        # System settings
        self.device = args.device
        self.verbose = args.verbose
        
        # Dataset configuration (from main config)
        self.pile_data_dir = PILE_DATA_DIR

        # Setup output directory
        folder_name = f"proj_layers_{self.proj_layers}_rank_total_{self.proj_rank_total}_rank_min_{self.proj_rank_min}_seed_{self.proj_seed}_dtype_{self.proj_dtype}_row_on_{self.proj_row_orthonormal}_emb_{self.include_embeddings}"
        self.proj_dir = os.path.join(self.proj_dir, folder_name)
        
        # Create output directory
        os.makedirs(self.proj_dir, exist_ok=True)
        
    def get_model_config(self):
        """Get GPT2 model configuration."""
        config_GPT = {
            'GPT2-Small': {
                'n_layer': 12,
                'n_head': 12,
                'n_embd': 768,
                'block_size': 1024,
                'vocab_size': 50304,
            },
            'GPT2-Medium': {
                'n_layer': 24,
                'n_head': 16,
                'n_embd': 1024,
                'block_size': 1024,
                'vocab_size': 50304,
            },
            'GPT2-Large': {
                'n_layer': 36,
                'n_head': 20,
                'n_embd': 1280,
                'block_size': 1024,
                'vocab_size': 50304,
            }
        }
        
        if self.architecture not in config_GPT:
            raise ValueError(f"Unknown GPT architecture: {self.architecture}")
            
        return config_GPT[self.architecture]
    
    def __repr__(self):
        return f"ProjectionConfig(architecture={self.architecture}, batch_size={self.batch_size})"