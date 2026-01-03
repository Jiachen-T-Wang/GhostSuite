"""Configuration management for the training script."""

import argparse
import os
import sys
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import build_result_dir


# Directory configurations
RESULTS_DIR = '/scratch/gpfs/PMITTAL/tianhao/GhostSuite/Examples/GradDotProd_LM/results'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='In-Run Data Shapley score computation.')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='Regular', choices=['Regular', 'GradDotProd'])
    
    # Architecture parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'LLaVA-7B', 'LLaVA-13B'])
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    
    # Dataset parameters
    parser.add_argument('--train_set', type=str, default='pile')
    parser.add_argument('--val_set', type=str, default='pile', help='Validation dataset name; currently not used')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--eval_iter', type=int, default=20)
    parser.add_argument('--eval_bs', type=int, default=16)

    # In-Run Shapley parameters
    parser.add_argument('--dot_prod_save_interval', type=int, default=10)
    
    # Precision parameters
    parser.add_argument('--model_dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'], 
                       help='Model data type')
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'], 
                       help='Training data type')

    # WandB logging
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='GhostSuite', help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode (online, offline, disabled)')
    parser.add_argument('--wandb_dir', type=str, default=None, help='Directory for Weights & Biases files')

    return parser.parse_args()




class TrainingConfig:
    """Training configuration class."""
    
    def __init__(self, args):

        self.args = args

        # Defer the model config to a separate function
        self.architecture = args.architecture
        
        # Training hyperparameters
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.learning_rate = args.learning_rate
        self.min_lr = self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed
        
        # Optimizer settings (currently just assume using AdamW)
        self.optimizer = args.optimizer
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        self.lr_decay_iters = 10000
        self.decay_lr = True
        
        # System settings
        self.device = 'cuda'
        self.compile = False
        self.backend = 'nccl'

        # Precision settings
        # To train LLAVA models, we use bfloat16 for both model and training
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype

        # Gradient accumulation
        self.full_batch_size = args.batch_size
        self.gradient_accumulation_steps = 1
        
        # Evaluation settings
        self.eval_iters = args.eval_iter
        self.eval_interval = args.eval_interval
        self.eval_bs = args.eval_bs
        self.dot_prod_save_interval = args.dot_prod_save_interval

        if self.dot_prod_save_interval is None:
            self.dot_prod_save_interval = self.eval_interval
        
        # Method-specific settings
        self.method = args.method
        self.use_wandb = args.wandb
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name
        self.wandb_mode = args.wandb_mode
        
        # Result directory setup (larger folder)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_folder = os.path.join(RESULTS_DIR, current_time)
        self.setup_result_directories()
        self.wandb_dir = args.wandb_dir or self.result_dir
    
    def _is_bf16_supported(self):
        """Check if bfloat16 is supported."""
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    def setup_result_directories(self):

        # Create result folder if it doesn't exist
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
            print(f"Results folder '{self.result_folder}' was created.")

        # Create specific result directory for this run
        self.result_dir = build_result_dir(self.result_folder, self.method, self.args)
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print(f"Results directory for this specific run '{self.result_dir}' was created.")
    
    def get_result_file_path(self):
        """Get the result file path for storing training statistics."""
        result_dir = self.result_dir
        return os.path.join(result_dir + '_results.json')

