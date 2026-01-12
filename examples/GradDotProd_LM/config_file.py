"""Configuration management for the training script."""

import argparse
import os
import sys
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import build_result_dir


# Directory configurations
RESULTS_DIR = '/scratch/gpfs/PMITTAL/tianhao/GhostSuite/examples/GradDotProd_LM/results'


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
    parser.add_argument('--dynamic_val_batch', action='store_true',
                        help='Refresh validation batch every training step for GradDotProd')
    parser.add_argument('--log_grad_norms', action='store_true',
                        help='Record per-sample training gradient norms and aggregated validation gradient norm')
    parser.add_argument('--replay_run_dir', type=str, default=None,
                        help='Path to a previous GradDotProd run directory for filtered replay training')
    parser.add_argument('--replay_filter_metric', type=str, default='dot_product',
                        choices=['dot_product', 'cosine'],
                        help='Metric used to filter replay samples')
    parser.add_argument('--replay_filter_threshold', type=float, default=0.0,
                        help='Drop samples with metric below this threshold (default drops negatives)')
    parser.add_argument('--replay_rebatch_size', type=int, default=None,
                        help='Optional rebatch size for replayed samples; defaults to current batch_size')
    parser.add_argument('--replay_drop_last', action='store_true',
                        help='Drop the final incomplete batch when replay data is exhausted')
    parser.add_argument('--replay_shuffle', action='store_true',
                        help='Shuffle filtered replay samples (loads all filtered samples into memory)')
    parser.add_argument('--replay_shuffle_seed', type=int, default=None,
                        help='Seed for replay shuffling; defaults to --seed when enabled')

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
        if args.wandb_run_name:
            self.wandb_run_name = args.wandb_run_name
        else:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = (
                f"{args.method}_{args.architecture}_bs{args.batch_size}_lr{args.learning_rate}_{current_time}"
            )
        self.wandb_mode = args.wandb_mode
        self.dynamic_val_batch = args.dynamic_val_batch
        self.log_grad_norms = args.log_grad_norms
        self.replay_run_dir = args.replay_run_dir
        self.replay_filter_metric = args.replay_filter_metric
        self.replay_filter_threshold = args.replay_filter_threshold
        self.replay_rebatch_size = args.replay_rebatch_size or self.batch_size
        self.replay_drop_last = args.replay_drop_last
        self.replay_shuffle = args.replay_shuffle
        self.replay_shuffle_seed = args.replay_shuffle_seed
        if self.replay_shuffle and self.replay_shuffle_seed is None:
            self.replay_shuffle_seed = self.seed
        
        # Result directory setup (larger folder)
        self.result_folder = os.path.join(RESULTS_DIR, self.wandb_run_name)
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
