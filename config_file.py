"""Configuration management for the training script."""

import argparse
import os

from utils import build_result_dir


# Directory configurations
RESULTS_DIR = '/scratch/gpfs/tw8948/ghostTest/Results'

# Dataset directories
PILE_DATA_DIR = '/scratch/gpfs/tw8948/pile_tokenized'
LLAVA_DATASET_DIR = '/scratch/gpfs/tw8948/llava_dataset'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='In-Run Data Shapley score computation.')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='Regular',
                        choices=['Regular', 'GradDotProd'])
    
    # Architecture parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'Pythia-410M', 'LLaVA-7B', 'LLaVA-13B'])
    
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
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'], 
                       help='Model data type')
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'], 
                       help='Training data type')

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
        # Note: we never use float16 for stability
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
        
        # Result directory setup (larger folder)
        self.result_folder = RESULTS_DIR
        self.setup_result_directories()
    
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


