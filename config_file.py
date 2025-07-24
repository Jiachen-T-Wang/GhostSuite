"""Configuration management for the training script."""

import argparse
import os


# Replace with your own data and result root directories
DATA_ROOT = "/scratch/gpfs/tw8948/GhostSuite/pile-bin"
RESULT_ROOT = "/scratch/gpfs/tw8948/InRunResult"


def get_model_config(architecture):
    """Get model configuration based on architecture name."""
    configs = {
        'GPT2-Small': {
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'block_size': 1024,
        },
        'GPT2-Medium': {
            'n_layer': 24,
            'n_head': 12,
            'n_embd': 1024,
            'block_size': 1024,
        },
        'GPT2-Large': {
            'n_layer': 36,
            'n_head': 20,
            'n_embd': 1280,
            'block_size': 1024,
        },
        'Pythia-410M': {
            'n_layer': 24,
            'n_head': 16,
            'n_embd': 1024,
            'block_size': 2048,
        }
    }
    
    if architecture not in configs:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    config = configs[architecture]
    # Add common parameters
    config.update({
        'dropout': 0.0,
        'bias': False,
        'vocab_size': 50304,  # GPT-2 vocab size rounded up for efficiency
    })
    
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='In-Run Data Shapley score computation.')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='Regular',
                        choices=['Regular', 'GradNorm', 'In-Run Data Shapley', 'GradDotProd'])
    
    # Architecture parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Small', 'GPT2-Medium', 'GPT2-Large', 'Pythia-410M'])
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size in each backward pass')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    
    # Dataset parameters
    parser.add_argument('--train_set', type=str, default='pile')
    parser.add_argument('--val_set', type=str, default='pile')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--eval_iter', type=int, default=20)
    parser.add_argument('--eval_bs', type=int, default=16)

    # In-Run Shapley parameters
    parser.add_argument('--dot_prod_save_interval', type=int, default=None)

    return parser.parse_args()


class TrainingConfig:
    """Training configuration class."""
    
    def __init__(self, args):
        self.args = args
        
        # Model configuration
        model_config = get_model_config(args.architecture)
        for key, value in model_config.items():
            setattr(self, key, value)
        
        # Training hyperparameters
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.learning_rate = args.learning_rate
        self.min_lr = self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed
        
        # Optimizer settings
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        self.lr_decay_iters = 10000
        self.decay_lr = True
        
        # System settings
        self.device = 'cuda'
        self.dtype = 'bfloat16' if self._is_bf16_supported() else 'float16'
        self.compile = False
        self.backend = 'nccl'
        
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
        
        # Result directory setup
        self.result_folder = RESULT_ROOT
        self.setup_result_directories()
    
    def _is_bf16_supported(self):
        """Check if bfloat16 is supported."""
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    def setup_result_directories(self):
        """Setup result directories."""
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
            print(f"Directory '{self.result_folder}' was created.")
        
        self.result_dir = os.path.join(
            self.result_folder, 
            f'{self.args.train_set}_{self.args.val_set}_result'
        )
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print(f"Directory '{self.result_dir}' was created.")
    
    def get_result_file_path(self):
        """Get the result file path."""
        from utils import build_result_dir  # Import here to avoid circular imports
        result_dir = build_result_dir(self.result_dir, self.method, self.args)
        return os.path.join(result_dir + '_results.json')


