import os
import torch
import random
import numpy as np


# Helper function to build a descriptive result directory and file name
def build_result_dir(base_dir, method, args):
    parts = [
        method,
        f"{args.train_set}",
        f"{args.val_set}",
        f"BS{args.batch_size}",
        f"ValBS{args.val_batch_size}",
        f"LR{args.learning_rate}",
        f"Warm{args.warmup_step}",
        args.optimizer,
        f"Seed{args.seed}"
    ]
    dir_name = "_".join(parts)
    return os.path.join(base_dir, dir_name)


def set_seed(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch's random number generator
    torch.manual_seed(seed)
    
    # If you are using GPUs
    if torch.cuda.is_available():
        # Set seed for all CUDA devices
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False