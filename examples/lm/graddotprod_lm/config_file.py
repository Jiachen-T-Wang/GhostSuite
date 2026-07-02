"""Configuration management for the training script."""

import argparse
import os
import sys
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import build_result_dir


# Directory configurations
# Anchored to this file's directory so each git worktree writes to its own
# results/ tree instead of a shared absolute path.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='In-Run Data Shapley score computation.')
    
    # Method parameters
    parser.add_argument('--method', type=str, default='Regular', choices=['Regular', 'GradDotProd'])
    
    # Architecture parameters
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                       choices=['GPT2-Tiny', 'GPT2-Small', 'GPT2-Medium', 'GPT2-Large'])
    parser.add_argument('--no_tie_weights', dest='tie_weights', action='store_false',
                        help='Untie the token-embedding and LM-head weight. Default: tied (standard '
                             'GPT-2). Tied weights are handled by all ghost paths (eager, '
                             'decoupled) via the cross-term tied-weight finalizer.')
    parser.set_defaults(tie_weights=True)
    # The decoupled in-graph + torch.compile fast path is the DEFAULT for GradDotProd (compile-clean:
    # native layer backward preserved; ~+9% step time). It applies to GPT-2 token models on a single
    # GPU with grad-accum 1 and bf16/fp32; main.py auto-falls back to the eager engine (with a notice)
    # for runs that can't use it. Use --eager to force the eager per-layer-hook engine.
    parser.add_argument('--decoupled_fn', dest='decoupled_fn', action='store_true',
                        help='Use the decoupled in-graph fast path (this is the default).')
    parser.add_argument('--eager', dest='decoupled_fn', action='store_false',
                        help='Use the eager per-layer-hook engine instead of the default decoupled '
                             'in-graph + torch.compile fast path.')
    parser.add_argument('--decoupled_compile', dest='decoupled_compile', action='store_true',
                        help='Regional-compile the transformer blocks via torch.compile (default on '
                             'with the decoupled path; the speedup lever).')
    parser.add_argument('--no_decoupled_compile', dest='decoupled_compile', action='store_false',
                        help='Keep the decoupled path but skip torch.compile (decoupled-eager; '
                             'usually slower than --eager — for debugging).')
    parser.set_defaults(decoupled_fn=True, decoupled_compile=True)
    parser.add_argument('--decoupled_compile_toplevel', action='store_true',
                        help='With --decoupled_compile, also compile the top-level in-graph layers '
                             '(output lm_head + final norm). Only the non-tied lm_head is compiled, '
                             'so this mainly helps the --no_tie_weights config (TorchTitan found the '
                             'output Linear is the only top-level layer worth compiling).')
    parser.add_argument('--decoupled_mem_budget', type=float, default=None,
                        help='With --decoupled_compile, set the Inductor activation-memory budget in '
                             '(0,1] (compile-native activation checkpointing): 1.0 saves everything '
                             '(default), lower recomputes more in backward to cut peak memory. '
                             'Matters at GPT-2-Medium/Large scale.')
    parser.add_argument('--separate_val', dest='separate_val', action='store_true',
                        help='Two-pass separate-val engine (decoupled path only; this is the '
                             'default): each step runs ONE plain backward on the val batch and '
                             'harvests autograd .grad as the cached per-param val gradient; the '
                             'train microbatches then run WITHOUT appended val rows, projecting '
                             'their in-graph dots against the cache. Training loss/grads become '
                             'bit-consistent with regular training, peak memory drops (~half), the '
                             'tied lm_head moves from the eager capture path to the compiled '
                             'in-graph dot, and the val cost is paid once per step instead of once '
                             'per microbatch. Logged dots equal the combined-batch dots up to a '
                             'constant rescale: N*(T_tr+T_v)^2/(T_tr*T_v) with per-microbatch train '
                             'tokens T_tr, val tokens T_v, grad-accum N (rankings unchanged).')
    parser.add_argument('--no_separate_val', dest='separate_val', action='store_false',
                        help='Restore the combined train+val batch engine (pre-v0.6 behavior and '
                             'dot scale).')
    parser.set_defaults(separate_val=True)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of microsteps to accumulate before the optimizer step. Each '
                             'microstep draws a distinct train sub-batch of --batch_size; the ghost '
                             'dot-products are collected per microstep and the per-microstep val '
                             'gradients are summed for a single subtract-val recovery.')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--warmup_step', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=None,
                        help='Floor of the cosine LR decay (default: 0.1 * learning_rate)')
    parser.add_argument('--lr_decay_iters', type=int, default=None,
                        help='Cosine LR decay horizon in steps (default: --max_steps)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw'],
                        help='Optimizer (only AdamW is implemented)')
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
    parser.add_argument('--eval_seed', type=int, default=1234,
                        help='Fixed RNG seed for the evaluation window set; identical across '
                             'training seeds so val/test loss noise is seed-independent')

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
    parser.add_argument('--score_val_from_eval_pool', action='store_true',
                        help='Draw the (dynamic) scoring validation batch from the SAME fixed '
                             'window pool used to compute eval loss (eval_seed/eval_iter/eval_bs), '
                             'so dot-product scores target exactly the eval population')
    parser.add_argument('--score_exclude_params', type=str, default=None,
                        help='Comma-separated parameter-name substrings to EXCLUDE from the logged '
                             'dot-product/cosine score (e.g. "wte,lm_head"); training is unaffected')
    parser.add_argument('--replay_run_dir', type=str, default=None,
                        help='Path to a previous GradDotProd run directory for filtered replay training')
    parser.add_argument('--replay_filter_metric', type=str, default='dot_product',
                        choices=['dot_product', 'cosine'],
                        help='Metric used to filter replay samples')
    parser.add_argument('--replay_filter_threshold', type=float, default=0.0,
                        help='Drop samples with metric below this threshold (default drops negatives)')
    parser.add_argument('--replay_filter_invert', action='store_true',
                        help='Keep samples with metric BELOW the threshold instead of at/above it '
                             '(the bottom-fraction / rejected arm of a selection experiment)')
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
        # Weight-tying of the token embedding / LM head (see --no_tie_weights).
        self.tie_weights = getattr(args, 'tie_weights', True)

        # Decoupled in-graph + regional-compile ghost path (fn-path) — the DEFAULT for GradDotProd
        # (see --eager to opt out). Tied weights (wte/lm_head) are supported on both the batched and
        # decoupled paths (cross-terms via finalize_tied_param); --no_tie_weights remains available.
        # main.py gates this down to the eager engine for runs that can't use the fast path.
        self.decoupled_fn = getattr(args, 'decoupled_fn', True)
        self.decoupled_compile = getattr(args, 'decoupled_compile', True)
        self.decoupled_compile_toplevel = getattr(args, 'decoupled_compile_toplevel', False)
        self.decoupled_mem_budget = getattr(args, 'decoupled_mem_budget', None)
        # Two-pass separate-val engine (see --separate_val): a mode of the decoupled path, so it
        # follows decoupled_fn — --eager (and the main.py capability gate, which clears
        # decoupled_fn) turns it off rather than erroring.
        self.separate_val = bool(getattr(args, 'separate_val', True)) and self.decoupled_fn

        # Sequence length (block size). For GPT architectures it is fixed by the
        # model config table; sampled windows must match it (e.g. GPT2-Tiny=64).
        try:
            from shared.GPT2_configs import get_model_config
            self.block_size = get_model_config(self.architecture)['block_size']
        except (ImportError, ValueError, KeyError):
            self.block_size = 1024
        
        # Training hyperparameters
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.learning_rate = args.learning_rate
        self.min_lr = args.min_lr if args.min_lr is not None else self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed

        # Optimizer settings (currently just assume using AdamW)
        self.optimizer = args.optimizer
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        # Cosine decay horizon; defaults to the full run (nanoGPT convention).
        self.lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.max_steps
        self.decay_lr = True
        
        # System settings
        self.device = 'cuda'
        self.compile = False
        self.backend = 'nccl'

        # Precision settings
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype

        # Gradient accumulation
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.full_batch_size = args.batch_size * args.gradient_accumulation_steps
        
        # Evaluation settings
        self.eval_iters = args.eval_iter
        self.eval_interval = args.eval_interval
        self.eval_bs = args.eval_bs
        self.eval_seed = args.eval_seed
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
        self.score_val_from_eval_pool = args.score_val_from_eval_pool
        self.score_exclude_params = (
            [s.strip() for s in args.score_exclude_params.split(',') if s.strip()]
            if args.score_exclude_params else []
        )
        self.replay_run_dir = args.replay_run_dir
        self.replay_filter_metric = args.replay_filter_metric
        self.replay_filter_threshold = args.replay_filter_threshold
        self.replay_filter_invert = args.replay_filter_invert
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

        # Create result folder if it doesn't exist (exist_ok: DDP ranks race here)
        if not os.path.exists(self.result_folder):
            print(f"Results folder '{self.result_folder}' was created.")
        os.makedirs(self.result_folder, exist_ok=True)

        # Create specific result directory for this run
        self.result_dir = build_result_dir(self.result_folder, self.method, self.args)

        if not os.path.exists(self.result_dir):
            print(f"Results directory for this specific run '{self.result_dir}' was created.")
        os.makedirs(self.result_dir, exist_ok=True)
    
    def get_result_file_path(self):
        """Get the result file path for storing training statistics."""
        result_dir = self.result_dir
        return os.path.join(result_dir + '_results.json')
