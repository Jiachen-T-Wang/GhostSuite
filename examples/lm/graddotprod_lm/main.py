#!/usr/bin/env python3
"""
Main training script for GPT models with In-Run Data Shapley support.

This script provides a clean interface for training GPT models with optional
In-Run Data Shapley value computation.
"""

import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config_file import parse_arguments, TrainingConfig
from shared.training_utils import (
    setup_distributed,
    setup_torch_backend,
    cleanup_distributed,
    print_training_info,
    setup_data_functions,
    load_dataset_main
)
from shared.model_setup import setup_model_and_optimizer
from training_loop import Trainer
from shared.utils import set_seed


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Create training config object from parsed arguments
    config = TrainingConfig(args)
    
    # Setup distributed training
    ddp_info = setup_distributed()
    
    # Set random seed
    set_seed(config.seed + ddp_info['seed_offset'])

    # Setup model and optimizer
    model, optimizer, scaler = setup_model_and_optimizer(
        config, ddp_info['device'], ddp_info
    )

    # Setup PyTorch backend. Passing the model keys the autocast decision on the
    # ACTUAL parameter dtype (the model may have fallen back from bf16), not the
    # claimed --model_dtype/--train_dtype pair.
    ctx = setup_torch_backend(config, model=model)

    # Print training information
    print_training_info(config)

    # The decoupled in-graph + torch.compile fast path is the default for GradDotProd, but it only
    # supports GPT-2 token models on a single GPU with bf16/fp32 (gradient accumulation IS supported:
    # the per-microstep val grads are summed for one subtract-val recovery). Fall back to the eager
    # engine (with a clear notice) for runs that can't use it, rather than erroring. Pass --eager to
    # select the eager engine explicitly.
    if config.method == 'GradDotProd' and config.decoupled_fn:
        reason = None
        if ddp_info.get('ddp', False):
            reason = "multi-GPU / DDP"
        elif scaler.is_enabled():
            reason = "float16 GradScaler (use --train_dtype bfloat16 or float32)"
        elif not str(config.architecture).startswith('GPT2'):
            reason = f"architecture {config.architecture} (the fast path supports GPT-2 token models)"
        elif config.log_grad_norms:
            reason = "--log_grad_norms (per-sample gradient norms are computed by the eager engine only)"
        if reason is not None:
            print(f"[INFO] Optimized decoupled+compile path unavailable ({reason}); using the eager "
                  f"engine. Pass --eager to select it explicitly.")
            config.decoupled_fn = False
            config.decoupled_compile = False

    # Load dataset
    dataset = load_dataset_main(args.train_set, args.val_set)
    
    # Setup data functions
    get_batch_fn, get_val_batch_fn = setup_data_functions(
        dataset, config, ddp_info['device'], ddp_info=ddp_info
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        ddp_info=ddp_info,
        get_batch_fn=get_batch_fn,
        get_val_batch_fn=get_val_batch_fn,
        ctx=ctx
    )

    trainer.run_training()
    
if __name__ == "__main__":
    main()
