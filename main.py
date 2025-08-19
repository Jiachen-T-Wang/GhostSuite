#!/usr/bin/env python3
"""
Main training script for GPT models with In-Run Data Shapley support.

This script provides a clean interface for training GPT models with optional
In-Run Data Shapley value computation.
"""

from config_file import parse_arguments, TrainingConfig
from training_utils import (
    setup_distributed,
    setup_torch_backend,
    cleanup_distributed,
    print_training_info
)
from model_setup import setup_model_and_optimizer
from training_loop import Trainer
from training_utils import setup_data_functions, load_dataset_main
from utils import set_seed


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Create training config object from parsed arguments
    config = TrainingConfig(args)
    
    # Setup distributed training
    ddp_info = setup_distributed()
    
    # Set random seed
    set_seed(config.seed + ddp_info['seed_offset'])
    
    # Setup PyTorch backend
    ctx = setup_torch_backend(config)
    
    # Load dataset
    dataset = load_dataset_main(args.train_set, args.val_set)
    
    # Setup data functions
    get_batch_fn, get_val_batch_fn = setup_data_functions(dataset, config, ddp_info['device'])
    
    # Print training information
    print_training_info(config)
    
    # Setup model and optimizer
    model, optimizer, scaler = setup_model_and_optimizer(
        config, ddp_info['device'], ddp_info
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
