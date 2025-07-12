#!/usr/bin/env python3
"""
Main training script for GPT models with In-Run Data Shapley support.

This script provides a clean interface for training GPT models with optional
In-Run Data Shapley value computation.
"""

import torch

from config_file import parse_arguments, TrainingConfig
from training_utils import (
    setup_distributed,
    setup_torch_backend,
    cleanup_distributed,
    print_training_info
)
from model_setup import setup_model_and_optimizer
from training_loop import Trainer
from dataloader import load_all_data, get_batch_from_dataset, get_batch_subdomain
from utils import set_seed


# def setup_data_functions(dataset, config):
#     """Setup data loading functions."""
#     def get_batch(split, batch_size, return_idx=False):
#         return get_batch_from_dataset(split, batch_size, dataset, return_idx=return_idx)
    
#     def get_val_batch(batch_size, domain_name, return_idx=False, return_first=False):
#         return get_batch('val', batch_size, return_idx=False)
    
#     return get_batch, get_val_batch


def setup_data_functions(dataset, config):
    """Setup data loading functions with split-specific RNGs."""

    train_gen = torch.Generator()
    train_gen.manual_seed(config.seed)

    val_gen = torch.Generator()
    val_gen.manual_seed(config.seed + 1)

    test_gen = torch.Generator()
    test_gen.manual_seed(config.seed + 2)

    generators = {'train': train_gen, 'val': val_gen, 'test': test_gen}

    def get_batch(split, batch_size, return_idx=False):
        gen = generators.get(split, train_gen)
        return get_batch_from_dataset(
            split, batch_size, dataset, return_idx=return_idx, generator=gen
        )

    def get_val_batch(batch_size, domain_name, return_idx=False, return_first=False):
        return get_batch('val', batch_size, return_idx=return_idx)
    
    return get_batch, get_val_batch


def main():
    """Main training function."""
    # Parse arguments and setup configuration
    args = parse_arguments()
    config = TrainingConfig(args)
    
    # Setup distributed training
    ddp_info = setup_distributed()
    
    # Set random seed
    set_seed(config.seed + ddp_info['seed_offset'])
    
    # Setup PyTorch backend
    device_type, ptdtype, ctx = setup_torch_backend(config)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_all_data()
    print('Pile dataset loaded')
    
    # Setup data functions
    get_batch_fn, get_val_batch_fn = setup_data_functions(dataset, config)
    
    # Calculate tokens per iteration
    tokens_per_iter = (config.gradient_accumulation_steps * 
                      ddp_info['ddp_world_size'] * 
                      config.batch_size * 
                      config.block_size)
    
    # Print training information
    print_training_info(config, tokens_per_iter)
    
    # Setup model and optimizer
    print("Setting up model and optimizer...")
    model, optimizer, scaler, trainable_layers = setup_model_and_optimizer(
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
        ctx=ctx,
        trainable_layers=trainable_layers
    )
    
    # Run training
    try:
        trainer.run_training()
    finally:
        # Cleanup
        if ddp_info['ddp']:
            cleanup_distributed()


if __name__ == "__main__":
    main()
