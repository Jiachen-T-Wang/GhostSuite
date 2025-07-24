"""Model setup and initialization utilities."""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel as GPT # Import from transformers


def create_model(config):
    """Create and initialize the GPT model."""

    # setting non-zero values for dropout parameters may cause deviations in training results between 'Regular' and 'Ghost' modes.
    # this is okey for practical purposes
    # but if you want to have exact the same results (e.g., for debugging), set all dropout parameters to 0.
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_positions=config.block_size,
        bos_token_id=config.vocab_size,
        eos_token_id=config.vocab_size,
        vocab_size=config.vocab_size,
        # resid_pdrop=0,
        # embd_pdrop=0,
        # attn_pdrop=0,
        # summary_first_dropout=0,
    )
    
    gptconf = GPT2Config(**model_args)
    model = GPT(gptconf)
    
    return model


def setup_model_and_optimizer(config, device, ddp_info):
    """Setup model, optimizer, and related components."""
    # Create model
    model = create_model(config)
    model.to(device)
    
    trainable_layers = None # This remains the same
    
    # Setup optimizer directly using torch.optim
    # The original model.configure_optimizers was likely a wrapper around this.
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(config.beta1, config.beta2), 
        weight_decay=config.weight_decay
    )
    
    # Setup gradient scaler
    # scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    scaler = torch.cuda.amp.GradScaler(enabled=False) # For testing purposes, set to False
    
    # Compile model if requested
    if config.compile:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Wrap in DDP if distributed
    if ddp_info['ddp']:
        model = DDP(model, device_ids=[ddp_info['ddp_local_rank']])
    
    return model, optimizer, scaler, trainable_layers


def get_raw_model(model, ddp):
    """Get the raw model (unwrap DDP if needed)."""
    return model.module if ddp else model
