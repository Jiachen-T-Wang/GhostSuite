"""Model setup and initialization utilities."""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .gpt2 import GPTConfig
from .gpt2 import GPT


DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def setup_model_and_optimizer(config, device, ddp_info):
    """Create model, optimizer, and scaler based on the configuration."""
    """Interface to main.py"""

    # Create model
    print(f"[INFO] Creating {config.architecture} model...")
    model = create_model(config)

    # Cast parameters to the requested --model_dtype (mirrors examples/lm/gradproj_lm).
    # bfloat16 falls back to float32 on CUDA devices without bf16 support;
    # config.model_dtype is updated so downstream consumers (autocast selection,
    # logging) see the actual parameter dtype.
    model_dtype = DTYPE_MAP[config.model_dtype]
    if (model_dtype == torch.bfloat16 and 'cuda' in str(device)
            and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())):
        print("[WARN] bfloat16 not supported on this CUDA device; using float32 model weights.")
        model_dtype = torch.float32
        config.model_dtype = 'float32'
    model.to(device=device, dtype=model_dtype)
    print(f"[INFO] Model created and moved to {device} ({model_dtype}).")

    # Setup optimizer and scaler
    print("[INFO] Setting up optimizer and scaler...")
    optimizer, scaler = setup_adamw_optimizer_and_scaler(model, config)
    print("[INFO] Optimizer and scaler set up.")
    
    # Compile model if requested
    if config.compile:
        print("Compiling the model ...")
        model = torch.compile(model)
        print("Model compiled successfully.")
    
    # Wrap in DDP if distributed
    if ddp_info['ddp']:
        model = DDP(model, device_ids=[ddp_info['ddp_local_rank']])
    
    return model, optimizer, scaler


def get_raw_model(model, ddp):
    """Get the raw model (unwrap DDP if needed)."""
    return model.module if ddp else model



def setup_adamw_optimizer_and_scaler(model, config):
    """Setup AdamW optimizer and gradient scaler"""

    weight_decay = config.weight_decay
    learning_rate = config.learning_rate
    beta1 = config.beta1
    beta2 = config.beta2
    
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to bias terms and layer norm parameters
            if any(nd in name for nd in ['bias', 'norm', 'ln']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8
    )
    
    # Create gradient scaler for mixed precision training with fp16
    enable_grad_scaler = (config.train_dtype == 'float16' and next(model.parameters()).dtype == torch.float32)
    scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)

    print(f"[INFO] Model parameters will be in {next(model.parameters()).dtype} precision.")
    
    return optimizer, scaler







def create_model(config):
    """Create and initialize the model based on the architecture."""

    if config.args.architecture.startswith("GPT"):
        model = setup_model_GPT(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    return model



def setup_model_GPT(config):
    """Initialize and setup the model"""

    from .GPT2_configs import get_model_config
    
    # Get model configuration
    model_config = get_model_config(config.architecture)
    n_layer = model_config['n_layer']
    n_head = model_config['n_head']
    n_embd = model_config['n_embd']
    block_size = model_config['block_size']
    dropout = 0.0
    bias = False
    vocab_size = 50304
    
    # Create model configuration
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
        # Embedding/LM-head weight tying (default standard GPT-2 tying; --no_tie_weights to untie).
        tie_weights=getattr(config, "tie_weights", True),
    )
    
    # Initialize model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    return model




def create_GPT_model(config):
    """Create and initialize the GPT model (shared custom nanoGPT-style GPT).

    Used by the gradproj_lm example. Builds the same `GPT` used elsewhere via
    the shared `GPT2_configs` table so the ghost layers (nn.Linear / nn.Embedding)
    are recognized by the projection engine.
    """
    from .GPT2_configs import get_model_config

    model_config = get_model_config(config.architecture)

    gptconf = GPTConfig(
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        block_size=model_config['block_size'],
        bias=False,
        vocab_size=50304,
        dropout=0.0,
    )
    model = GPT(gptconf)

    return model

