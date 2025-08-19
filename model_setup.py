"""Model setup and initialization utilities."""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel as GPT 
from transformers import LlavaForConditionalGeneration


def setup_model_and_optimizer(config, device, ddp_info):
    """Create model, optimizer, and scaler based on the configuration."""
    """Interface to main.py"""

    # Create model
    print(f"[INFO] Creating {config.architecture} model...")
    model = create_model(config)
    model.to(device)
    print(f"[INFO] Model created and moved to {device}.")
    
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

    if config.args.architecture.startswith("LLaVA"):
        model = create_llava_model(config)
    elif config.args.architecture.startswith("GPT"):
        model = create_GPT_model(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    
    return model


def create_GPT_model(config):
    """Create and initialize the GPT model."""

    config_GPT = {
        'GPT2-Small': {
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'block_size': 1024,
            'vocab_size': 50304,
        },
        'GPT2-Medium': {
            'n_layer': 24,
            'n_head': 12,
            'n_embd': 1024,
            'block_size': 1024,
            'vocab_size': 50304,
        },
        'GPT2-Large': {
            'n_layer': 36,
            'n_head': 20,
            'n_embd': 1280,
            'block_size': 1024,
            'vocab_size': 50304,
        }
    }

    if config.architecture not in config_GPT:
        raise ValueError(f"Unknown GPT architecture: {config.architecture}")

    model_config = config_GPT[config.architecture]

    model_args = dict(
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        n_positions=model_config['block_size'],
        bos_token_id=model_config['vocab_size'],
        eos_token_id=model_config['vocab_size'],
        vocab_size=model_config['vocab_size'],
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        summary_first_dropout=0,
        use_cache=False,
    )
    
    gptconf = GPT2Config(**model_args)
    model = GPT(gptconf)
    
    return model


def create_llava_model(config):
    """Create and initialize the LLaVA model."""

    model_id_map = {
        "LLaVA-7B": "llava-hf/llava-1.5-7b-hf",
        "LLaVA-13B": "llava-hf/llava-1.5-13b-hf",
    }

    if config.args.architecture in model_id_map.keys():
        repo_id = model_id_map[config.args.architecture]
    else:
        raise ValueError(f"Unknown LLaVA architecture: {config.args.architecture}")

    base_model = LlavaForConditionalGeneration.from_pretrained(
        repo_id,
        torch_dtype=config.model_dtype,
    )

    model = LLaVAModelWrapper(base_model)

    # Freeze the vision tower parameters
    for p in model.base_model.model.vision_tower.parameters():
        p.requires_grad = False        # CLIP ViT is frozen

    print(f"[INFO] LLaVA model {config.args.architecture} created with vision tower frozen.")

    return model


from torch import nn
class LLaVAModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, input_ids, labels=None, **kwargs):
        # If input_ids is a dict, unpack it
        if isinstance(input_ids, dict):
            return self.base_model(**input_ids, labels=labels)
        else:
            # Regular model call for non-LLaVA models
            return self.base_model(input_ids=input_ids, labels=labels, **kwargs)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the base model
        if name == 'base_model':
            return super().__getattr__(name)
        return getattr(self.base_model, name)


