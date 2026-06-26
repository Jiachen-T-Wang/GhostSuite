"""Model + LoRA + optimizer setup for the GREATS SFT example.

Loads a HF causal LM (Llama-2-7b by default, from a local cached path), applies a
peft LoRA adapter mirroring the upstream `base_training_args.sh`, and builds an
AdamW optimizer + linear schedule over the LoRA parameters.
"""

from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_tokenizer(config):
    tok = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
    if tok.pad_token is None:
        # Llama-2 has no pad token; right-pad with EOS (masked out via attention_mask/labels).
        tok.pad_token = tok.eos_token
    return tok


def load_model(config, device) -> torch.nn.Module:
    dtype = _DTYPES[config.model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    return model


def build_optimizer_and_scheduler(model, config, num_training_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    warmup_steps = int(config.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def setup_model_optimizer_tokenizer(config, device, num_training_steps: int) -> Tuple:
    tokenizer = load_tokenizer(config)
    model = load_model(config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, num_training_steps)
    return model, optimizer, scheduler, tokenizer
