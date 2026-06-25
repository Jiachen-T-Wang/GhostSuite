#!/usr/bin/env python3
"""Entry point for the GREATS SFT (LoRA instruction tuning) example.

Self-contained online batch selection on a HF causal LM + peft LoRA, scored with
the GradDotProd ghost engine. See docs/plans/greats_sft_phaseB_2026-06-25.md.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# sft -> greats -> examples -> repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from config_file import parse_arguments, TrainingConfig
from data_utils import load_instruction_samples, build_mmlu_val
from model_setup import load_tokenizer, load_model, build_optimizer_and_scheduler
from training_loop import GreatsSFTTrainer


def main():
    args = parse_arguments()
    config = TrainingConfig(args)
    torch.manual_seed(config.seed)

    device = config.device

    # Tokenizer first (needed to encode data), then data, then model/optimizer.
    tokenizer = load_tokenizer(config)

    train_samples = load_instruction_samples(
        config.train_files, tokenizer, config.max_seq_length,
        percentage=config.percentage, seed=config.data_seed,
        max_train_samples=config.max_train_samples,
        read_limit_per_file=config.read_limit_per_file,
    )
    val_samples = build_mmlu_val(
        config.data_dir, tokenizer, config.subject, config.n_val, config.max_seq_length
    )

    consume = config.candidate_batch_size if config.method == "GREATS" else config.batch_size
    steps_per_epoch = max(1, len(train_samples) // consume)
    total_steps = steps_per_epoch * config.num_train_epochs
    if config.max_steps and config.max_steps > 0:
        total_steps = min(total_steps, config.max_steps)
    print(f"[INFO] total optimizer steps: {total_steps} "
          f"({steps_per_epoch}/epoch x {config.num_train_epochs} epochs, consume={consume})")

    model = load_model(config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, total_steps)

    trainer = GreatsSFTTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, tokenizer=tokenizer,
        config=config, device=device, train_samples=train_samples,
        val_samples=val_samples, total_steps=total_steps,
    )
    trainer.train()


if __name__ == "__main__":
    main()
