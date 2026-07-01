"""Stage-1 training loop: train the model and capture per-step projected gradients.

Unlike gradproj_lm's loop (which runs at a fixed checkpoint with no optimizer), this
loop takes real optimizer steps so the captured per-sample projected gradients lie on
an actual training trajectory — exactly what Data Value Embedding needs. Each step:

    zero_grad -> forward+backward (triggers ghost hooks) -> collect_batch (save proj +
    per-step lr/order) -> optimizer.step() -> clear_gradients
"""

import math
import os
import sys
import time

import torch
from tqdm import tqdm

# examples/lm/ provides `shared` (this file lives at examples/dve_lm/).
_LM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lm")
if _LM_DIR not in sys.path:
    sys.path.insert(0, _LM_DIR)
from shared.dataloader import get_batch_from_dataset


def compute_lr(step, config):
    """Per-step learning rate: constant, linear, or cosine -- each with linear warmup."""
    # Linear warmup from 0 to learning_rate over the first warmup_steps.
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, progress))

    if config.lr_schedule == 'constant':
        return config.learning_rate

    if config.lr_schedule == 'linear':
        # Linear decay to 0 at max_steps (matches HF get_linear_schedule_with_warmup,
        # the reference DVE pretraining schedule).
        return config.learning_rate * (1.0 - progress)

    # cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def build_optimizer(model, config):
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay, eps=1e-8)
    elif config.optimizer == 'sgd':
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate,
            momentum=config.momentum, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_and_capture(model, engine, optimizer, dataset, config, device, ctx):
    """Run the training trajectory, capturing projected gradients every step."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    model.train()
    total_loss = 0.0
    step_times = []

    pbar = tqdm(range(config.max_steps), desc="Train+capture")
    for step in pbar:
        t0 = time.time()
        lr = compute_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr

        X, Y, sample_ix = get_batch_from_dataset(
            split='train', batch_size=config.batch_size, dataset=dataset,
            block_size=config.block_size, device=device, device_type=device.type,
            generator=generator, return_idx=True)

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            with torch.enable_grad():
                outputs = model(X, Y)
                loss = outputs.loss
        loss.backward()

        # Capture per-sample projected gradients at the CURRENT parameters, tagged with
        # window identity, learning rate, and training-step order for the recursion.
        batch_indices = [int(i) for i in sample_ix]
        engine.collect_batch(batch_indices=batch_indices,
                             extra={'lr': float(lr), 'order': int(step)})

        optimizer.step()
        engine.clear_gradients()

        batch_loss = loss.item()
        total_loss += batch_loss
        step_times.append(time.time() - t0)
        pbar.set_postfix({'loss': f'{batch_loss:.4f}',
                          'avg': f'{total_loss / (step + 1):.4f}',
                          'lr': f'{lr:.2e}'})

    avg_loss = total_loss / max(1, config.max_steps)
    print(f"\n[train] {config.max_steps} steps, avg loss {avg_loss:.4f}, "
          f"avg step {sum(step_times) / len(step_times):.3f}s")
    return {'avg_loss': avg_loss, 'steps': config.max_steps}
