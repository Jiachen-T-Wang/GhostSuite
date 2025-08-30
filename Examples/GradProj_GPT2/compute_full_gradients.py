"""
Compute and store exact per-sample full-model gradients (batch_size=1).

This script mirrors data/model setup used in Examples/GradProj_GPT2/main.py,
but saves the full flattened parameter gradient vector per sample.

Notes
- Requires explicit --max_samples and enforces --batch_size 1 (no silent fallback).
- Stores one file per iteration: fullgrad_iter_XXXXXX.pt with {'grad', 'iter', 'batch_idx'}.
- Saves fullgrad_meta.json with parameter slice map and shapes for later masking.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np

# Make project paths available (same strategy as Examples/GradProj_GPT2/main.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataloader import load_all_data, get_batch_from_dataset
from model_setup import create_GPT_model
import importlib.util
_cfg_spec = importlib.util.spec_from_file_location(
    "examples_gradproj_config",
    str(Path(__file__).with_name("config_file.py"))
)
examples_cfg = importlib.util.module_from_spec(_cfg_spec)  # type: ignore
assert _cfg_spec is not None and _cfg_spec.loader is not None
_cfg_spec.loader.exec_module(examples_cfg)  # type: ignore


def build_param_slices(model: torch.nn.Module) -> Tuple[Dict[str, Dict[str, int]], int]:
    """Build a mapping from parameter names to flattened slices in the full vector."""
    slices: Dict[str, Dict[str, int]] = {}
    offset = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        slices[name] = {
            'start': offset,
            'end': offset + numel,
            'numel': numel,
            'shape': list(p.shape),
        }
        offset += numel
    return slices, offset


def flatten_grads(model: torch.nn.Module, total_dim: int, slices: Dict[str, Dict[str, int]]) -> torch.Tensor:
    """Flatten parameter .grad tensors into a single float32 vector of length total_dim."""
    out = torch.zeros(total_dim, dtype=torch.float32, device='cpu')
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            raise RuntimeError(f"Missing gradient for parameter {name}")
        s = slices[name]
        out[s['start']:s['end']] = g.detach().reshape(-1).to(torch.float32).cpu()
    return out


def main():
    print("=" * 80)
    print("Exact Full-Model Gradient Extraction (batch_size=1)")
    print("=" * 80)

    # Parse args via existing config for consistency
    args = examples_cfg.parse_arguments()
    config = examples_cfg.ProjectionConfig(args)

    if config.batch_size != 1:
        raise ValueError("This script requires --batch_size 1 to compute per-sample gradients.")
    if config.max_samples is None or config.max_samples <= 0:
        raise ValueError("Please specify a positive --max_samples to bound gradient storage.")

    # Device and dtype
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; please set --device cpu or enable CUDA.")

    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    model_dtype = dtype_map[config.model_dtype]
    train_dtype = dtype_map[config.train_dtype]

    # Load dataset
    print("Loading Pile dataset ...")
    dataset = load_all_data()
    train_data = dataset['train']
    total_samples = (len(train_data) - config.block_size) // config.block_size
    num_samples = min(config.max_samples, total_samples)
    num_iterations = num_samples  # batch_size=1
    print(f"Will process {num_samples} samples in {num_iterations} iterations")

    # Model
    print(f"Creating {config.architecture} model ...")
    model = create_GPT_model(config)
    model.to(device)
    if model_dtype != torch.float32:
        model = model.to(model_dtype)
    model.eval()
    model.config.use_cache = False
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Autocast context
    ctx = torch.amp.autocast(device_type='cuda', dtype=train_dtype, enabled=(device.type == 'cuda' and train_dtype != torch.float32))

    # Prepare output dir
    base_out = Path(config.proj_dir).parent  # use parent of Results/proj_layers_* if provided
    # Put full grads under Results/fullgrads_* sibling
    full_dir_name = f"fullgrads_seed_{config.seed}_arch_{config.architecture}_dtype_{config.train_dtype}"
    full_dir = base_out / full_dir_name
    full_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_cfg = {
        'architecture': config.architecture,
        'batch_size': config.batch_size,
        'block_size': config.block_size,
        'num_samples': num_samples,
        'seed': config.seed,
        'model_dtype': config.model_dtype,
        'train_dtype': config.train_dtype,
        'device': str(device),
    }
    with open(full_dir / 'run_config_fullgrad.json', 'w') as f:
        json.dump(run_cfg, f, indent=2)

    # Build param slices and save metadata once
    param_slices, total_dim = build_param_slices(model)
    meta = {
        'total_param_dim': total_dim,
        'param_slices': param_slices,
    }
    with open(full_dir / 'fullgrad_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Total parameter dimension: {total_dim}")

    # Generator for reproducible sampling
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    # Main loop
    for it in range(num_iterations):
        # Sample one sequence
        X, Y = get_batch_from_dataset(
            split='train',
            batch_size=1,
            dataset=dataset,
            block_size=config.block_size,
            device=device,
            device_type=device.type,
            generator=generator
        )

        print(X)

        model.zero_grad(set_to_none=True)
        with ctx:
            with torch.enable_grad():
                outputs = model(input_ids=X, labels=Y)
                loss = outputs.loss
        loss.backward()

        gvec = flatten_grads(model, total_dim=total_dim, slices=param_slices)
        save_obj = {
            'iter': it,
            'batch_size': 1,
            'grad': gvec,
        }
        torch.save(save_obj, full_dir / f"fullgrad_iter_{it:06d}.pt")
        if (it + 1) % 10 == 0:
            print(f"Saved {it+1}/{num_iterations} full gradients ...")

    print(f"\nSaved full gradients to: {full_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
