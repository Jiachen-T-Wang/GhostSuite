"""Data Value Embedding (DVE) — 4-stage pipeline entry point.

Stages (selected by flags; can be run together or separately, all sharing config-derived
paths so a later invocation finds the earlier outputs):

  1. --train_and_store_grad : train the model, capture per-step projected gradients
  2. --compute_embedding    : reverse-recursion -> data value embeddings
  3. --compute_value        : project test gradients, dot vs embeddings -> value matrix
  4. --attribute            : rank training points by value for selected test points

Example (synthetic smoke, all stages):
  python main.py --data_source synthetic --architecture GPT2-Tiny --device cuda \
      --optimizer sgd --max_steps 20 --batch_size 8 --n_test 16 \
      --train_and_store_grad --compute_embedding --compute_value --attribute
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config_file import parse_arguments, DVEConfig
from shared.dataloader import load_all_data, make_synthetic_dataset, get_batch_from_dataset
from shared.model_setup import create_GPT_model
from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine
from ghostEngines.gradProjection.dve_embedding import compute_embeddings_reverse
from ghostEngines.gradProjection.dve_value import compute_values, attribute

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


def _resolve_device(config):
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    return device


def _make_ctx(config, device):
    train_dtype = DTYPE_MAP[config.train_dtype]
    # Keep autocast on only for CUDA half-precision; device_type tracks the real
    # device so the context is valid (a no-op) on CPU runs.
    enabled = device.type == 'cuda' and train_dtype != torch.float32
    return torch.amp.autocast(device_type=device.type, dtype=train_dtype, enabled=enabled)


def _build_model(config, device):
    model = create_GPT_model(config)
    model = model.to(device)
    model_dtype = DTYPE_MAP[config.model_dtype]
    if model_dtype != torch.float32:
        model = model.to(model_dtype)
    model.config.use_cache = False
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
    except ImportError:
        pass
    return model


def _engine_config(config, proj_dir):
    return {
        'proj_layers': config.proj_layers,
        'proj_rank_total': config.proj_rank_total,
        'proj_rank_min': config.proj_rank_min,
        'proj_seed': config.proj_seed,
        'proj_dtype': config.proj_dtype,
        'proj_dir': proj_dir,
        'proj_row_orthonormal': config.proj_row_orthonormal,
        'proj_save_interval': 1,  # DVE needs every step
        'include_embeddings': config.include_embeddings,
    }


def _load_dataset(config):
    if config.data_source == 'synthetic':
        print("Building synthetic dataset (random tokens)...")
        return make_synthetic_dataset(seed=config.seed)
    print("Loading Pile dataset...")
    return load_all_data()


# --------------------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------------------
def stage_train(config, device, ctx, dataset):
    from dve_train_loop import train_and_capture, build_optimizer

    print("\n" + "=" * 60 + "\n[Stage 1] Train and capture projected gradients\n" + "=" * 60)
    os.makedirs(config.capture_dir, exist_ok=True)

    model = _build_model(config, device)
    print(f"Model dtype: {next(model.parameters()).dtype}")

    engine = GradProjLoraEngine(model, **_engine_config(config, config.capture_dir))
    engine.attach()

    optimizer = build_optimizer(model, config)
    stats = train_and_capture(model, engine, optimizer, dataset, config, device, ctx)
    engine.detach()

    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"[train] saved final checkpoint to {config.checkpoint_path}")

    with open(os.path.join(config.capture_dir, 'dve_run_config.json'), 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(),
                   'config': repr(config), 'stats': stats,
                   'optimizer': config.optimizer, 'lr_mode': config.lr_mode,
                   'max_steps': config.max_steps, 'batch_size': config.batch_size}, f, indent=2)


# --------------------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------------------
def stage_embedding(config, device):
    print("\n" + "=" * 60 + "\n[Stage 2] Compute data value embeddings (reverse recursion)\n" + "=" * 60)
    # Recursion runs in float32; on GPU if available for speed.
    rec_device = 'cuda' if (device.type == 'cuda') else 'cpu'
    compute_embeddings_reverse(config.capture_dir, config.embed_dir,
                               lr_mode=config.lr_mode, device=rec_device)


# --------------------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------------------
def _compute_test_projections(config, device, ctx, dataset):
    """Project per-sample TEST gradients using the same P (final checkpoint)."""
    model = _build_model(config, device)
    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {config.checkpoint_path}; run --train_and_store_grad first.")
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()

    test_capture = os.path.join(config.value_dir, '_test_capture')
    engine = GradProjLoraEngine(model, **_engine_config(config, test_capture))
    engine.attach()

    generator = torch.Generator()
    generator.manual_seed(config.seed + 777)  # distinct stream from training

    n_test = config.n_test
    bs = config.test_batch_size
    blocks, ids = [], []
    n_done = 0
    while n_done < n_test:
        cur = min(bs, n_test - n_done)
        X, Y, ix = get_batch_from_dataset(
            split='test', batch_size=cur, dataset=dataset, block_size=config.block_size,
            device=device, device_type=device.type, generator=generator, return_idx=True)
        model.zero_grad(set_to_none=True)
        with ctx:
            with torch.enable_grad():
                loss = model(X, Y).loss
        loss.backward()
        # Transient pass: we use the returned tensor directly, never the disk file.
        proj = engine.collect_batch(batch_indices=[int(i) for i in ix], save=False)  # [cur, total]
        blocks.append(proj.detach().float().cpu())
        ids.extend(int(i) for i in ix)
        engine.clear_gradients()
        n_done += cur

    engine.detach()
    test_proj = torch.cat(blocks, dim=0)  # [n_test, total]
    print(f"[value] computed test projections {tuple(test_proj.shape)}")
    return test_proj, ids


def stage_value(config, device, ctx, dataset):
    print("\n" + "=" * 60 + "\n[Stage 3] Compute value matrix (test grads . embeddings)\n" + "=" * 60)
    os.makedirs(config.value_dir, exist_ok=True)
    test_proj, test_ids = _compute_test_projections(config, device, ctx, dataset)
    rec_device = 'cuda' if (device.type == 'cuda') else 'cpu'
    compute_values(config.embed_dir, test_proj, test_ids=test_ids,
                   save_path=os.path.join(config.value_dir, 'values.pt'), device=rec_device)


# --------------------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------------------
def stage_attribute(config):
    print("\n" + "=" * 60 + "\n[Stage 4] Corpus attribution\n" + "=" * 60)
    values_path = os.path.join(config.value_dir, 'values.pt')
    if not os.path.exists(values_path):
        raise FileNotFoundError(f"No {values_path}; run --compute_value first.")
    result = torch.load(values_path, map_location='cpu')
    attribute(result, top_k=config.top_k)


def main():
    args = parse_arguments()
    config = DVEConfig(args)
    print("=" * 80 + f"\nData Value Embedding pipeline\n{config}\n" + "=" * 80)

    if not any([config.train_and_store_grad, config.compute_embedding,
                config.compute_value, config.attribute]):
        raise SystemExit("No stage selected. Pass at least one of --train_and_store_grad, "
                         "--compute_embedding, --compute_value, --attribute.")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config)
    ctx = _make_ctx(config, device)
    print(f"Using device: {device}")

    need_data = config.train_and_store_grad or config.compute_value
    dataset = _load_dataset(config) if need_data else None

    if config.train_and_store_grad:
        stage_train(config, device, ctx, dataset)
    if config.compute_embedding:
        stage_embedding(config, device)
    if config.compute_value:
        stage_value(config, device, ctx, dataset)
    if config.attribute:
        stage_attribute(config)

    print("\n" + "=" * 80 + f"\nDone. Outputs under: {config.run_dir}\n" + "=" * 80)


if __name__ == '__main__':
    main()
