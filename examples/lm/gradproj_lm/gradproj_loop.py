"""Projection computation loop for processing Pile dataset."""

import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import gc

# Import from shared modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.dataloader import get_batch_from_dataset


def compute_projections(model, engine, dataset, config, device, ctx, num_iterations):
    """
    Compute gradient projections for the dataset.
    
    Args:
        model: The GPT2 model
        engine: GradProjLoraEngine instance
        dataset: Dataset dictionary with 'train', 'val', 'test' splits
        config: Configuration object
        device: Device to use
        ctx: Autocast context for mixed precision
        num_iterations: Number of iterations to run
    """
    
    # Create generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    
    # Statistics tracking
    total_loss = 0.0
    total_tokens = 0
    batch_times = []

    # Microbatches pooled per saved projection. With n_accum > 1 the engine buffers
    # each microbatch and concatenates into a [n_accum*batch_size, dim] projection so
    # every sample is captured (not just the last microbatch). The per-microbatch loss
    # is NOT divided by n_accum: each keeps its own mean reduction, which the engine's
    # `* batch_size` factor undoes with the local microbatch size, so the pooled result
    # matches a single batch of size n_accum*batch_size.
    n_accum = max(1, int(getattr(config, 'gradient_accumulation_steps', 1)))

    # Progress bar
    pbar = tqdm(range(num_iterations), desc="Computing projections")

    for iter_num in pbar:
        start_time = time.time()

        # Zero gradients for the step.
        model.zero_grad()

        if n_accum > 1:
            engine.begin_step()

        # Run the accumulation microbatch loop, pooling per-sample projections.
        batch_indices = []
        step_loss = 0.0
        for _micro in range(n_accum):
            # Get batch data. Request the actual sampled window offsets so the saved
            # projections are tagged with real sample identities (windows are drawn
            # randomly via the generator, not sequentially).
            X, Y, sample_ix = get_batch_from_dataset(
                split='train',
                batch_size=config.batch_size,
                dataset=dataset,
                block_size=config.block_size,
                device=device,
                device_type=device.type,
                generator=generator,
                return_idx=True,
            )

            # Forward pass with autocast
            with ctx:
                with torch.enable_grad():  # Ensure gradients are enabled
                    outputs = model(X, Y)  # custom GPT: forward(idx, targets)
                    loss = outputs.loss

            # Backward pass to compute gradients (no 1/n_accum scaling; see above).
            loss.backward()

            batch_indices.extend(int(i) for i in sample_ix)
            step_loss += loss.item()
            total_tokens += X.numel()

            if n_accum > 1:
                # Buffer this microbatch's projections before the next backward
                # overwrites the per-layer scratch.
                engine.collect_microbatch()

        # Collect the pooled projection once per step (this also saves it based on
        # proj_save_interval), tagged with the real sampled window offsets.
        try:
            projections = engine.collect_batch(batch_indices)
            proj_shape = projections.shape
            proj_dtype = projections.dtype
        except RuntimeError as e:
            print(f"Warning at iteration {iter_num}: {e}")
            print("Continuing to next iteration...")
            continue

        # Clear cached data to free memory
        if hasattr(engine, 'clear_gradients'):
            engine.clear_gradients()

        # Update statistics (mean microbatch loss over the step)
        batch_loss = step_loss / n_accum
        total_loss += batch_loss
        
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # Update progress bar
        avg_loss = total_loss / (iter_num + 1)
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'proj_shape': str(proj_shape),
            'time': f'{batch_time:.2f}s'
        })
        
        # Verbose output
        if config.verbose and iter_num % 10 == 0:
            print(f"\nIteration {iter_num}/{num_iterations}:")
            print(f"  Batch loss: {batch_loss:.4f}")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Projection shape: {proj_shape}")
            print(f"  Projection dtype: {proj_dtype}")
            print(f"  Batch time: {batch_time:.2f}s")
            print(f"  Tokens processed: {total_tokens}")
        
        # Periodic memory cleanup
        if iter_num % 50 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Final statistics
    print("\n" + "=" * 40)
    print("Projection Computation Statistics:")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Total tokens processed: {total_tokens}")
    print(f"  Average loss: {total_loss / num_iterations:.4f}")
    print(f"  Average batch time: {sum(batch_times) / len(batch_times):.2f}s")
    print(f"  Total time: {sum(batch_times):.2f}s")
    
    # Save final statistics
    import json
    stats_path = os.path.join(config.proj_dir, 'projection_stats.json')
    stats = {
        'num_iterations': num_iterations,
        'total_tokens': total_tokens,
        'average_loss': float(total_loss / num_iterations),
        'total_time_seconds': sum(batch_times),
        'average_batch_time': sum(batch_times) / len(batch_times),
        'batch_size': config.batch_size,
        'block_size': config.block_size,
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")
