# Gradient Projection Language Model

This example demonstrates efficient gradient projection computation for GPT-2 models on the Pile dataset using the GradProjLoraEngine.

## Overview

This tool processes the Pile dataset through a GPT-2 model and computes low-dimensional gradient projections for each sample. These projections preserve gradient similarity structure while significantly reducing storage requirements.

## Features

- **Efficient Gradient Projection**: Uses LoRA-style architecture to compute projections without materializing full gradients
- **Flexible Layer Selection**: Choose which layers to project (MLP, attention, embeddings)
- **Configurable Dimensions**: Control projection rank with automatic k_i, k_o optimization
- **Batch Processing**: Handle large datasets with configurable batch sizes
- **Progress Tracking**: Real-time progress bar with loss and timing statistics
- **Incremental Saving**: Projections saved periodically to manage memory

## Installation

No additional installation needed - uses the main GhostSuite environment. Ensure you're in the `Examples/GradProj_LM/` directory when running commands.

## Quick Start

```bash
# Using the launch script (recommended)
./train.sh --batch_size 2 --max_samples 1000

# Or run directly with Python
python main.py --batch_size 2 --max_samples 10 --proj_layers "mlp"

# Process more data with specific configuration
python main.py \
    --architecture GPT2-Small \
    --batch_size 4 \
    --max_samples 1000 \
    --proj_layers "mlp,attn" \
    --proj_rank_total 256 \
    --proj_save_interval 10
```

## Command Line Arguments

### Model Configuration
- `--architecture`: Model architecture (`GPT2-Small`, `GPT2-Medium`, `GPT2-Large`)
  - Default: `GPT2-Small`

### Projection Parameters
- `--proj_layers`: Comma-separated layer patterns to project
  - Default: `"mlp,attn"`
  - Options: `"mlp"`, `"attn"`, `"mlp,attn"`, specific patterns like `"mlp.c_fc"`
- `--proj_rank_total`: Target total projection dimension per layer
  - Default: `256`
- `--proj_rank_min`: Minimum dimension for k_i and k_o
  - Default: `8`
- `--proj_seed`: Random seed for projection matrices
  - Default: `42`
- `--proj_dtype`: Data type for storing projections
  - Default: `"bfloat16"`
  - Options: `"float16"`, `"bfloat16"`, `"float32"`
- `--proj_row_orthonormal`: Use row-orthonormal projections
  - Default: `False`
- `--include_embeddings`: Include embedding layers in projections
  - Default: `False`
- `--proj_save_interval`: Save projections every N iterations
  - Default: `1`

### Processing Parameters
- `--batch_size`: Batch size for processing
  - Default: `2` (small due to GPU memory constraints)
- `--max_samples`: Maximum number of samples to process
  - Default: `None` (process entire dataset)
- `--block_size`: Sequence length for GPT2
  - Default: `1024`
- `--seed`: Random seed for data sampling
  - Default: `42`

### Output Parameters
- `--output_dir`: Directory to save projections
  - Default: `"./projections"`

### System Parameters
- `--model_dtype`: Model precision
  - Default: `"bfloat16"`
  - Options: `"float32"`, `"float16"`, `"bfloat16"`
- `--device`: Device to use
  - Default: `"cuda"`
- `--verbose`: Print detailed progress information
  - Default: `False`

## Output Structure

```
projections/
├── metadata.json           # Projection configuration and layer metadata
├── run_config.json         # Complete run configuration
├── projection_stats.json   # Statistics (loss, timing, tokens)
├── proj_iter_000001.pt     # Projection tensors
├── proj_iter_000002.pt
└── ...
```

### Projection File Format
Each `.pt` file contains a dictionary with:
```python
{
    'proj': torch.Tensor,    # Shape: [batch_size, total_proj_dim]
    'iter': int,             # Iteration number
    'batch_size': int,       # Number of samples in batch
    'batch_idx': List[int]   # Sample indices (optional)
}
```

### Metadata Format
`metadata.json` contains:
```python
{
    'proj_rank_total': int,
    'proj_seed': int,
    'proj_method': str,
    'total_proj_dim': int,
    'layers': [
        {
            'name': str,
            'type': str,
            'original_shape': tuple,
            'k_i': int,
            'k_o': int,
            'slice_start': int,
            'slice_end': int
        },
        ...
    ]
}
```

## Usage Examples

### Example 1: Quick Test
```bash
# Test with minimal data
python main.py --batch_size 1 --max_samples 5 --verbose
```

### Example 2: MLP Layers Only
```bash
# Project only MLP layers with higher rank
python main.py \
    --proj_layers "mlp" \
    --proj_rank_total 512 \
    --batch_size 4 \
    --max_samples 1000
```

### Example 3: Specific Layer Patterns
```bash
# Project specific transformer blocks
python main.py \
    --proj_layers "transformer.h.0,transformer.h.11" \
    --proj_rank_total 128 \
    --include_embeddings
```

### Example 4: Full Dataset Processing
```bash
# Process entire Pile dataset (will take significant time)
python main.py \
    --architecture GPT2-Medium \
    --batch_size 8 \
    --proj_layers "mlp,attn" \
    --proj_save_interval 100 \
    --output_dir "./pile_projections"
```

## Loading Projections

To load and use the saved projections:

```python
import torch
import json
import glob

# Load metadata
with open('projections/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load all projection files
proj_files = sorted(glob.glob('projections/proj_iter_*.pt'))
all_projections = []

for file_path in proj_files:
    data = torch.load(file_path)
    all_projections.append(data['proj'])

# Concatenate all projections
projections = torch.cat(all_projections, dim=0)
print(f"Loaded projections: {projections.shape}")

# Access layer information
for layer in metadata['layers']:
    print(f"Layer: {layer['name']}, k_i={layer['k_i']}, k_o={layer['k_o']}")
    start, end = layer['slice_start'], layer['slice_end']
    layer_proj = projections[:, start:end]
    print(f"  Projection slice: {layer_proj.shape}")
```

## Memory Considerations

- **GPU Memory**: Batch size is limited by GPU memory. Start with batch_size=2 and increase if possible.
- **Disk Space**: Each projection file size = batch_size × total_proj_dim × dtype_size
- **Processing Time**: Roughly 2-5 seconds per batch on a V100 GPU

## Tips for Large-Scale Processing

1. **Start Small**: Test with `--max_samples 100` before full dataset
2. **Monitor Memory**: Use `nvidia-smi` to check GPU memory usage
3. **Save Frequently**: Use `--proj_save_interval 1` for robustness
4. **Use Checkpointing**: The system saves iteration number, allowing resume capability
5. **Consider Precision**: Use `bfloat16` for 2x memory savings over `float32`

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--block_size` (sequence length)
- Use `--proj_dtype bfloat16` or `float16`

### Slow Processing
- Increase `--batch_size` if memory allows
- Use `--proj_save_interval` > 1 to reduce I/O
- Ensure using GPU with `--device cuda`

### Import Errors
- Ensure running from the `Examples/GradProj_LM/` directory
- Check that shared modules and ghostEngines are accessible








## Analysis and Visualization

### Plotting Gradient Projection Errors

The `plot_error_with_dim.py` script analyzes how well lower-dimensional projections approximate gradient dot-products. It provides a CLI and multiple reference modes.

#### CLI

- `--results_dir`: Root directory containing result subfolders (e.g., `Examples/GradProj_GPT2/Results`).
- `--results_pattern`: Pattern to match subfolder names that only differ by `rank_total_K`.
- `--pattern_type`: Interpretation of `--results_pattern` (`regex`|`glob`, default: `regex`).
- `--num_ref`: Number of reference samples to average (default: 50).
- `--max_iters`: Max number of `proj_iter_*.pt` files to load per folder (default: 100).
- `--reference`: Reference mode:
  - `rank=NNN`: Use projections at rank NNN (e.g., `rank=1024`).
  - `full`: Use exact full-model gradients (see below).
  - `full_layers`: Use exact gradients restricted to projected layers.
  - `naive_proj_layers=NNN`: Rebuild P for rank NNN using metadata+seed, apply to exact per-layer grads.

The script validates that all matched subfolders are identical except for the `rank_total_K` token.

#### Examples

- Compare ranks against 1024-D reference (mlp-only, seed 9, row_on False):
```bash
python plot_error_with_dim.py \
  --results_dir Results \
  --results_pattern '^proj_layers_mlp_rank_total_\\d+_rank_min_4_seed_9_dtype_bfloat16_row_on_False_emb_False$' \
  --num_ref 50 --max_iters 100 --reference rank=1024
```

- Compare against exact full gradients (requires full grads to be precomputed):
```bash
python plot_error_with_dim.py \
  --results_dir Results \
  --results_pattern '*min_4_seed_42_dtype_bfloat16_row_on_False_emb_False' \
  --pattern_type glob \
  --num_ref 1 --max_iters 10 --reference full
```

#### Outputs

Saved under `Plots/` with a base name that encodes the matched settings (using `rank_total_ALL`) and the reference tag:
- `...__rmse_vs_dimension.pdf`
- `...__relative_error_vs_dimension.pdf`
- `...__error_analysis_loglog.pdf`
- `...__error_vs_dimension_results.json`

### Exact Full-Model Gradients

Use `compute_full_gradients.py` to compute and store exact per-sample full-model gradients (flattened across all trainable parameters).

Requirements
- Use `--batch_size 1` (enforced).
- Provide a positive `--max_samples` to bound runtime and disk usage.

Example
```bash
python compute_full_gradients.py \
  --architecture GPT2-Small \
  --batch_size 1 \
  --max_samples 10 \
  --device cuda \
  --model_dtype bfloat16 \
  --train_dtype bfloat16 \
  --output_dir ./Results
```

Outputs
- One file per sample: `fullgrad_iter_XXXXXX.pt` with `{'grad': float32 vector, 'iter': int, 'batch_size': 1}`.
- Metadata: `fullgrad_meta.json` with `total_param_dim` and `param_slices` per parameter.
- Directory: `Results/fullgrads_seed_{seed}_arch_{architecture}_dtype_{train_dtype}`.

Notes
- When using `plot_error_with_dim.py` with `--reference full` or `full_layers`, ensure full grads are generated in the same `--results_dir`. The script will auto-discover a `fullgrads*` folder.

## Related Documentation

- [Gradient Projection Engine Documentation](../../Development/CodeReview/doc_grad_store_with_lora.md)
- [Engine Manager Integration](../../Development/CodeReview/doc_gradproj_gpt2.md)
- [GhostSuite Main README](../../README.md)
