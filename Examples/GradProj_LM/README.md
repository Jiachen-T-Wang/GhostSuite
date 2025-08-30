# Efficient Per-sample Gradient Projection for Language Models

`GradDotProdEngine` excels at **gradient-based online data selection** (e.g., [GREATS](https://openreview.net/pdf?id=232VcN8tSx)), computing pairwise gradient similarities on-the-fly without requiring additional backpropagation passes. While powerful for online scenarios, `GradDotProdEngine` alone cannot handle **gradient-based offline data selection** on large datasets. The key constraint is that **we cannot fit an entire large dataset into a single batch**. For offline selection or similar use cases requiring pairwise gradient similarity computation with respect to a fixed model checkpoint across extensive datasets, we need a different approach—one that can persist per-sample gradients (or their projections) to disk for subsequent analysis.

## Overview

`GradProjLoraEngine` is another engine we develop that achieves the following: 
- Computes projected gradients efficiently without modifying the backpropagation implementation
- Avoids instantiating memory-intensive projection matrices
- Enables scalable gradient analysis across datasets of any size

This methodology is adapted from [**LogIX**](https://arxiv.org/abs/2405.13954), originally developed to accelerate influence function computation.


## How the engine works
- Per‑sample layer gradients have a Kronecker (outer‑product) form: for each position \(t\), \(\mathrm{vec}(\Delta W) = \sum_{t} x_{i,t} \otimes \mathcal{D}x_{o,t}\). We project without materializing \(\Delta W\) by using a Kronecker‑structured random projection \(P = P_i \otimes P_o\) so \(P\,\mathrm{vec}(\Delta W) = \sum_{t} (P_i x_{i,t}) \otimes (P_o \, \mathcal{D}x_{o,t})\).
- A zero‑impact LoRA‑style side branch \(y = W x + P_o^{\top} \, G \, P_i \, x\) (with \(G\) initialized to zero and \(P_i, P_o\) fixed) makes \(\tfrac{\partial \ell}{\partial G}\) exactly the projected per‑sample gradient, so we can read out low‑dimensional gradients via standard autograd.
- Non‑invasive hooks cache activations and backward signals, apply the fixed projections, aggregate per‑layer results, and stream concatenated projections to disk—preserving inner products (JL) while scaling to large datasets.


## Quick Start

### Get Tokenized Dataset
Process the Pile dataset by domain:
```bash
python Examples/shared/data_processing/tokenize_pile_by_domain.py
```
*Note: This process can take ~24 hours depending on your system. For a minimal example, see `Examples/ghost_gradproj_mlp.py` and `Examples/ghost_gradproj_lm.py`.* 

```bash
cd Examples/GradProj_LM/

./train.sh --batch_size 16 --max_samples 1000
```

### Projection Parameters
- `--proj_layers`: Comma-separated layer patterns to project
  - Default: `"mlp,attn"`
  - Options: `"mlp"`, `"attn"`, `"mlp,attn"`, specific patterns like `"mlp.c_fc"` (these are GPT-family naming styles; needs to adapt to your specific architectures)
- `--proj_rank_total`: Target total projection dimension per layer
- `--proj_rank_min`: Minimum dimension for k_i and k_o
- `--proj_seed`: Random seed for projection matrices
- `--proj_dtype`: Data type for storing projections
- `--proj_row_orthonormal`: Use row-orthonormal projections
- `--include_embeddings`: Include embedding layers in projections
- `--proj_save_interval`: Save projections every N iterations
  - Default: `1`

### Output Parameters
- `--output_dir`: Directory to save projections



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