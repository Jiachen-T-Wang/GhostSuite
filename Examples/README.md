# Examples Directory

This directory contains several examples demonstrating the use of the Ghost Engine framework for efficient per-sample gradient computation.


## Available Examples


### 1. Minimal Examples (No data preparation required)
Simplified implementations demonstrating core concepts:

- **`ghost_mlp.py`**: Basic GradDotProd usage for MLP models
  - Trains for 10 steps on synthetic data
  - Prints per-parameter gradient dot-products

- **`ghost_gradproj_mlp.py`**: Per-sample gradient projection computation and storage for MLP

- **`ghost_gradproj_lm.py`**: Per-sample gradient projection computation and storage for language models
  - Projects gradients for transformer layers
  - Demonstrates similarity computation from saved projections

**Run minimal examples:**
```bash
python ghost_mlp.py
python ghost_gradproj_mlp.py --mode project --proj_rank_total 64
python ghost_gradproj_lm.py --proj_layers "attn.c_attn,mlp.c_fc"
```

### 2. GradDotProd Language Model (`GradDotProd_LM/`)
Full demonstration of pair-wise gradient dot product computation during language model training on the Pile dataset. Useful for research projects such as online data selection that requires computing gradient similarities during the model training. 

See `Examples/GradDotProd_LM/README.md` for detailed instructions. 


### 3. Gradient Projection Language Model (`GradProj_LM/`)
Full demonstration of per-sample gradient projection computation and storage for a languagem model checkpoint. Useful for research projects such as offline data selection that requires computing gradient similarities for a *fixed* model checkpoint. 

See `Examples/GradProj_LM/README.md` for detailed instructions. 


## How the Ghost Engines Work

### GradDotProd Engine
1. **Batch Concatenation**: Training and validation batches are concatenated for a single forward pass
2. **Gradient Computation**: During backpropagation, the engine computes:
   - Per-parameter gradient dot products between validation and training samples. 
   - Aggregated training gradients are recovered seperately and stored in `.grad` before optimizer step. 

### GradProj Engine
- Uses LoRA-style low-rank projection matrices
- Projects high-dimensional gradients to lower-dimensional space
- Enables efficient per-sample gradient storage without materializing full gradients
- Supports both MLP and attention layer projections

See individual example directories for detailed documentation and configuration options.