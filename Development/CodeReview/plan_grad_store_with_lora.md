# Efficient Gradient Projection with LoRA-style implementation

## 1. Motivation

### The Problem
Computing per-sample gradients and pairwise gradient similarities is crucial for data-centric ML research (data valuation, influence functions, etc.), but faces severe computational challenges:

- **Naive approach**: Requires batch size = 1, backpropagation for each sample, storing model-sized gradient vectors
- **Memory bottleneck**: Gradient vectors are huge (size of model parameters)
- **Compute bottleneck**: O(n) backpropagation passes for n samples
- **Storage bottleneck**: Cannot store full gradients for large datasets

## 2. Ghost Dot-Product (already implemented): Single Batch Solution

### Key Insight
We can compute pairwise gradient dot products **without instantiating gradient vectors** by leveraging information already computed during standard backpropagation.

### Mathematical Derivation

For a linear layer $s = aW$ where:
- $W ∈ ℝ^{d_1 × d_2}$ is the weight matrix
- $a^{(i)} ∈ ℝ^{d_1}$ is input for sample i
- $b^{(i)} = \frac{∂ℓ^{(i)}}{∂s^{(i)}}$ is the output gradient for sample i

The gradient decomposition is:
$$\frac{∂ℓ^{(i)}}{∂W} = a^{(i)} ⊗ b^{(i)}$$

### Computing Gradient Dot Products

#### Non-Sequential Data
For two samples in the same batch:
$$\frac{∂ℓ^{(1)}}{∂W} \cdot \frac{∂ℓ^{(2)}}{∂W} = (a^{(1)} ⊗ b^{(1)}) \cdot (a^{(2)} ⊗ b^{(2)}) = (a^{(1)} \cdot a^{(2)})(b^{(1)} \cdot b^{(2)})$$

**Key**: Compute dot products between activations and between gradients separately, then multiply!

#### Sequential Data (Sequence Length T)
$$\frac{∂ℓ^{(1)}}{∂W} \cdot \frac{∂ℓ^{(2)}}{∂W} = \text{sum}[(b^{(1)}(b^{(2)})^T) \odot (a^{(1)T}a^{(2)})]$$

### Benefits
- **Single backpropagation** for all pairwise similarities in a batch
- **No gradient materialization** - only use activations and output gradients
- **Memory efficient** - no model-sized vectors needed

### Limitation
Cannot fit entire large dataset in a single batch!

## 3. Gradient Projection with LoRA-style Implementation (TODO)

### The Challenge
For large datasets, we need to:
1. Process data in multiple batches
2. Store gradient information for later similarity computation
3. But storing full gradients is prohibitive

### The Solution: Gradient Projection
Use random projection to reduce gradient dimensionality while preserving similarities (Johnson-Lindenstrauss lemma).

### Mathematical Framework

#### Traditional Approach (Inefficient)
1. Compute full gradient: $\text{vec}(DW) ∈ ℝ^{d_1 \cdot d_2}$
2. Project: $P \cdot \text{vec}(DW)$ where $P ∈ ℝ^{k × (d_1 \cdot d_2)}$, $k \ll d_1 \cdot d_2$

#### What we will do

High‑level idea: LoRA‑style gradient projection

1. **Per‑sample layer gradient has a Kronecker (outer‑product) form.**
   For a linear/conv layer treated as a matrix multiply $x_o = W x_i$ (unfolding convs as usual), the per‑sample weight gradient can be written as a sum over sequence/position index $t$:

   $$
   \mathrm{vec}(\Delta W)
   \;=\;
   \sum_{t=1}^{T} \; x_{i,t}\;\otimes\; \mathcal{D}x_{o,t},
   $$

   where $x_{i,t}$ is the forward activation “into” the layer at position $t$, and $\mathcal{D}x_{o,t}$ is the upstream (pre‑activation) gradient “out of” the layer at the same position.

2. **Project the gradient without ever materializing it** by **imposing a Kronecker structure on the projection itself**:

   $$
   P \;\triangleq\; P_i \otimes P_o, 
   \quad P_i \in \mathbb{R}^{k_i\times n_i},\;
   P_o \in \mathbb{R}^{k_o\times n_o}.
   $$

   Using $(A\!\otimes\!B)(u\!\otimes\!v)=(Au)\!\otimes\!(Bv)$,

   $$
   P\,\mathrm{vec}(\Delta W)
   \;=\;\sum_{t=1}^T (P_i x_{i,t}) \otimes (P_o\,\mathcal{D}x_{o,t}),
   $$

   i.e., **project forward activations and backward signals once** and form their outer products in the *low‑dimensional* spaces.

3. **Realize this as a zero‑impact LoRA‑style side branch.**
   Add an auxiliary path in parallel to $W$:

   $$
   y \;=\; W x \;+\; \underbrace{P_o^{\!\top} \, G \, P_i}_{\text{adapter}}\; x,
   $$

   with **$G \in \mathbb{R}^{k_o\times k_i}$** the only trainable matrix in the branch, and **initialize $G=0$** so the model’s outputs and the main gradient w\.r.t. $W$ are unchanged. In this construction

   $$
   \frac{\partial \ell}{\partial G}
   \;=\;\sum_{t=1}^T \big(P_o^{\!\top}\mathcal{D}x_{o,t}\big)\;\big(P_i x_{i,t}\big)^{\!\top},
   $$

   so **$\mathrm{vec}(\tfrac{\partial \ell}{\partial G}) = (P_i \otimes P_o)\,\mathrm{vec}(\Delta W)$**.
   Therefore, the **per‑sample gradient of $G$** *is exactly the projected per‑sample gradient* you want.


### Architecture Design

```
     Input x_i ──→ [Module W] ──→ Output x_o
         ↓                             ↓
    [Proj P_i]                    [Proj P_o]  
         ↓                             ↓
    x_i_projected               grad_o_projected
         ↓                             ↓
         └──────→ [Bottleneck] ←──────┘
                        ↓
                Projected Gradient
```

### Implementation Details

#### 1. Add-on Modules
- Projection layers are **parallel add-ons**, not modifications
- Don't interfere with main forward/backward pass
- Can be added/removed without changing model

#### 2. Bottleneck Layer Role
- **Aggregation point** for projected quantities
- Computes projected gradients without full materialization
- Handles sequential data by summing over time steps

#### 3. Choosing projection dimension $k_i, k_o$ per layer

You typically fix a **target projection size $k_\ell$** per layer. To minimize forward/backward cost $n_i k_i + n_o k_o$ subject to $k_i k_o = k_\ell$, set

$$
\frac{k_i}{k_o} \approx \sqrt{\frac{n_o}{n_i}}
\;\;\Rightarrow\;\;
k_i \approx \sqrt{k_\ell \frac{n_o}{n_i}},\qquad
k_o \approx \sqrt{k_\ell \frac{n_i}{n_o}}.
$$

Round to integers and cap to $[1, n_i]$ and $[1, n_o]$.

#### 4. Initialization of $P_i, P_o$

You want a **near‑isometry** so inner products are preserved after projection. Any JL‑style subgaussian works:

* **Gaussian JL (simple, good default):**
  $ (P_i)_{ab} \sim \mathcal{N}(0, 1/k_i)$, $ (P_o)_{ab} \sim \mathcal{N}(0, 1/k_o)$.
  This satisfies $\mathbb{E}[P^\top P] \approx I$ (in expectation) and works well on GPU.

* **Rademacher JL (memory‑light):**
  entries $\pm 1/\sqrt{k_i}$ and $\pm 1/\sqrt{k_o}$ with prob. 1/2.

* **Row‑orthonormal (more numerically stable):**
  draw Gaussian, do QR, take the first $k_i$ (resp. $k_o$) **row‑orthonormal** rows; then $P_i P_i^\top = I_{k_i}$, $P_o P_o^\top = I_{k_o}$, and consequently
  $(P_i\!\otimes\!P_o)(P_i\!\otimes\!P_o)^\top = I_{k_i k_o}$.
  This gives exact energy preservation along the projection rows.

> **Important:** keep $G$ **zero‑initialized** and **do not update** $P_i, P_o$. They are fixed random projections (seed them once for reproducibility).


#### 5. What to store and how

* For each example $u$, write the concatenated vector $g_u$ (flattened per‑layer $G$ gradients) to a single row in a memory‑mapped array on disk, typically `float16` or `bfloat16`.
* Keep a JSON/NPY sidecar with the random seeds and $\{k_i^\ell, k_o^\ell\}$ so you can **reproduce** the projection later.
* For pairwise similarities, compute blockwise $G G^\top$ over chunks to stay within RAM.


### Tips

1. **Precision.**
   Computing per‑sample grads can be memory‑heavy. Mixed precision is fine for forward/backward, but compute **grads for $G$ in FP32 or BF16**, then cast to `float16` for storage to reduce JL distortion from rounding.

2. **Layer‑wise scaling.**
   If you mix different projection schemes across layers, verify that **$\Vert P_\ell\Vert_F^2$** is consistent so that each layer’s contribution is on a comparable scale. Using the JL scalings above (variance $1/k_i$, $1/k_o$) or row‑orthonormal $P_i,P_o$ automatically normalizes.

3. **Autograd plumbing.**
   You do **not** need custom backward. All major frameworks will give $\partial \ell/\partial G$ for free. Just be careful to collect **per‑sample**, not batch‑summed

4. **The new code should be put in 'ghostEngines/gradProjection'. An existing file, 'lora_modules.py', is in 'ghostEngines/gradProjection'. Feel free to leverage and adapt it.**


### Goal of Implementation

1. **Non-invasive**: Original model training unchanged
2. **Efficient**: Only compute when needed
3. **Modular**: Plug-and-play for the linear/embedding layers for any architecture
4. **Make it integrate with the full ghostEngine package (user interface should be also through 'engine_manager.py')**.


## Detailed plan and sample code

This section turns the design above into a concrete, implementation-ready plan with file layout, integration points, detailed steps, and sample code snippets. The new engine computes and stores per-sample projected gradients G for large datasets using LoRA-style projections without changing the model’s behavior or training dynamics.


**Scope and Outputs**
- Output: For each training example `u`, a single projected gradient vector `g_u` obtained by concatenating per-layer vec(dL/dG_ℓ) for all adapted layers ℓ. Stored as a memory-mapped array on disk plus a metadata sidecar describing the projection setup and layer ordering.
- Non-invasive: Model forward/backward and optimizer behavior remain identical. The LoRA side branches are zero-impact (initialized to produce exact zeros) and not part of the optimizer param groups.
- Efficiency: Projections use small ranks per layer. Hooks compute per-sample contributions in low dimension and stream to disk in shards to bound memory.


**Files to Add (no core file modifications)**
- Add: `ghostEngines/gradProjection/gradproj_engine.py` — Standalone engine (hooks, storage, orchestration). No dependency on training loop or config_file.
- Add: `ghostEngines/gradProjection/autograd_gradproj.py` — Hook utilities for forward/backward capture and per-sample G-grad computation.
- Add: `Examples/ghost_gradproj_mlp.py` — Example for MLP; includes non-interference and naive-equality routines.
- Add: `Examples/ghost_gradproj_lm.py` — Example for a fixed pretrained LM checkpoint.
- Add: `Test/test_gradproj_mlp.py` — Unit test verifying criteria on a two-layer MLP.

Notes:
- Do NOT modify `config_file.py`. Examples define and validate their own CLI.
- Optional future integration with `engine_manager.py` can be considered later, but is out of scope now.


**Configuration (local to examples in Folder 'Examples'; strict, no fallback)**
- Required example flags (validated; raise on missing):
  - `--proj_rank_total` (int), `--proj_rank_min` (int)
  - `--proj_layers` (str)
  - `--proj_dtype` (str: float16|bfloat16|float32)
  - `--proj_seed` (int)
  - `--proj_dir` (str)
- Optional flags: `--proj_row_orthonormal`, `--proj_include_embeddings`, `--proj_include_conv2d`.
The engine is constructed directly in examples and does not rely on training configs.


**Layer coverage (GPT-2 priority)**
- Support: `nn.Linear`, `transformers.pytorch_utils.Conv1D`, `nn.Embedding`, `nn.LayerNorm` (ignored for LoRA projection), and optionally `nn.Conv2d`.
- For GPT-2, most dense layers are `Conv1D`; treat as Linear with weight `[out, in]`.


### Step-by-step Implementation

1) Projection shape helper
- Implement `choose_ki_ko(n_i, n_o, k_total, k_min)` computing k_i,k_o using the ratio rule:
  - r = sqrt(n_o / n_i); k_i = clamp(round(sqrt(k_total) * r), 1..n_i), k_o = max(1, k_total // k_i); adjust by ±1 to minimize |k_i·k_o − k_total| under bounds; enforce k_i ≥ k_min and k_o ≥ k_min (raise if impossible).

Sample:
```
def choose_ki_ko(n_i: int, n_o: int, k_total: int, k_min: int) -> tuple[int, int]:
    import math
    if min(n_i, n_o) <= 0 or k_total <= 0:
        raise ValueError("Invalid dims for projection")
    root = int(math.sqrt(k_total))
    r = math.sqrt(n_o / max(1, n_i))
    k_i = max(k_min, min(n_i, max(1, int(round(root * r)))))
    k_o = max(k_min, min(n_o, max(1, k_total // k_i)))
    # Adjust to hit product near k_total under bounds
    best = (k_i, k_o)
    best_err = abs(k_i * k_o - k_total)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ki, ko = k_i + di, k_o + dj
            if k_min <= ki <= n_i and k_min <= ko <= n_o:
                err = abs(ki * ko - k_total)
                if err < best_err:
                    best, best_err = (ki, ko), err
    ki, ko = best
    if ki < k_min or ko < k_min:
        raise ValueError("Cannot satisfy k_min constraints")
    return ki, ko
```

2) Non-invasive side-car projections (no forward modification)
- Do NOT wrap or replace model layers. Build a projection bank per selected layer ℓ with fixed `P_i^ℓ ∈ R^{k_i×n_i}` and `P_o^ℓ ∈ R^{k_o×n_o}` stored in the engine (not the model).
- Initialize via JL (reuse lora_modules initializers to generate matrices), set `requires_grad=False`. Store per-layer metadata and vector slices for concatenation.

3) Hooking for per-sample projected gradients
- Register hooks directly on base modules; compute projections with side-car `P_i, P_o`.
- Shapes handling and core logic (Linear/Conv1D-like) follow the Kronecker derivation; Embedding uses index-gather for `A_proj`.

Sample (autograd hook core):
```
# autograd_gradproj.py (core idea)
import torch

def _flatten_tokens(x):
    if x.dim() <= 2:
        return x.unsqueeze(1)
    B, *mid, D = x.shape
    T = int(torch.tensor(mid).prod().item())
    return x.reshape(B, T, D)

def fwd_hook_store_inputs(layer, inputs):
    layer._ghost_A_raw = inputs[0].detach()

def bwd_hook_compute_proj(layer, grad_input, grad_output, P_i, P_o):
    B_out = grad_output[0].detach()
    A_raw = getattr(layer, '_ghost_A_raw', None)
    if A_raw is None:
        raise RuntimeError('Missing cached activations for GradProjection')
    A = _flatten_tokens(A_raw)
    B = _flatten_tokens(B_out)
    A_proj = torch.matmul(A, P_i.t())      # [B,T,k_i]
    G_proj = torch.matmul(B, P_o.t())      # [B,T,k_o]
    gradG = torch.einsum('btk,btj->bkj', G_proj, A_proj)  # [B,k_o,k_i]
    layer._ghost_grad_proj = gradG.to(torch.float32)
    delattr(layer, '_ghost_A_raw')
    return None
```

4) Engine class (standalone) and lifecycle
- `GradProjLoraEngine(module, **proj_kwargs)` builds projections and hooks without touching optimizer or training loop.
- Key methods:
  - `attach()`: register hooks on selected layers; `detach()`: remove handles and buffers.
  - `collect_batch()`: concatenate per-layer `layer._ghost_grad_proj.reshape(B,-1)` into `[B,K_total]`, save shard to `--proj_dir` with `metadata.json` on first call, and return the tensor.
- The engine never writes to `param.grad` or optimizer states.

Skeleton:
```
# ghostEngines/gradProjection/gradproj_engine.py
class GradProjLoraEngine:
    def __init__(self, module, proj_layers, proj_rank_total, proj_rank_min, proj_seed,
                 proj_dtype, proj_dir, proj_row_orthonormal=False,
                 include_embeddings=False, include_conv2d=False):
        ...
    def attach(self):
        ...
    def detach(self):
        ...
    def collect_batch(self):
        ...
```

5) Example scripts (no training loop/config changes)
- `Examples/ghost_gradproj_mlp.py`:
  - CLI adds projection args and `--mode {project,non_interf,naive_check}`.
  - project: run forward+backward, `engine.collect_batch()` to save vectors.
  - non_interf: enforce deterministic algorithms, run T steps twice (with/without engine), assert identical loss and parameters at each step.
  - naive_check: compare `g_engine` to `g_naive` for one batch (see below for naive computation).
- `Examples/ghost_gradproj_lm.py`:
  - Load GPT-2 small (or local checkpoint); call `ghostEngines.transformers_support.forward_swapper(model)`; attach engine to selected layers; run one backward; collect and save vectors.

6) Automated test (Test/test_gradproj_mlp.py)
- Construct a two-layer MLP and synthetic batch(es) with fixed seed and deterministic flags.
- Non-interference (criterion 1):
  - Train T steps with SGD/Adam without engine; record loss series and parameter clones per step.
  - Reset model/optimizer to identical initial state; attach engine; train same T steps; assert losses and parameters match exactly (`torch.equal`) under deterministic settings. If the platform cannot guarantee bitwise determinism, use `allclose` with tight tol and document the flags in the test.
- Naive equality (criterion 2):
  - After one backward with engine attached, compute `g_engine = engine.collect_batch()`.
  - Compute `g_naive` by materializing per-sample grads for each adapted layer and projecting with `P = P_i ⊗ P_o`:
    • Linear: `A_raw -> [B,T,d], B_out -> [B,T,p]`. Form `G_b = Σ_t A[b,t]^T @ B[b,t]` (or via einsum), then `vec_proj_b = (P_i ⊗ P_o) vec(G_b)`; or equivalently `Σ_t (P_i A[b,t]) ⊗ (P_o B[b,t])`.
    • Embedding: build full grad via index_add over vocabulary then project.
  - Assert `torch.allclose(g_engine, g_naive, rtol=1e-5, atol=1e-6)`.

7) DDP and storage (as needed by examples)
- Single-process by default. With DDP, write to `--proj_dir/rank{rank}` and merge offline.
- Save shards as `.pt` with `{'proj': tensor, 'iter': i}` and write `metadata.json` on first save: layer names, `(k_i,k_o)`, dtype, seed.

8) Precision and performance
- Compute: Keep A_proj and G_proj in model/training dtype; accumulate `dL/dB` in FP32 to minimize rounding error; cast to `proj_dtype` only at disk write time.
- Memory: Drop `_a_proj` immediately after use; never keep per-layer grads past `aggregate_and_log`.
- Throughput: If overhead is high, gate the computation by a frequency flag `--proj_every` to only compute on every k-th iteration (raise error if not provided; or integrate with `proj_save_interval`).

9) Testing and verification
- Smoke test (GPT2-Small, Pile):
  - Train with `--method GradProjLoRA --batch_size 2 --proj_rank_total 256 --proj_rank_min 8 --proj_layers 'attn.c_attn,mlp.c_fc,lm_head' --proj_dtype bfloat16 --proj_seed 1234 --proj_save_interval 10 --proj_dirname grad_proj`
  - Confirm: a) Training loss/metrics identical to Regular training; b) Shard files appear under result dir; c) Metadata file lists layers and vector layout; d) Per-batch stored tensor has shape `[batch_size, K_total]`.
- Unit check: For a single adapted Linear layer, verify that batch-summed `dL/dB` from hooks equals `autograd.grad(loss.sum(), B.weight)` up to dtype tolerance.

10) Offline use example (pairwise similarities)
```
import torch, glob, json, os
root = '<result_dir>/grad_proj/rank0'
meta = json.load(open(os.path.join(root, 'metadata.json')))
files = sorted(glob.glob(os.path.join(root, 'proj_iter_*.pt')))
vecs = []
idxs = []
for f in files:
    pack = torch.load(f, map_location='cpu')
    vecs.append(pack['proj'].float())
    idxs.extend(pack['batch_idx'])
G = torch.cat(vecs, dim=0)  # [N, K_total]
# Blockwise similarity to save RAM
bs = 8192
sims = torch.empty((G.size(0), G.size(0)))
for i in range(0, G.size(0), bs):
    Gi = G[i:i+bs]
    sims[i:i+bs] = Gi @ G.t()
```


### Optional extension: rectangular G (ki×ko)
- If exact `k_total = k_i·k_o` is required, extend `lora_modules` to support rectangular `B` by replacing the single `rank` with `(rank_i, rank_o)`; `logix_lora_B: Linear(rank_i, rank_o, bias=False)` and set `logix_lora_C: Linear(rank_o, out, ...)`, `logix_lora_A: Linear(in, rank_i, ...)`. The hook math remains unchanged; only shapes vary. Update `AdapterMeta` and vector layout accordingly.


### Guardrails and errors (no fallback)
- If no layer matches `proj_layers`, raise immediately.
- If any required CLI arg is missing/empty under `GradProjLoRA`, raise in config initialization.
- On hook execution, if `_a_proj` is missing or tensor ranks mismatch, raise to surface logic errors.
- Disallow using this engine together with `GradDotProd` in the same run.


### Notes on integration with existing utilities
- Reuse `ghostEngines/transformers_support.forward_swapper` for GPT-2 to ensure correct per-sample gradient semantics, same as the dot-product engine.
- Maintain code style: PEP 8, type hints for new public APIs, and respect project’s import ordering.


This plan stays minimal, leverages the existing `lora_modules.py`, and leaves the model and optimizer unchanged while providing accurate, efficient projected per-sample gradients suitable for large-scale similarity analysis.

