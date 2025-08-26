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




# Implementation Plan for "Gradient Projection with LoRA-style"

As a starting point, we should just focus on Linear layers. After finishing the implementation, let's add a test in 'Test' which evaluate the gradient projection implementation on a two-layer MLP. The criterias are (1) the added LoRA module does not interfare with model training. The model training dynamics should be exactly the same with and without the added LoRA module. (2) the gradient projection value are the same as (or very close to, up to numerical error) the gradient projection computed naively. 

## Detailed plan and sample code
TODO
