# JAX GhostEngines Migration Plan

## Executive Summary

This document outlines a comprehensive plan for porting the GhostEngines package from PyTorch to JAX. The migration will leverage JAX's functional programming paradigm, automatic differentiation capabilities, and built-in support for per-sample gradients to create a more efficient and scalable implementation.

## Current PyTorch Architecture Analysis

### Core Components

1. **GhostEngineManager** (`engine_manager.py`)
   - Unified interface for engine initialization
   - Manages validation data attachment
   - Coordinates gradient computation and saving

2. **GradDotProdEngine** (`graddotprod_engine.py`)
   - Computes gradient dot products between validation and training samples
   - Uses PyTorch hooks to intercept forward/backward passes
   - Accumulates gradients in custom attributes (`train_grad`, `grad_dot_prod`)

3. **Autograd Hooks** (`autograd_grad_sample_dotprod.py`)
   - Registers forward hooks to capture activations
   - Registers backward hooks to compute gradients and dot products
   - Manages gradient accumulation without interfering with optimizer

4. **Layer-Specific Samplers** (`supported_layers_grad_samplers_dotprod.py`)
   - Custom gradient computation for Linear, Embedding, LayerNorm, Conv layers
   - Implements "ghost computation" optimization based on tensor dimensions

### Key Challenges in Current Implementation

- Relies heavily on PyTorch's hook mechanism
- Stores intermediate values as layer attributes (stateful)
- Manual gradient accumulation and management
- Complex interaction with optimizer and DDP

## JAX Architecture Design

### Core Advantages of JAX

1. **Functional Paradigm**: Pure functions make gradient tracking explicit
2. **vmap**: Native support for per-sample gradient computation
3. **Custom VJP**: Fine-grained control over backward pass
4. **Pytrees**: Natural handling of nested parameter structures
5. **JIT Compilation**: Potential performance improvements

### Proposed JAX Architecture

```
ghostEngines_jax/
├── __init__.py
├── engine_manager.py       # JAX version of engine manager
├── graddotprod_engine.py   # Core gradient dot product logic
├── gradient_utils.py       # JAX-specific gradient utilities
├── layer_handlers.py       # Layer-specific gradient computations
└── state_management.py     # Functional state handling
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 State Management Module
```python
# state_management.py
@dataclass
class GhostEngineState:
    """Immutable state for ghost engine computations."""
    params: PyTree
    val_data: Tuple[Array, Array]
    dot_product_log: List[Dict]
    iteration: int
    
    def update(self, **kwargs) -> 'GhostEngineState':
        """Return new state with updated fields."""
        return replace(self, **kwargs)
```

#### 1.2 Gradient Utilities
```python
# gradient_utils.py
def compute_per_sample_gradients(loss_fn, params, x_batch, y_batch):
    """Compute gradients for each sample using vmap."""
    single_loss = lambda p, x, y: loss_fn(p, x[None, ...], y[None, ...])
    return vmap(grad(single_loss), in_axes=(None, 0, 0))(params, x_batch, y_batch)

def compute_gradient_dot_products(val_grads, train_grads):
    """Compute dot products between validation and training gradients."""
    return tree_map(
        lambda v, t: vmap(lambda t_i: jnp.vdot(v.flatten(), t_i.flatten()))(t),
        val_grads, train_grads
    )
```

### Phase 2: Core Engine Implementation (Week 2)

#### 2.1 GradDotProdEngine JAX Version
```python
# graddotprod_engine.py
class GradDotProdEngineJAX:
    def __init__(self, loss_fn, val_batch_size, ...):
        self.loss_fn = loss_fn
        self.val_batch_size = val_batch_size
        self.state = None
    
    def compute_step(self, params, x_train, y_train, x_val, y_val):
        """Single training step with gradient dot product computation."""
        # Compute validation gradients
        val_loss, val_grads = value_and_grad(self.loss_fn)(params, x_val, y_val)
        
        # Compute per-sample training gradients
        train_grads_per_sample = compute_per_sample_gradients(
            self.loss_fn, params, x_train, y_train
        )
        
        # Compute average training gradient for optimizer
        train_grad_avg = tree_map(lambda x: x.mean(axis=0), train_grads_per_sample)
        
        # Compute dot products
        dot_products = compute_gradient_dot_products(val_grads, train_grads_per_sample)
        
        return train_grad_avg, dot_products, val_loss
```

#### 2.2 Custom VJP for Activation Capture
```python
def capture_activations_transform(f):
    """Transform function to capture intermediate activations."""
    @custom_vjp
    def f_with_capture(params, x):
        return f(params, x)
    
    def f_fwd(params, x):
        activations = {}
        y, aux = f(params, x, capture_intermediates=True)
        return y, (params, x, aux['intermediates'])
    
    def f_bwd(residuals, g):
        params, x, activations = residuals
        # Custom backward pass with access to activations
        return grad(f)(params, x) * g, None
    
    f_with_capture.defvjp(f_fwd, f_bwd)
    return f_with_capture
```

### Phase 3: Layer-Specific Optimizations (Week 3)

#### 3.1 Ghost Computation Heuristic
```python
# layer_handlers.py
def should_use_ghost_computation(weight_shape, activation_shape, grad_shape):
    """Determine if ghost computation is more efficient."""
    T = np.prod(activation_shape[1:-1]) if len(activation_shape) > 2 else 1
    d = activation_shape[-1]
    p = grad_shape[-1]
    num_params = np.prod(weight_shape)
    return 2 * T**2 <= num_params

@jit
def compute_linear_gradient_dotprod(weight, act_train, act_val, grad_train, grad_val):
    """Optimized gradient computation for linear layers."""
    if should_use_ghost_computation(weight.shape, act_train.shape, grad_train.shape):
        # Use ghost computation trick
        return ghost_compute_dotprod(act_train, act_val, grad_train, grad_val)
    else:
        # Standard computation
        return standard_compute_dotprod(weight, act_train, act_val, grad_train, grad_val)
```

### Phase 4: Integration with Training Loop (Week 4)

#### 4.1 Training Loop Adapter
```python
class JAXTrainer:
    def __init__(self, model_fn, optimizer, config):
        self.model_fn = model_fn
        self.opt_state = optimizer.init(params)
        self.ghost_engine = GradDotProdEngineJAX(loss_fn, config)
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, params, opt_state, batch, val_batch):
        """Single training step with ghost engine."""
        x_train, y_train = batch
        x_val, y_val = val_batch
        
        # Compute gradients and dot products
        grads, dot_products, metrics = self.ghost_engine.compute_step(
            params, x_train, y_train, x_val, y_val
        )
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, dot_products, metrics
```

#### 4.2 Compatibility Layer
```python
class PyTorchCompatibilityWrapper:
    """Wrapper to maintain API compatibility with PyTorch version."""
    def __init__(self, jax_engine):
        self.jax_engine = jax_engine
    
    def attach(self, optimizer):
        # Map PyTorch optimizer to JAX optimizer
        pass
    
    def attach_train_batch(self, X_train, Y_train, iter_num):
        # Convert PyTorch tensors to JAX arrays
        pass
```

## Migration Milestones

### Milestone 1: Proof of Concept (Week 1-2)
- [ ] Implement basic gradient dot product computation in JAX
- [ ] Demonstrate per-sample gradient computation using vmap
- [ ] Create simple test comparing PyTorch and JAX outputs

### Milestone 2: Core Engine (Week 3-4)
- [ ] Port GradDotProdEngine core logic
- [ ] Implement state management system
- [ ] Add support for common layers (Linear, LayerNorm)

### Milestone 3: Optimizations (Week 5-6)
- [ ] Implement ghost computation heuristic
- [ ] Add JIT compilation for critical paths
- [ ] Optimize memory usage for large models

### Milestone 4: Integration (Week 7-8)
- [ ] Create training loop integration
- [ ] Add checkpointing and logging
- [ ] Implement distributed training support

### Milestone 5: Testing & Documentation (Week 9-10)
- [ ] Comprehensive unit tests
- [ ] Performance benchmarks vs PyTorch
- [ ] Migration guide and API documentation

## Technical Considerations

### Memory Management
- JAX's functional approach may increase memory usage
- Use gradient checkpointing for large models
- Consider using `jax.lax.scan` for sequential computations

### Performance Optimizations
- JIT compilation crucial for performance
- Batch operations using vmap
- Profile and optimize hot paths

### Distributed Training
- Use `jax.pmap` for data parallelism
- Consider `jax.experimental.maps` for model parallelism
- Ensure compatibility with existing DDP setup

## Testing Strategy

### Unit Tests
```python
def test_gradient_dot_product_equivalence():
    """Ensure JAX and PyTorch produce same dot products."""
    # Setup identical models and data
    torch_engine = GradDotProdEngine(...)
    jax_engine = GradDotProdEngineJAX(...)
    
    # Compute dot products
    torch_dots = torch_engine.compute(...)
    jax_dots = jax_engine.compute(...)
    
    # Assert equivalence within tolerance
    assert_allclose(torch_dots, jax_dots, rtol=1e-5)
```

### Integration Tests
- End-to-end training with small model
- Verify checkpointing and recovery
- Test distributed training setup

### Performance Benchmarks
- Measure time per iteration
- Compare memory usage
- Profile gradient computation overhead

## Risk Assessment

### High Risk
- **Numerical Differences**: JAX and PyTorch may produce slightly different results
  - *Mitigation*: Extensive testing with tolerance thresholds
  
- **API Breaking Changes**: JAX's functional style differs significantly
  - *Mitigation*: Provide compatibility wrapper

### Medium Risk
- **Performance Regression**: Initial JAX version may be slower
  - *Mitigation*: Profile and optimize iteratively
  
- **Memory Issues**: Functional approach may use more memory
  - *Mitigation*: Implement gradient checkpointing

### Low Risk
- **Feature Parity**: Some PyTorch features may be hard to replicate
  - *Mitigation*: Document limitations clearly

## Conclusion

The migration to JAX offers significant advantages through its functional programming model, native per-sample gradient support, and potential performance improvements through JIT compilation. The proposed architecture maintains the core functionality of the PyTorch implementation while leveraging JAX's strengths.

Key success factors:
1. Maintaining numerical equivalence with PyTorch version
2. Achieving comparable or better performance
3. Providing clear migration path for existing users
4. Comprehensive testing at each phase

The phased approach allows for incremental validation and reduces risk of major issues. The compatibility layer ensures existing code can migrate gradually.