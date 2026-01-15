from typing import Dict, List, Optional, Tuple
import math
import os
import threading
import time
import warnings

import torch
import torch.nn as nn

from .supported_layers_grad_samplers_dotprod import (
    _supported_layers_dotprod,
    _create_or_accumulate_train_grad
)

ACCUM_DTYPE = torch.float32


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


_DOTPROD_BENCH_ENABLED = os.getenv("GHOST_DOTPROD_BENCH", "0") == "1"
_DOTPROD_BENCH_EVERY = _env_int("GHOST_DOTPROD_BENCH_EVERY", 1)
_DOTPROD_BENCH_WARMUP = _env_int("GHOST_DOTPROD_BENCH_WARMUP", 5)
_DOTPROD_BENCH_SYNC = os.getenv("GHOST_DOTPROD_BENCH_SYNC", "1") == "1"
_DOTPROD_BENCH_STATS: Dict[str, "_DotprodBenchStats"] = {}
_DOTPROD_BENCH_LOCK = threading.Lock()


class _DotprodBenchStats:
    __slots__ = ("count", "measured", "total_s", "max_s", "min_s", "last_s")

    def __init__(self) -> None:
        self.count = 0
        self.measured = 0
        self.total_s = 0.0
        self.max_s = 0.0
        self.min_s = float("inf")
        self.last_s = 0.0


def _update_dotprod_bench_stats(
    layer: nn.Module,
    backprops: torch.Tensor,
    elapsed_s: float,
    compute_dtype: Optional[torch.dtype],
) -> None:
    layer_name = getattr(layer, "name", layer.__class__.__name__)
    with _DOTPROD_BENCH_LOCK:
        stats = _DOTPROD_BENCH_STATS.get(layer_name)
        if stats is None:
            stats = _DotprodBenchStats()
            _DOTPROD_BENCH_STATS[layer_name] = stats
        stats.count += 1
        if stats.count <= _DOTPROD_BENCH_WARMUP:
            return
        stats.measured += 1
        stats.total_s += elapsed_s
        stats.last_s = elapsed_s
        if elapsed_s > stats.max_s:
            stats.max_s = elapsed_s
        if elapsed_s < stats.min_s:
            stats.min_s = elapsed_s
        should_log = _DOTPROD_BENCH_EVERY > 0 and stats.measured % _DOTPROD_BENCH_EVERY == 0
        if not should_log:
            return
        measured_count = stats.measured
        avg_ms = (stats.total_s / measured_count) * 1e3
        last_ms = stats.last_s * 1e3
        min_ms = stats.min_s * 1e3
        max_ms = stats.max_s * 1e3

    activation = getattr(layer, "activations", None)
    act_shape = tuple(activation.shape) if hasattr(activation, "shape") else None
    bp_shape = tuple(backprops.shape) if hasattr(backprops, "shape") else None
    device = backprops.device
    print(
        "[ghost dotprod bench] "
        f"[{layer_name}] "
        f"avg_ms={avg_ms:.3f} "
        f"act_shape={act_shape} bp_shape={bp_shape} "
        f"bp_dtype={backprops.dtype} compute_dtype={compute_dtype}"
    )


def requires_grad(module: nn.Module) -> bool:
    """
    Checks if any parameters in a specified module require gradients.
    """
    return any(p.initially_requires_grad for p in module.parameters())


class _NamedSavedTensorManager:
    """Captures autograd-saved tensors using a scope stack."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._enabled: bool = False
        self._captured: Dict[str, List[torch.Tensor]] = {}
        self._tensor_meta: Dict[int, str] = {}
        self._layers_by_name: Dict[str, nn.Module] = {}
        self._val_batch_size: int = 0
        self._loss_reduction: str = "mean"

        # Book-keeping of tensor ids that have been used for activations (to avoid double usage across layers).
        self._used_ids: set[int] = set()

        self._debug: bool = os.getenv("GHOST_SAVED_TENSOR_DEBUG", "0") == "1"

    def _get_stack(self) -> List[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _get_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        with self._lock:
            self._enabled = True
            self._captured = {}
            self._tensor_meta = {}
            self._used_ids = set()
        self._get_stack().clear()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._captured = {}
            self._tensor_meta = {}
            self._used_ids = set()
        self._get_stack().clear()

    def push(self, name: str) -> None:
        # If enabled, pushes the module name onto the thread‑local scope stack (forward‑pre hook).
        if not self._get_enabled():
            return
        self._get_stack().append(name)

    def pop(self, name: str) -> None:
        # If enabled, pops the matching module name from the thread‑local stack (forward‑post hook).
        if not self._get_enabled():
            return
        stack = self._get_stack()
        if not stack:
            return
        if stack[-1] == name:
            stack.pop()
            return
        # Fall back to removing the most recent matching scope if present.
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx] == name:
                stack.pop(idx)
                return

    def pack_hook(self, x: torch.Tensor) -> torch.Tensor:
        if not self._get_enabled():
            return x
        stack = self._get_stack()
        with self._lock:
            if stack:
                name = stack[-1]
                self._captured.setdefault(name, []).append(x)
                self._tensor_meta[id(x)] = name
            if self._debug:
                scope = stack[-1] if stack else "<none>"
                print(
                    "[ghost_saved_tensor] "
                    f"tid={threading.get_ident()} scope={scope} "
                    f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                    f"captured_entries={len(self._captured.get(scope, []))}"
                )
        return x

    def unpack_hook(self, x: torch.Tensor) -> torch.Tensor:
        if not self._get_enabled():
            return x

        if not x.is_floating_point():
            return x

        name = self._tensor_meta.get(id(x))
        if not name:
            return x

        layer = self._layers_by_name.get(name)
        if layer is None:
            return x

        input_shape = getattr(layer, "_ghost_input_shape", None)
        train_bs = getattr(layer, "_ghost_train_bs", None)
        total_bs = getattr(layer, "_ghost_total_bs", None)
        if input_shape is None or train_bs is None or total_bs is None:
            return x

        if train_bs <= 0 or train_bs >= total_bs:
            return x

        flat_shape = None
        tokens_per_sample = 1
        if len(input_shape) > 1:
            if len(input_shape) > 2:
                tokens_per_sample = int(math.prod(input_shape[1:-1]))
            flat_shape = (input_shape[0] * tokens_per_sample, input_shape[-1])

        shape = tuple(x.shape)
        if shape != input_shape and (flat_shape is None or shape != flat_shape):
            return x

        if self._loss_reduction == "mean":
            scale = float(total_bs) / float(train_bs)
        else:
            scale = 1.0 / float(train_bs)

        masked = x.clone()
        if shape == input_shape:
            if scale != 1.0:
                masked[:train_bs] = masked[:train_bs] * scale
            masked[train_bs:] = 0
            return masked

        split_idx = train_bs * tokens_per_sample
        if scale != 1.0:
            masked[:split_idx] = masked[:split_idx] * scale
        masked[split_idx:] = 0
        return masked

    def resolve_activation(self, layer: nn.Module) -> Optional[torch.Tensor]:
        name = getattr(layer, "name", None)
        if not name:
            return None

        params = list(layer.parameters(recurse=False))
        param_ids = {id(p) for p in params}

        if self._debug:
            print(f"[resolve_activation] [{name}] param_ids: {param_ids}")

        def _is_param_view(tensor: torch.Tensor) -> bool:
            """
            Checks if a tensor is a view of a parameter.
            This is used to filter out parameter views like weight.t() so 
            we don't mistake them for activation tensors.
            """
            base = getattr(tensor, "_base", None)
            return base is not None and id(base) in param_ids

        input_shape = getattr(layer, "_ghost_input_shape", None)
        flat_shape = None
        if input_shape is not None and len(input_shape) > 1:
            flat_shape = (int(math.prod(input_shape[:-1])), input_shape[-1])

        with self._lock:
            if not self._enabled:
                return None

            if self._debug:
                if not self._captured.get(name, []):
                    print(f"[resolve_activation] [{name}] no captures found")

            capture_pool = self._captured.get(name, [])
            if not capture_pool:
                return None

            non_param = [
                t for t in capture_pool
                if id(t) not in param_ids and not _is_param_view(t) and id(t) not in self._used_ids
            ]
            if not non_param:
                return None

            def _match_shape(shape):
                if shape is None:
                    return None
                matching = [t for t in non_param if tuple(t.shape) == tuple(shape)]

                # If there is only one matching tensor, that's the activation we want.
                if len(matching) == 1:
                    chosen = matching[0]
                    self._used_ids.add(id(chosen))
                    return chosen

                # If there are multiple matching tensors, choose the non-leaf one.
                # TODO: Here we assume there is only one non-leaf tensor, which is not always the case for weight tying.
                # Need to test this with weight tying later.
                if matching:
                    for tensor in matching:
                        if not tensor.is_leaf:
                            self._used_ids.add(id(tensor))
                            return tensor

                    # If all tensors are leaf, we choose the first one.
                    # For example, the first layer input tensor is a leaf tensor.
                    chosen = matching[0]
                    self._used_ids.add(id(chosen))
                    return chosen

                return None

            chosen = _match_shape(input_shape)
            if chosen is not None:
                return chosen

            chosen = _match_shape(flat_shape)
            if chosen is not None:
                return chosen

            # If we reach here, we have failed to find a matching tensor.
            # This could happen for certain techniques, e.g., weight tying
            # where we need to strengthen the logic to handle this case.
            candidate_shapes = [tuple(t.shape) for t in non_param]
            raise RuntimeError(
                "Failed to resolve activation: no saved tensor matched "
                f"input_shape={input_shape} or flat_shape={flat_shape}. "
                f"layer={name} candidates={candidate_shapes}"
            )

    def clear_layer(self, name: str) -> None:
        with self._lock:
            self._captured.pop(name, None)


def add_hooks(
    model: nn.Module,
    val_batch_size: int,
    loss_reduction: str = 'mean',
    log_grad_norms: bool = False
):
    r"""
    Adds hooks to a model to compute gradient dot products and accumulate
    training gradients.

    The hooks will:
    1. Capture autograd-saved activations via saved_tensors_hooks for each layer.
    2. In the backward pass:
        a. Compute the gradient dot product between the validation batch
           gradient and each training sample's gradient.
        b. Compute and accumulate the averaged gradient for the
           training batch into `param.train_grad`.

    Args:
        model: The PyTorch model to which hooks are added.
        val_batch_size: The number of samples in the validation set.
        loss_reduction: The loss reduction type, 'mean' or 'sum'.
        Note: Train gradients are always averaged over the training portion of the batch.
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    handles = []
    manager = _NamedSavedTensorManager()
    manager._val_batch_size = val_batch_size
    manager._loss_reduction = loss_reduction
    model._ghost_saved_tensor_mgr = manager

    for name, layer in model.named_modules():
        if type(layer) in _supported_layers_dotprod and requires_grad(layer):

            layer.name = name
            layer._ghost_saved_tensor_mgr = manager
            manager._layers_by_name[name] = layer

            # push the layer name to the scope stack before forward pass
            def _push_scope(this_layer, inputs):
                manager.push(this_layer.name)
                if manager._get_enabled() and inputs and hasattr(inputs[0], "shape"):
                    input_shape = tuple(inputs[0].shape)
                    this_layer._ghost_input_shape = input_shape
                    total_bs = input_shape[0]
                    this_layer._ghost_total_bs = total_bs
                    this_layer._ghost_train_bs = total_bs - val_batch_size

            # pop the layer name from the scope stack after forward pass
            def _pop_scope(this_layer, inputs, output):
                manager.pop(this_layer.name)

            handles.append(layer.register_forward_pre_hook(_push_scope))
            handles.append(layer.register_forward_hook(_pop_scope))

            def _register_output_hook(this_layer, inputs, output):
                def _grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    _compute_dotprod_from_backprops(
                        this_layer, grad, val_batch_size, loss_reduction, log_grad_norms
                    )

                    if isinstance(this_layer, nn.Embedding):
                        masked_grad = _mask_embedding_grad_output(
                            this_layer, grad, val_batch_size, loss_reduction
                        )
                        _cleanup_layer_state(this_layer)
                        return masked_grad

                    if isinstance(this_layer, (nn.LayerNorm, nn.RMSNorm)):
                        # Keep activation for the full backward hook to fix grad_input.
                        this_layer._ghost_saved_activation = getattr(this_layer, "activations", None)
                        return grad

                    _cleanup_layer_state(this_layer)
                    return grad

                def _maybe_register_hook(out: torch.Tensor) -> None:
                    if out.requires_grad:
                        out.register_hook(_grad_hook)

                if isinstance(output, torch.Tensor):
                    _maybe_register_hook(output)
                elif isinstance(output, (tuple, list)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            _maybe_register_hook(out)

            handles.append(layer.register_forward_hook(_register_output_hook))

            if isinstance(layer, (nn.LayerNorm, nn.RMSNorm)):

                def norm_backward_hook(this_layer, grad_input, grad_output):
                    if not grad_output:
                        return None

                    activation = getattr(this_layer, "_ghost_saved_activation", None)
                    if activation is None:
                        activation = getattr(this_layer, "activations", None)
                    if activation is None:
                        manager = getattr(this_layer, "_ghost_saved_tensor_mgr", None)
                        if manager is not None:
                            activation = manager.resolve_activation(this_layer)

                    if activation is None:
                        return None

                    if isinstance(this_layer, nn.RMSNorm):
                        corrected = _compute_rmsnorm_grad_input(
                            this_layer, activation, grad_output[0]
                        )
                    else:
                        corrected = _compute_layernorm_grad_input(
                            this_layer, activation, grad_output[0]
                        )
                        backprops = grad_output[0].detach()
                        if loss_reduction == "mean":
                            backprops = backprops * float(backprops.shape[0])
                        _, compute_layer_train_grad = _supported_layers_dotprod.get(
                            type(this_layer), (None, None)
                        )

                        if compute_layer_train_grad is not None:
                            compute_layer_train_grad(
                                this_layer, activation, backprops, val_batch_size
                            )
                        else:
                            raise ValueError(
                                f"Layer {this_layer.__class__.__name__} is not supported for training gradient computation. "
                                "Ensure it is included in the _supported_layers_dotprod dictionary."
                            )

                    _cleanup_layer_state(this_layer)

                    if not grad_input:
                        return None
                    new_grad_input = list(grad_input)
                    new_grad_input[0] = corrected
                    return tuple(new_grad_input)

                handles.append(layer.register_full_backward_hook(norm_backward_hook))

        else:
            is_atomic_layer = not list(layer.children())
            if is_atomic_layer and requires_grad(layer):
                supported = ", ".join(cls.__name__ for cls in _supported_layers_dotprod)
                warnings.warn(
                    f"Skipping unsupported leaf layer '{name}' ({type(layer).__name__}). "
                    f"Only supported types: {supported}",
                    category=UserWarning,
                    stacklevel=2,
                )

    model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)


def remove_hooks(model: nn.Module):
    """Removes hooks added by `add_hooks()`."""
    if hasattr(model, "autograd_grad_sample_hooks"):
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks
    if hasattr(model, "_ghost_saved_tensor_mgr"):
        model._ghost_saved_tensor_mgr.disable()
        delattr(model, "_ghost_saved_tensor_mgr")
    for _, layer in model.named_modules():
        if hasattr(layer, "_ghost_saved_tensor_mgr"):
            delattr(layer, "_ghost_saved_tensor_mgr")
        if hasattr(layer, "_ghost_input_shape"):
            delattr(layer, "_ghost_input_shape")
        if hasattr(layer, "_ghost_train_bs"):
            delattr(layer, "_ghost_train_bs")
        if hasattr(layer, "_ghost_total_bs"):
            delattr(layer, "_ghost_total_bs")
        if hasattr(layer, "_ghost_saved_activation"):
            delattr(layer, "_ghost_saved_activation")
        if hasattr(layer, "activations"):
            delattr(layer, "activations")
        if hasattr(layer, "backprops"):
            delattr(layer, "backprops")


def _scale_logged_grad_norms(layer: nn.Module, grad_scale: float) -> None:
    """
    Rescales stored gradient norm stats to reflect the scaled backprops.
    """
    if grad_scale == 1.0:
        return

    scale_sq = grad_scale * grad_scale
    for param_name in ("weight", "bias"):
        if not hasattr(layer, param_name):
            continue
        param = getattr(layer, param_name)
        if param is None:
            continue
        if hasattr(param, "grad_train_norm") and param.grad_train_norm is not None:
            param.grad_train_norm = param.grad_train_norm * scale_sq
        if hasattr(param, "grad_val_norm_sq") and param.grad_val_norm_sq is not None:
            param.grad_val_norm_sq = param.grad_val_norm_sq * scale_sq


def _select_compute_dtype(layer: nn.Module, A: torch.Tensor, B: torch.Tensor) -> Optional[torch.dtype]:
    """
    Decide the compute dtype for dot-product calculations.

    - Keep embedding activations as integer indices; use backprop/weight dtype for compute.
    - Otherwise, prefer matching activation/backprop dtype or warn and fall back to activations.
    """
    if isinstance(layer, nn.Embedding):
        if B.is_floating_point():
            return B.dtype
        if hasattr(layer, "weight") and hasattr(layer.weight, "dtype"):
            return layer.weight.dtype
        return None

    if A.dtype != B.dtype:
        layer_name = getattr(layer, "name", layer.__class__.__name__)
        warnings.warn(
            "Mismatched dtypes for dot-product compute in "
            f"{layer_name}: A.dtype={A.dtype}, B.dtype={B.dtype}; using A.dtype.",
            UserWarning,
            stacklevel=2,
        )

    return A.dtype


def _reshape_activation_if_needed(layer: nn.Module, activation: torch.Tensor) -> torch.Tensor:
    input_shape = getattr(layer, "_ghost_input_shape", None)
    if input_shape is None or not hasattr(activation, "shape"):
        return activation
    flat_shape = None
    if len(input_shape) > 1:
        flat_shape = (int(math.prod(input_shape[:-1])), input_shape[-1])
    if flat_shape is not None and tuple(activation.shape) == tuple(flat_shape):
        return activation.reshape(input_shape)
    return activation


def _compute_dotprod_from_backprops(
    layer: nn.Module,
    backprops: torch.Tensor,
    val_batch_size: int,
    loss_reduction: str = "mean",
    log_grad_norms: bool = False,
) -> None:
    """
    Compute dot products from a single backprop tensor (grad_output).
    """
    backprops = backprops.detach()
    manager = getattr(layer, "_ghost_saved_tensor_mgr", None)

    if not hasattr(layer, "activations") or layer.activations is None:
        if manager is None:
            raise RuntimeError(
                f"Missing saved tensor manager for layer {getattr(layer, 'name', '<unnamed>')}."
            )

        activation = manager.resolve_activation(layer)
        if activation is None:
            raise RuntimeError(
                f"Failed to capture saved activations for layer {getattr(layer, 'name', '<unnamed>')}. "
                "Ensure the saved_tensors_hooks context is active around forward/backward."
            )

        activation = _reshape_activation_if_needed(layer, activation)
        layer.activations = activation

    compute_layer_dotprod, _ = _supported_layers_dotprod.get(type(layer))
    compute_dtype = _select_compute_dtype(layer, layer.activations, backprops)

    if manager is not None and manager._debug:
        print(
            "[compute_dotprod_from_backprops] "
            f"[{layer.name}] activations dtype: {layer.activations.dtype}, "
            f"backprops dtype: {backprops.dtype}, compute_dtype: {compute_dtype}, accum_dtype: {ACCUM_DTYPE}"
        )

    bench_start = None
    if _DOTPROD_BENCH_ENABLED:
        if _DOTPROD_BENCH_SYNC and backprops.is_cuda:
            torch.cuda.synchronize(backprops.device)
        bench_start = time.perf_counter()

    compute_layer_dotprod(
        layer,
        layer.activations,
        backprops,
        val_batch_size=val_batch_size,
        log_grad_norms=log_grad_norms,
        compute_dtype=compute_dtype,
        accum_dtype=ACCUM_DTYPE,
    )

    if bench_start is not None:
        if _DOTPROD_BENCH_SYNC and backprops.is_cuda:
            torch.cuda.synchronize(backprops.device)
        elapsed_s = time.perf_counter() - bench_start
        _update_dotprod_bench_stats(layer, backprops, elapsed_s, compute_dtype)


def _mask_embedding_grad_output(
    layer: nn.Module,
    grad_output: torch.Tensor,
    val_batch_size: int,
    loss_reduction: str,
) -> torch.Tensor:
    total_bs = getattr(layer, "_ghost_total_bs", grad_output.shape[0])
    train_bs = getattr(layer, "_ghost_train_bs", total_bs - val_batch_size)
    if train_bs <= 0 or train_bs >= total_bs:
        return grad_output

    if loss_reduction == "mean":
        scale = float(total_bs) / float(train_bs)
    else:
        scale = 1.0 / float(train_bs)

    masked = grad_output.clone()
    if scale != 1.0:
        masked[:train_bs] = masked[:train_bs] * scale
    masked[train_bs:] = 0
    return masked


def _cleanup_layer_state(layer: nn.Module) -> None:
    if hasattr(layer, "activations"):
        del layer.activations
    if hasattr(layer, "backprops"):
        del layer.backprops
    if hasattr(layer, "_ghost_saved_activation"):
        delattr(layer, "_ghost_saved_activation")
    manager = getattr(layer, "_ghost_saved_tensor_mgr", None)
    if manager is not None and hasattr(layer, "name"):
        manager.clear_layer(layer.name)


def _compute_rmsnorm_grad_input(
    layer: nn.RMSNorm, x: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_f = x.to(compute_dtype)
    go_f = grad_output.to(compute_dtype)

    weight = getattr(layer, "weight", None)
    if weight is not None:
        go_f = go_f * weight.to(compute_dtype)

    norm_dims = tuple(range(-len(layer.normalized_shape), 0))
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=norm_dims, keepdim=True) + layer.eps)
    x_hat = x_f * inv_rms
    go_xhat_mean = (go_f * x_hat).mean(dim=norm_dims, keepdim=True)
    grad_input = inv_rms * (go_f - x_hat * go_xhat_mean)
    return grad_input.to(grad_output.dtype)


def _compute_layernorm_grad_input(
    layer: nn.LayerNorm, x: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_f = x.to(compute_dtype)
    go_f = grad_output.to(compute_dtype)

    weight = getattr(layer, "weight", None)
    if weight is not None:
        go_f = go_f * weight.to(compute_dtype)

    norm_dims = tuple(range(-len(layer.normalized_shape), 0))
    mean = x_f.mean(dim=norm_dims, keepdim=True)
    var = x_f.var(dim=norm_dims, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + layer.eps)
    x_hat = (x_f - mean) * rstd

    n = 1
    for dim in layer.normalized_shape:
        n *= dim

    go_sum = go_f.sum(dim=norm_dims, keepdim=True)
    go_xhat_sum = (go_f * x_hat).sum(dim=norm_dims, keepdim=True)
    grad_input = (1.0 / n) * rstd * (n * go_f - go_sum - x_hat * go_xhat_sum)
    return grad_input.to(grad_output.dtype)


def _apply_train_grad(
    layer: nn.Module,
    val_batch_size: int,
    loss_reduction: str = 'mean'
):
    """
    Computes and applies the training gradient for a given layer's parameters.
    This function acts as a dispatcher based on the layer type.
    """
    _, compute_layer_train_grad = _supported_layers_dotprod.get(type(layer), (None, None))

    if not compute_layer_train_grad:
        raise ValueError(
            f"Layer {layer.__class__.__name__} is not supported for training gradient computation. "
            "Ensure it is included in the _supported_layers_dotprod dictionary."
        )

    # LayerNorm's function is self-contained and handles both weight and bias.
    if isinstance(layer, nn.LayerNorm):
        compute_layer_train_grad(
            layer, layer.activations, layer.backprops, val_batch_size
        )
    else:

        # For other layers (Linear, Embedding), handle weight and bias separately.
        # --- Handle Weight ---
        if hasattr(layer, 'weight') and layer.weight.initially_requires_grad:
            grad_weight = compute_layer_train_grad(
                layer,
                layer.activations,
                layer.backprops,
                val_batch_size
            )

            # This check is now robust because only functions that return tensors will reach here.
            if grad_weight is not None:
                _create_or_accumulate_train_grad(layer.weight, grad_weight)
            else:
                raise ValueError(
                    f"Layer {layer.__class__.__name__} returned None for weight gradient. "
                    "Ensure the compute_layer_train_grad function is implemented correctly."
                )

        # --- Handle Bias ---
        if hasattr(layer, 'bias') and layer.bias is not None and layer.bias.initially_requires_grad:
            grad_bias = _compute_train_grad_bias(
                layer.backprops,
                val_batch_size,
                loss_reduction=loss_reduction
            )
            _create_or_accumulate_train_grad(layer.bias, grad_bias)

    # Cleanup is performed for all supported layers after processing.
    if hasattr(layer, 'activations'):
        del layer.activations
    if hasattr(layer, 'backprops'):
        del layer.backprops
    if hasattr(layer, '_ghost_input_shape'):
        delattr(layer, '_ghost_input_shape')
    manager = getattr(layer, "_ghost_saved_tensor_mgr", None)
    if manager is not None and hasattr(layer, "name"):
        manager.clear_layer(layer.name)


def _compute_train_grad_bias(
    B: torch.Tensor,
    val_batch_size: int,
    loss_reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes the sum or average of gradients across the training data for a bias term.
    """
    train_batch_size = B.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute gradients, check batch sizes.")

    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Sum over the batch dimension (0) and all sequence/spatial dimensions
    # (from 1 to n-1), leaving only the last (feature) dimension.
    sum_dims = list(range(B_train.dim() - 1))
    summed_grad_bias = B_train.sum(dim=sum_dims)
    # The result will have shape [features], which matches the bias parameter.

    if loss_reduction == 'mean':
        summed_grad_bias /= train_batch_size

    return summed_grad_bias
