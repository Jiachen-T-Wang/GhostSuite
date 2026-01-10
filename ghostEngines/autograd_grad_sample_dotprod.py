from typing import Dict, List, Optional, Tuple
import math
import os
import threading
import warnings

import torch
import torch.nn as nn

from .supported_layers_grad_samplers_dotprod import (
    _supported_layers_dotprod,
    _create_or_accumulate_train_grad
)

ACCUM_DTYPE = torch.float32


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
        self._captured_all: List[torch.Tensor] = []

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
            self._captured_all = []
            self._used_ids = set()
        self._get_stack().clear()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._captured = {}
            self._captured_all = []
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
            self._captured_all.append(x)
            if stack:
                name = stack[-1]
                self._captured.setdefault(name, []).append(x)
            if self._debug:
                scope = stack[-1] if stack else "<none>"
                print(
                    "[ghost_saved_tensor] "
                    f"tid={threading.get_ident()} scope={scope} "
                    f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                    f"captured_all={len(self._captured_all)}"
                )
        return x

    def unpack_hook(self, x: torch.Tensor) -> torch.Tensor:
        return x

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

            capture_pool = self._captured.get(name, []) or self._captured_all
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
    model._ghost_saved_tensor_mgr = manager

    for name, layer in model.named_modules():
        if type(layer) in _supported_layers_dotprod and requires_grad(layer):

            layer.name = name
            layer._ghost_saved_tensor_mgr = manager

            # push the layer name to the scope stack before forward pass
            def _push_scope(this_layer, inputs):
                manager.push(this_layer.name)
                if manager._get_enabled() and inputs and hasattr(inputs[0], "shape"):
                    this_layer._ghost_input_shape = tuple(inputs[0].shape)

            # pop the layer name from the scope stack after forward pass
            def _pop_scope(this_layer, inputs, output):
                manager.pop(this_layer.name)

            handles.append(layer.register_forward_pre_hook(_push_scope))
            handles.append(layer.register_forward_hook(_pop_scope))

            def backward_hook(this_layer, grad_input, grad_output):

                # compute the gradient dot products and store them on the layer
                _prepare_sample_grad_or_dotprod(
                    this_layer, grad_output, val_batch_size, loss_reduction, log_grad_norms
                )

                # compute and accumulate the training gradients
                _apply_train_grad(this_layer, val_batch_size)

                return None

            handles.append(layer.register_full_backward_hook(backward_hook))

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
    - Otherwise, prefer a promoted dtype between weight and backprop when both exist.
    """
    if isinstance(layer, nn.Embedding):
        if B.is_floating_point():
            return B.dtype
        if hasattr(layer, "weight") and hasattr(layer.weight, "dtype"):
            return layer.weight.dtype
        return None

    bp_dtype = B.dtype if B.is_floating_point() else None
    weight_dtype = None
    if hasattr(layer, "weight") and getattr(layer, "weight", None) is not None:
        weight_dtype = layer.weight.dtype

    if bp_dtype is not None and weight_dtype is not None:
        return torch.promote_types(bp_dtype, weight_dtype)
    return bp_dtype or weight_dtype


def _prepare_sample_grad_or_dotprod(
    layer: nn.Module,
    grad_output: Tuple[torch.Tensor],
    val_batch_size: int,
    loss_reduction: str = 'mean',
    log_grad_norms: bool = False,
):
    """
    Backward hook handler that captures backprops and computes the gradient dot product.
    """
    backprops = grad_output[0].detach()
    grad_scale = float(backprops.shape[0]) if loss_reduction == 'mean' else 1.0
    manager = getattr(layer, "_ghost_saved_tensor_mgr", None)

    if not hasattr(layer, 'activations') or layer.activations is None:
        if manager is None:
            raise RuntimeError(
                f"Missing saved tensor manager for layer {getattr(layer, 'name', '<unnamed>')}."
            )

        # resolve the activation tensor from the saved tensors during forward pass
        activation = manager.resolve_activation(layer)
        if activation is None:
            raise RuntimeError(
                f"Failed to capture saved activations for layer {getattr(layer, 'name', '<unnamed>')}. "
                "Ensure the saved_tensors_hooks context is active around forward/backward."
            )

        # for linear layers, we need to reshape the activation tensor (batch_size * seq_len, d_model)
        # back to (batch_size, seq_len, d_model) for the dot product computation.
        input_shape = getattr(layer, "_ghost_input_shape", None)
        if input_shape is not None and hasattr(activation, "shape"):
            flat_shape = None
            if len(input_shape) > 1:
                flat_shape = (int(math.prod(input_shape[:-1])), input_shape[-1])
            if flat_shape is not None and tuple(activation.shape) == tuple(flat_shape):
                activation = activation.reshape(input_shape)

        layer.activations = activation

    # The function to compute the dot product is retrieved from the support dictionary.
    # We assume the second function returned is for computing the training gradient.
    compute_layer_dotprod, _ = _supported_layers_dotprod.get(type(layer))

    compute_dtype = _select_compute_dtype(layer, layer.activations, backprops)

    if manager is not None and manager._debug:
        print(
            "[prepare_sample_grad_or_dotprod] "
            f"[{layer.name}] activations dtype: {layer.activations.dtype}, "
            f"backprops dtype: {backprops.dtype}, compute_dtype: {compute_dtype}, accum_dtype: {ACCUM_DTYPE}"
        )

    compute_layer_dotprod(
        layer,
        layer.activations,
        backprops,
        val_batch_size=val_batch_size,
        log_grad_norms=log_grad_norms,
        compute_dtype=compute_dtype,
        accum_dtype=ACCUM_DTYPE,
    )

    if grad_scale != 1.0:
        if log_grad_norms:
            _scale_logged_grad_norms(layer, grad_scale)
        # Scale the backprops since the value is being divided by train_batch_size+val_batch_size.
        backprops = backprops * grad_scale

    # Store (scaled) backprops for the next function in the hook.
    layer.backprops = backprops


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
