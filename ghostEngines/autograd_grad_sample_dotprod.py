from typing import Tuple

import torch
import torch.nn as nn
import transformers

import time

# Assuming these are defined in the dot-product specific samplers file
from .supported_layers_grad_samplers_dotprod import (
    _supported_layers_dotprod,
    _create_or_accumulate_train_grad
)


def requires_grad(module: nn.Module) -> bool:
    """
    Checks if any parameters in a specified module require gradients.
    """
    return any(p.initially_requires_grad for p in module.parameters())


def add_hooks(
    model: nn.Module,
    val_batch_size: int,
    loss_reduction: str = 'mean'
):
    r"""
    Adds hooks to a model to compute gradient dot products and accumulate
    training gradients.

    The hooks will:
    1. Save activations into ``layer.activations`` during the forward pass.
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

    for name, layer in model.named_modules():
        if type(layer) in _supported_layers_dotprod and requires_grad(layer):

            handles.append(layer.register_forward_hook(_capture_activations))

            layer.name = name

            def backward_hook(this_layer, grad_input, grad_output):

                # start_time = time.time()
                # 1. Compute the gradient dot products and store them on the layer.
                _prepare_sample_grad_or_dotprod(
                    this_layer, grad_output, val_batch_size, loss_reduction
                )

                # 2. Compute and accumulate the training gradients.
                _apply_train_grad(this_layer, val_batch_size)

                # A backward hook must return None or a new grad_input tuple.
                # Since we are not modifying the gradient flow, we return None.
                return None

            handles.append(layer.register_full_backward_hook(backward_hook))
            # print(f"[Debug] Added hooks to layer: {name} of type {type(layer)}")

        else:
            is_atomic_layer = not list(layer.children())
            if is_atomic_layer and requires_grad(layer):
                print(f"WARNING: Skipping atomic layer '{name}' of type {type(layer)} because it is not supported.")

    model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)


def remove_hooks(model: nn.Module):
    """Removes hooks added by `add_hooks()`."""
    if hasattr(model, "autograd_grad_sample_hooks"):
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks


def _capture_activations(layer: nn.Module, inputs: Tuple, outputs: Tuple):
    """Forward hook handler captures and saves activations."""
    layer.activations = inputs[0].detach()


def _prepare_sample_grad_or_dotprod(
    layer: nn.Module,
    grad_output: Tuple[torch.Tensor],
    val_batch_size: int,
    loss_reduction: str = 'mean',
):
    """
    Backward hook handler that captures backprops and computes the gradient dot product.
    """
    backprops = grad_output[0].detach()

    if not hasattr(layer, 'activations'):
        layer.activations = None

    # The function to compute the dot product is retrieved from the support dictionary.
    # We assume the second function returned is for computing the training gradient.
    compute_layer_dotprod, _ = _supported_layers_dotprod.get(type(layer))

    # This logic correctly handles mixed precision.
    if layer.activations is not None and layer.activations.dtype != backprops.dtype:
        common_type = torch.promote_types(layer.activations.dtype, backprops.dtype)
        compute_layer_dotprod(
            layer,
            layer.activations.to(common_type),
            backprops.to(common_type),
            val_batch_size=val_batch_size
        )
    else:
        compute_layer_dotprod(
            layer,
            layer.activations,
            backprops,
            val_batch_size=val_batch_size
        )

    if loss_reduction == 'mean':
        # Scale the backprops since the value is being divided by train_batch_size+val_batch_size.
        backprops = backprops * backprops.shape[0]

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
