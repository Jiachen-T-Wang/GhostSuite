import logging
import math
import types
from typing import Dict, Optional, Sequence, Union
import os
import warnings

import torch
from torch import nn

from . import autograd_grad_sample_dotprod
from . import transformers_support
from .supported_layers_grad_samplers_dotprod import _supported_layers_dotprod



class GradDotProdEngine:
    """
    An engine to compute gradient dot products between a validation set and
    training samples, and to update the model using the training gradients.
    """
    def __init__(
        self,
        module: nn.Module,
        val_batch_size: int,
        loss_reduction: str = 'mean',
        use_dummy_bias: bool = False,
        dot_prod_save_path: Optional[str] = None,
    ):
        """
        Initializes the GradDotProdEngine.

        Args:
            module: The PyTorch module to which the engine will be attached.
            val_batch_size: The number of samples in the fixed validation batch.
            loss_reduction: The reduction used for the loss function ('mean' or 'sum').
                          This is needed to correctly scale the backpropagated gradients.
            use_dummy_bias: If True, the bias parameters are set to not require gradients.
            dot_prod_save_path: The directory where the dot product log will be saved.
        """
        super().__init__()

        self.module = module
        self.val_batch_size = val_batch_size
        self.loss_reduction = loss_reduction
        self.dot_prod_save_path = dot_prod_save_path

        self.named_params = list(
            (name, param) for (name, param) in module.named_parameters() if param.requires_grad
        )

        # Internal state to prevent race conditions during the optimizer step
        self._grad_creation_locked = False

        # --- Initialize a list to log dot products on the GPU ---
        self.dot_product_log = []

        # A list to log batch indices corresponding to the dot products
        self.batch_idx_lst = []

        # Improving efficiency through dummy bias trick (if enabled)
        # Explanation: Since we compute the weight gradient by ourself
        # in the standard backward pass, we can set all requires_grad to False and improve efficiency.
        for name, param in module.named_parameters():

            # Store the original requires_grad status
            param.initially_requires_grad = bool(param.requires_grad)

            # If we use dummy_bias trick, set all requires_grad to False
            if use_dummy_bias:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Fix for Hugging Face model incompatibility
        transformers_support.forward_swapper(module=module)

    def _lock_grad_creation(self):
        """
        Prevents the creation of new gradients until the optimizer step is complete.
        """
        self._grad_creation_locked = True

    def _unlock_grad_creation(self):
        """
        Allows the creation of new gradients for the next training step.
        """
        self._grad_creation_locked = False

    def attach(self, optimizer: torch.optim.Optimizer):
        """ Attach the engine to an optimizer and register autograd hooks. """

        self.optimizer = optimizer

        autograd_grad_sample_dotprod.add_hooks(
            model=self.module,
            val_batch_size=self.val_batch_size,
            loss_reduction=self.loss_reduction
        )

        # Keep a reference to the engine on the optimizer for convenience
        optimizer.grad_dot_prod_engine = self

    def detach(self):
        """
        Detaches the engine from the optimizer, restoring its original state and
        cleaning up hooks and custom attributes.
        """
        optimizer = self.optimizer

        if hasattr(optimizer, "grad_dot_prod_engine"):
            del optimizer.grad_dot_prod_engine

        # Remove the hooks from the model
        autograd_grad_sample_dotprod.remove_hooks(self.module)
        self.module.zero_grad()

        # Clean up custom attributes from all parameters
        for param in self.module.parameters():
            if hasattr(param, 'train_grad'):
                del param.train_grad
            if hasattr(param, 'grad_dot_prod'):
                del param.grad_dot_prod
            # Clean up temporary attributes left by hooks
            if hasattr(param, 'activations'):
                del param.activations
            if hasattr(param, 'backprops'):
                del param.backprops


    def _prepare_and_apply_train_grad(self):
        """
        Moves the accumulated training gradients from `param.train_grad` to
        `param.grad` so the optimizer can use them for the update.
        """
        if self._grad_creation_locked:
            # This is a safeguard, though the new step logic doesn't require it as strictly.
            return

        for name, param in self.module.named_parameters():

            if not param.initially_requires_grad:
                continue

            # if "dummy_bias" included in the name, skip it
            if "dummy_bias" in name:
                continue
            
            if hasattr(param, 'train_grad'):
                # Ensure the train_grad attribute exists
                if param.train_grad is None:
                    raise ValueError(
                        f"Parameter {name} requires grad has no accumulated training gradient. "
                    )
                else:
                    param.grad = param.train_grad
            else:
                raise ValueError(
                    f"Parameter {name} requires grad but does not have 'train_grad' attribute. "
                )
                
        # Lock to prevent accidental re-creation of gradients before step is done.
        self._lock_grad_creation()


    def _clear_train_grad(self):
        """
        Deletes the `param.train_grad` attribute from parameters after the
        optimizer step is complete. Also unlocks gradient creation.
        """
        for param in self.module.parameters():
            if hasattr(param, 'train_grad'):
                del param.train_grad
        
        # Unlock to allow the next backward pass to create new gradients.
        self._unlock_grad_creation()


    def prepare_gradients(self):
        """Move accumulated training gradients to ``.grad`` for optimizer."""
        self._prepare_and_apply_train_grad()

    def clear_gradients(self):
        """Remove stored training gradients after the optimizer step."""
        self._clear_train_grad()

    def aggregate_and_log(self):
        """Aggregate per-layer dot products and append to the log list."""
        self._aggregate_and_log_dot_products()


    def _aggregate_and_log_dot_products(self):
        """
        Calculates the total dot product for the current iteration by summing
        across all layers, and logs the result to a list on the GPU.
        """
        total_dot_product_iter = None

        for name, param in self.module.named_parameters():

            if hasattr(param, 'grad_dot_prod') and param.initially_requires_grad:

                # Check if tensor is not empty
                if param.grad_dot_prod.numel() > 0:
                    if total_dot_product_iter is None:
                        # Initialize with the first dot product tensor found
                        total_dot_product_iter = param.grad_dot_prod
                    else:
                        # Add subsequent dot product tensors element-wise
                        total_dot_product_iter += param.grad_dot_prod
                
                # Clean up the per-parameter attribute immediately to save memory
                delattr(param, 'grad_dot_prod')            

        if total_dot_product_iter is not None:

            # Summarize batch info (batch data + grad dot products)
            info_this_iter = {
                'dot_product': to_device(total_dot_product_iter, 'cpu'),
                'X_train': to_device(self.X_train, 'cpu'),
                'Y_train': to_device(self.Y_train, 'cpu'),
                'iter_num': self.iter_num,
                'batch_idx': self.batch_idx
            }

            print(f"Dot product info for this iteration: {total_dot_product_iter}")

            self.dot_product_log.append(info_this_iter)

        else:
            # If no dot products were computed, log a warning
            warnings.warn("No gradient dot products computed for this iteration.")


    def save_dot_product_log(self, iter_num: int):
        """
        Moves the GPU log of dot products to the CPU, saves it to a file,
        and clears the log.

        Args:
            save_path: The directory where the file will be saved.
            iter_num: The current training iteration number, used for the filename.
        """

        if not self.dot_product_log:
            raise ValueError("No gradient dot products found to save for this iteration.")

        print(f"[INFO] Saving dot product log at iteration {iter_num} ...")

        file_path = os.path.join(self.dot_prod_save_path, f"dot_prod_log_iter_{iter_num}.pt")
        torch.save(self.dot_product_log, file_path)

        # Clear the log to start fresh
        self.dot_product_log.clear()


    def attach_and_store_valset(self, X_val, Y_val):
        """
        Attaches a fixed validation batch to the engine for dot product calculations.
        """

        valset_path = os.path.join(self.dot_prod_save_path, "valset.pt")
        torch.save({'X_val': to_device(X_val, 'cpu'), 'Y_val': to_device(Y_val, 'cpu')}, valset_path)
        print(f"[INFO] Saved validation set to {valset_path}")

    def attach_train_batch(self, X_train, Y_train, iter_num, batch_idx=None):
        """
        Attaches the current training batch for reference. 
        """

        self.X_train = X_train
        self.Y_train = Y_train
        self.iter_num = iter_num
        self.batch_idx = batch_idx


# TODO: move this to a utility file
def to_device(data, device):
    """Move data to device, handling both tensors and dicts of tensors."""
    if isinstance(data, dict):
        return {k: v.to(device) if hasattr(v, 'to') else v for k, v in data.items()}
    elif hasattr(data, 'to'):
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type for moving to GPU: {type(data)}. Expected tensor or dict of tensors.")
