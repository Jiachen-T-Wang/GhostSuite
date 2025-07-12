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
        *,
        val_batch_size: int,
        loss_reduction: str = 'mean',
        average_grad: bool = True,
        origin_params: Optional[Sequence[str]] = None,
        **unused_kwargs,
    ):
        """
        Initializes the GradDotProdEngine.

        Args:
            module: The PyTorch module to which the engine will be attached.
            val_batch_size: The number of samples in the fixed validation batch.
            loss_reduction: The reduction used for the loss function ('mean' or 'sum').
                          This is needed to correctly scale the backpropagated gradients.
            average_grad: If True, the model is updated with the average of the
                          training gradients. If False (default), the sum is used.
            origin_params: A list of parameter names to be used for the ghost
                           differentiation trick. Gradients will only be computed
                           starting from these parameters.
        """
        del unused_kwargs
        super().__init__()

        self.module = module
        self.val_batch_size = val_batch_size
        self.loss_reduction = loss_reduction
        self.average_grad = average_grad
        self.named_params = list(
            (name, param) for (name, param) in module.named_parameters() if param.requires_grad
        )

        # Internal state to prevent race conditions during the optimizer step
        self._grad_creation_locked = False

        # --- Initialize a list to log dot products on the GPU ---
        self.dot_product_log = []

        #----- Ghost differentiation trick through origin parameters -----
        # This part is kept to enable efficient gradient computation.
        for name, param in module.named_parameters():

            # Store the original requires_grad status
            param.initially_requires_grad = bool(param.requires_grad)

            # Only set requires_grad to True for the very first layer (e.g., embedding layer)
            param.requires_grad = False
            if name == "transformer.wte.weight":
                param.requires_grad = True
            if name == "transformer.wpe.weight":
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
        """
        Attaches the GradDotProdEngine to a PyTorch optimizer, overriding its
        step() method to use the custom training gradients.
        """
        self.optimizer = optimizer

        autograd_grad_sample_dotprod.add_hooks(
            model=self.module,
            val_batch_size=self.val_batch_size,
            loss_reduction=self.loss_reduction
        )

        def dot_prod_step(self, closure=None):
            """
            Performs a single optimization step using the gradients stored in
            `param.train_grad`.
            """
            # 1. Zero out any existing gradients on the parameters.
            self.zero_grad()

            # 2. Prepare the training gradients for the optimizer step.
            self.grad_dot_prod_engine._prepare_and_apply_train_grad()

            # TODO: add gradient clipping here if needed

            # --- Aggregate and log dot products for this step ---
            self.grad_dot_prod_engine._aggregate_and_log_dot_products()

            # 3. Call the original optimizer's step function to update weights.
            self.original_step(closure=closure)
            
            # 4. Clean up the used gradients to prepare for the next iteration.
            self.grad_dot_prod_engine._clear_train_grad()

        # Store a reference to the engine on the optimizer for access in the step
        optimizer.grad_dot_prod_engine = self

        # Save the original step method and override it with our custom one
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dot_prod_step, optimizer)

    def detach(self):
        """
        Detaches the engine from the optimizer, restoring its original state and
        cleaning up hooks and custom attributes.
        """
        optimizer = self.optimizer
        optimizer.step = optimizer.original_step

        # Remove the references we added
        del optimizer.original_step
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


    def _aggregate_and_log_dot_products(self):
        """
        Calculates the total dot product for the current iteration by summing
        across all layers, and logs the result to a list on the GPU.
        """
        total_dot_product_iter = None

        for param in self.module.parameters():
            if hasattr(param, 'grad_dot_prod'):
                if param.grad_dot_prod.numel() > 0: # Check if tensor is not empty
                    if total_dot_product_iter is None:
                        # Initialize with the first dot product tensor found
                        total_dot_product_iter = param.grad_dot_prod
                    else:
                        # Add subsequent dot product tensors element-wise
                        total_dot_product_iter += param.grad_dot_prod
                
                # Clean up the per-parameter attribute immediately to save memory
                delattr(param, 'grad_dot_prod')
        
        if total_dot_product_iter is not None:
            # Append the final aggregated tensor for this iteration to the log
            self.dot_product_log.append(total_dot_product_iter)
        else:
            # If no dot products were computed, log a warning
            warnings.warn("No gradient dot products computed for this iteration.")


    def save_dot_product_log(self, save_path: str, iter_num: int):
        """
        Moves the GPU log of dot products to the CPU, saves it to a file,
        and clears the log.

        Args:
            save_path: The directory where the file will be saved.
            iter_num: The current training iteration number, used for the filename.
        """
        if not self.dot_product_log:
            raise ValueError("No gradient dot products found to save for this iteration.")

        print(f"Saving dot product log at iteration {iter_num} ...")
        
        # Move all logged tensors to the CPU at once
        log_cpu = [t.cpu() for t in self.dot_product_log]
        
        # Stack into a single tensor: [num_logged_iterations, batch_size]
        full_log_tensor = torch.stack(log_cpu)
        
        file_path = os.path.join(save_path, f"dot_prod_log_iter_{iter_num}.pt")
        torch.save(full_log_tensor, file_path)
        print(f"Saved dot product log of shape {full_log_tensor.shape} to {file_path}")

        # Clear the log to start fresh
        self.dot_product_log.clear()


