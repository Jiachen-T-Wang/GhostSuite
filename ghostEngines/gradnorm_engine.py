import torch
from torch.optim import Optimizer
import torch.nn as nn
from . import autograd_grad_sample
from typing import List, Union
import warnings

from .supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping
from . import autograd_grad_sample, transformers_support



class GradNormEngine:
    """
    GradNormEngine enables the computation of per-sample gradient norms during
    standard model training. It hooks into a model to compute per-sample gradients
    and calculates their norms without altering the gradients used for optimization.
    This is useful for analysis and monitoring of the training process.

    It is designed to be a drop-in component. You attach it to your model and
    optimizer, and then call `engine.step()` instead of `optimizer.step()`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        batch_size: int = 1,
        *args, **kwargs
    ):
        """
        Initializes the GradNormEngine.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (Optimizer): The optimizer for the training.
            batch_size (int): The batch size used in training. This is only used as a
                              fallback if the batch size cannot be inferred from gradients.
        """
        self.model = model
        self.optimizer = optimizer
        self._batch_size = batch_size

        self._original_step = optimizer.step
        optimizer.step = self.step
        
        self.grad_norm_history = []
        self._hooks_enabled = True

        # ----- ghost differentiation trick (TODO) through origin parameter -----
        for name, param in self.model.named_parameters():
            # print(f"Initializing parameter: {name} with requires_grad={param.requires_grad}")
            param.initially_requires_grad=bool(param.requires_grad)

        def _supported_and_trainable(layer):            
            if type(layer) in _supported_layers_norm_sample_AND_clipping and ((hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad) or (hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad)):
                return True
            return False

        # store layer's name and create list of named layers for blockwise clipping
        self.named_layers=[]
        for name,layer in self.model.named_modules():
            if _supported_and_trainable(layer):
                self.named_layers.append((name, layer))

        self.n_layers=len(self.named_layers) 
        
        self.n_components=0
        for name, layer in self.named_layers:
            self.n_components+=sum([1 for p in layer.parameters() if p.initially_requires_grad])
        print("Number of trainable components: ", self.n_components, "; Number of trainable layers: ", self.n_layers)

        self.block_heads = [] 

        # Fix the position embeddings broadcast issue.
        transformers_support.forward_swapper(module=self.model)

        # This is the primary mechanism that enables per-sample gradient computation
        autograd_grad_sample.add_hooks(self.model, block_heads=self.block_heads)



    def _get_per_sample_grad_norm(self, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 norm of per-sample gradients for a given parameter.

        Args:
            p (torch.Tensor): The parameter tensor.

        Returns:
            torch.Tensor: A tensor containing the L2 norm of the gradient for each sample.
        """
        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per-sample gradients not found. Make sure you are calling loss.backward() before engine.step()."
            )
        
        # The .grad_sample attribute is added by the hooks from autograd_grad_sample
        grad_sample = p.grad_sample

        return grad_sample.view(len(grad_sample), -1).norm(2, dim=-1)

    def _compute_total_per_sample_grad_norm(self) -> torch.Tensor:
        """
        Computes the total L2 norm of per-sample gradients across all parameters for each sample.
        The norm is calculated as sqrt(sum_of_squared_norms_of_each_layer).

        Returns:
            torch.Tensor: A 1-D tensor of size (batch_size,) containing the total
                          gradient norm for each sample in the batch.
        """
        total_norm_sq = None
        batch_size_inferred = None

        for p in self.model.parameters():
            if p.grad is not None and hasattr(p, "grad_sample"):
                if batch_size_inferred is None:
                    batch_size_inferred = len(p.grad_sample)

                per_sample_norm_sq = self._get_per_sample_grad_norm(p) ** 2
                
                if total_norm_sq is None:
                    total_norm_sq = per_sample_norm_sq
                else:
                    # Ensure dimensions match for broadcasting if necessary, though they should be the same.
                    if total_norm_sq.shape != per_sample_norm_sq.shape:
                         warnings.warn(f"Shape mismatch in grad norm accumulation. "
                                       f"Total norm shape: {total_norm_sq.shape}, "
                                       f"Param norm shape: {per_sample_norm_sq.shape}")
                         # Attempt to align if one is a subset of the other
                         min_len = min(len(total_norm_sq), len(per_sample_norm_sq))
                         total_norm_sq = total_norm_sq[:min_len] + per_sample_norm_sq[:min_len]
                    else:
                        total_norm_sq += per_sample_norm_sq

        if total_norm_sq is None:
            # Use the fallback batch_size if no grad_samples were found
            device = self.optimizer.param_groups[0]["params"][0].device
            return torch.zeros(self._batch_size, device=device)

        return torch.sqrt(total_norm_sq)

    def _clear_grad_samples(self):
        """
        Removes the .grad_sample attribute from all model parameters to free up memory.
        """
        for p in self.model.parameters():
            if hasattr(p, "grad_sample"):
                del p.grad_sample

    def step(self):
        """
        Performs a single optimization step.
        This method computes and stores the per-sample gradient norms,
        then calls the original optimizer's step function to update the weights.
        """

        if not self._hooks_enabled:

            # If hooks are disabled, raise warning and skip gradient norm computation
            warnings.warn(
                "Hooks are disabled. No per-sample gradient norms will be computed. "
            )

            # Just perform a regular optimizer step
            self._original_step()
            return
            
        # 1. Compute and store per-sample gradient norms
        grad_norms = self._compute_total_per_sample_grad_norm()
        self.grad_norm_history.append(grad_norms.detach().cpu())

        # 2. Perform the original optimizer step with the original (non-private) gradients
        self._original_step()

        # 3. Clean up the per-sample gradients to save memory before the next batch
        self._clear_grad_samples()

    def get_grad_norm_history(self) -> List[torch.Tensor]:
        """
        Returns the history of per-sample gradient norms computed at each step.

        Returns:
            List[torch.Tensor]: A list of tensors. Each tensor corresponds to one
                                training step and contains the per-sample gradient
                                norms for the batch processed in that step.
        """
        return self.grad_norm_history

    def __enter__(self):
        # This allows using the engine with a 'with' statement to temporarily disable hooks
        self._original_hooks_status = self._hooks_enabled
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the hooks status
        if self._original_hooks_status:
            self.enable_hooks()
        else:
            self.disable_hooks()

    def disable_hooks(self):
        """
        Disables the hooks used for computing per-sample gradients by calling
        the library's `remove_hooks` function.
        When hooks are disabled, `engine.step()` behaves exactly like `optimizer.step()`,
        and no gradient norms are computed. This is useful for validation loops.
        """
        autograd_grad_sample.remove_hooks(self.model)
        self._hooks_enabled = False

    def enable_hooks(self):
        """
        Enables the hooks for computing per-sample gradients by calling
        the library's `add_hooks` function.
        """
        autograd_grad_sample.add_hooks(self.model, block_heads=self.block_heads)
        self._hooks_enabled = True




