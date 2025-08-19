"""
Ghost Engine Manager - Unified interface for managing gradient computation engines.

This module provides a clean interface for initializing and managing ghost engines
based on configuration, removing the need for method-specific code in training loops.
"""

import os
from typing import Optional, Union, Dict, Any

import torch
from torch import nn

from .graddotprod_engine import GradDotProdEngine


class GhostEngineManager:
    """
    A unified manager for ghost engines that provides a clean interface
    for training loops without method-specific initialization code.
    """
    
    def __init__(self, config, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 ddp_info: Dict[str, Any], val_data=None):
        """
        Initialize the ghost engine manager based on configuration.
        
        Args:
            config: Training configuration object with method and other settings
            model: The PyTorch model to attach engines to
            optimizer: The optimizer to integrate with
            ddp_info: Distributed training information
            val_data: Tuple of (X_val, Y_val) validation data (required for GradDotProd)
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.ddp_info = ddp_info
        
        # Initialize engine based on method
        self.engine = None
        if val_data is not None:
            self.X_val, self.Y_val = val_data
        else:
            self.X_val, self.Y_val = None, None
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the appropriate engine based on config.method."""
        if self.config.method == 'GradDotProd':
            self._initialize_graddotprod_engine()
        elif self.config.method == 'Regular':
            # No engine needed for regular training
            print("[INFO] Regular training mode - no ghost engine required.")
        else:
            print(f"[WARNING] Unknown method '{self.config.method}' - no ghost engine initialized.")
    
    def _initialize_graddotprod_engine(self):
        """Initialize GradDotProdEngine with all required setup."""
        print("[INFO] Initializing GradDotProdEngine ...")
        
        # Prepare directory for saving dot products
        dot_prod_save_path = os.path.join(self.config.result_dir, "grad_dotprods")
        if self.ddp_info['master_process']:
            os.makedirs(dot_prod_save_path, exist_ok=True)
        
        # Initialize the engine
        self.engine = GradDotProdEngine(
            module=self.model,
            val_batch_size=self.config.val_batch_size,
            loss_reduction='mean',
            average_grad=True,
            use_dummy_bias=True,
            dot_prod_save_path=dot_prod_save_path
        )
        
        # Attach to optimizer
        self.engine.attach(self.optimizer)
        
        # Validate that validation data is provided
        if self.X_val is None or self.Y_val is None:
            raise ValueError("X_val and Y_val are required for GradDotProd method")
        
        # Attach validation data to the engine
        self.engine.attach_and_store_valset(self.X_val, self.Y_val)
        
        print("[INFO] GradDotProdEngine initialized successfully.")
    

    def is_active(self) -> bool:
        """Check if any ghost engine is active."""
        return self.engine is not None
    
    def get_method(self) -> str:
        """Get the current method name."""
        return self.config.method
    
    def attach_train_batch(self, X_train, Y_train, iter_num, batch_idx=None):
        """Attach training batch information to the engine (if applicable)."""
        if self.engine and hasattr(self.engine, 'attach_train_batch'):
            self.engine.attach_train_batch(X_train, Y_train, iter_num, batch_idx)
    
    def prepare_gradients(self):
        """Prepare gradients after backward pass (if applicable)."""
        if self.engine and hasattr(self.engine, 'prepare_gradients'):
            self.engine.prepare_gradients()
    
    def aggregate_and_log(self):
        """Aggregate and log metrics after optimizer step (if applicable)."""
        if self.engine and hasattr(self.engine, 'aggregate_and_log'):
            self.engine.aggregate_and_log()
    
    def clear_gradients(self):
        """Clear gradients after optimizer step (if applicable)."""
        if self.engine and hasattr(self.engine, 'clear_gradients'):
            self.engine.clear_gradients()
    
    def should_save_metrics(self, iter_num: int) -> bool:
        """Check if metrics should be saved at this iteration."""
        if self.config.method == 'GradDotProd' and iter_num > 0:
            return iter_num % self.config.dot_prod_save_interval == 0
        # Add other engine-specific save intervals here
        return False
    
    def save_metrics(self, iter_num: int):
        """Save metrics to disk (if applicable)."""
        if self.config.method == 'GradDotProd' and self.engine:
            self.engine.save_dot_product_log(iter_num=iter_num)
    
    def get_validation_data(self):
        """Get validation data for methods that need it."""
        return self.X_val, self.Y_val
    
    def prepare_forward_input(self, X_train, Y_train):
        """
        Prepare forward pass input by concatenating with validation data if needed.
        
        Returns:
            Tuple of (X, Y) ready for forward pass
        """
        if self.config.method == 'GradDotProd' and self.X_val is not None:
            # Concatenate train and val batches for GradDotProd method
            if isinstance(self.X_val, dict):
                # Handle dictionary inputs (e.g., LLaVA model)
                X_cat = {}
                X_cat["input_ids"] = torch.cat((X_train["input_ids"], self.X_val["input_ids"]), dim=0)
                X_cat["pixel_values"] = torch.cat((X_train["pixel_values"], self.X_val["pixel_values"]), dim=0)
                X_cat["attention_mask"] = torch.cat((X_train["attention_mask"], self.X_val["attention_mask"]), dim=0)
                Y_cat = torch.cat((Y_train, self.Y_val), dim=0)
                return X_cat, Y_cat
            else:
                # Handle tensor inputs
                X_cat = torch.cat((X_train, self.X_val), dim=0)
                Y_cat = torch.cat((Y_train, self.Y_val), dim=0)
                return X_cat, Y_cat
        else:
            # Regular training - return original inputs
            return X_train, Y_train
    
    def detach_for_evaluation(self):
        """Detach engines during evaluation to avoid interference."""
        if self.engine:
            if hasattr(self.engine, 'detach'):
                self.engine.detach()
            elif hasattr(self.engine, 'disable_hooks'):
                self.engine.disable_hooks()
    
    def reattach_after_evaluation(self):
        """Reattach engines after evaluation."""
        if self.engine:
            if hasattr(self.engine, 'attach'):
                self.engine.attach(self.optimizer)
            elif hasattr(self.engine, 'enable_hooks'):
                self.engine.enable_hooks()
    
    def cleanup(self):
        """Cleanup and save any remaining data during training termination."""
        if not self.engine:
            return
            
        # Save any remaining metrics
        if (self.config.method == 'GradDotProd' and 
            hasattr(self.engine, 'dot_product_log') and 
            self.engine.dot_product_log):
            try:
                # Save with a special cleanup iteration number
                self.engine.save_dot_product_log(iter_num=-1)
            except Exception as e:
                print(f"Error saving remaining dot products during cleanup: {e}")
        
        # Detach the engine
        try:
            if hasattr(self.engine, 'detach'):
                self.engine.detach()
            elif hasattr(self.engine, 'disable_hooks'):
                self.engine.disable_hooks()
        except Exception as e:
            print(f"Error detaching ghost engine during cleanup: {e}")
