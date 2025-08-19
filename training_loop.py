"""Main training loop implementation."""

import time
import numpy as np
import torch
import os

# Local imports
from training_utils import (
    get_learning_rate,
    update_learning_rate,
    estimate_loss,
    save_training_results, 
    to_device
)

# Ghost Engines
from ghostEngines import GhostEngineManager


class Trainer: 
    """Main trainer class that orchestrates the training process."""
    
    def __init__(self, model, optimizer, scaler, config, ddp_info, 
                 get_batch_fn, get_val_batch_fn, ctx):
        
        print("[INFO] Initializing Trainer ...")

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.ddp_info = ddp_info
        self.get_batch = get_batch_fn
        self.get_val_batch = get_val_batch_fn
        self.ctx = ctx
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9

        # Prepare validation data for ghost engines (if needed)
        val_data = None
        if self.config.method == 'GradDotProd':
            X_val, Y_val = self.get_val_batch(
                self.config.val_batch_size, return_idx=False
            )
            X_val = to_device(X_val, self.ddp_info['device'])
            Y_val = to_device(Y_val, self.ddp_info['device'])
            val_data = (X_val, Y_val)

        # Initialize ghost engine manager
        self.ghost_engine = GhostEngineManager(
            config=self.config,
            model=self.model, 
            optimizer=self.optimizer,
            ddp_info=self.ddp_info,
            val_data=val_data
        )


    def run_training(self):

        print("[INFO] Starting training...")
        
        result_file = self.config.get_result_file_path()
        
        try:
            
            while self.iter_num < self.config.max_steps:

                # Evaluation every eval_interval steps
                if self.iter_num % self.config.eval_interval == 0:
                    self._run_evaluation(result_file)

                if self.config.args.eval_only:
                    print('Eval only mode, exiting now')
                    break
                
                # Perform complete training step
                self._training_step(self.iter_num)
                                
                self.iter_num += 1
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup(result_file)
    

    def _training_step(self, iter_num):
        """Perform one complete training step including data loading, LR update, and ghost engine operations."""
        
        # Get training batch
        X, Y, batch_idx = self.get_batch(
            'train', 
            batch_size=self.config.batch_size, 
            return_idx=True
        )

        # Store batch info for ghost engine
        self.ghost_engine.attach_train_batch(X, Y, iter_num, batch_idx)
        
        # Update learning rate
        lr = get_learning_rate(iter_num, self.config) if self.config.decay_lr else self.config.learning_rate
        update_learning_rate(self.optimizer, lr)
        
        # Save ghost engine metrics at their own interval (before training step)
        if iter_num > 0 and self.ddp_info['master_process']:
            if self.ghost_engine.should_save_metrics(iter_num):
                self.ghost_engine.save_metrics(iter_num)
        
        loss = None

        # Forward and backward pass with gradient accumulation
        for micro_step in range(self.config.gradient_accumulation_steps):
            if self.ddp_info['ddp']:
                self.model.require_backward_grad_sync = (
                    micro_step == self.config.gradient_accumulation_steps - 1
                )
            
            with self.ctx:
                # Prepare input based on the ghost engine method
                X_forward, Y_forward = self.ghost_engine.prepare_forward_input(X, Y)
                
                # Forward pass with method-appropriate input
                outputs = self.model(input_ids=X_forward, labels=Y_forward)
                logits, loss = outputs.logits, outputs.loss
                
                # Scale loss for gradient accumulation
                if loss is not None:
                    loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if loss is not None:
                self.scaler.scale(loss).backward()
        
        # Prepare gradients using ghost engine
        self.ghost_engine.prepare_gradients()

        print(f"Step {iter_num}, Loss: {loss.item() if loss is not None else 'N/A'}, LR: {lr:.6f}")

        # Gradient clipping and optimization step
        self.scaler.unscale_(self.optimizer)
        
        # if self.config.grad_clip != 0.0:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # This will call the custom engine's step() if it's enabled, which computes values
        # before calling the original optimizer step.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Aggregate metrics and clear gradients using ghost engine
        self.ghost_engine.aggregate_and_log()
        self.ghost_engine.clear_gradients()

        self.optimizer.zero_grad(set_to_none=True)
    

    def _run_evaluation(self, result_file):

        # Detach ghost engines during evaluation
        self.ghost_engine.detach_for_evaluation()

        losses = estimate_loss(self.model, self.get_batch, self.config, self.ctx)

        # Reattach ghost engines after evaluation
        self.ghost_engine.reattach_after_evaluation()

        train_loss, val_loss, test_loss = losses['train'], losses['val'], losses['test']
        
        print(f"step {self.iter_num}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, test loss {test_loss:.4f}")
        
        save_training_results(result_file, train_loss, val_loss, test_loss, self.iter_num)


    def _cleanup(self, result_file):
        """Cleanup training resources and run a final evaluation."""

        print("Running cleanup ...")

        # Run evaluation at the end of training
        self._run_evaluation(result_file)

        # Cleanup ghost engines
        self.ghost_engine.cleanup()
