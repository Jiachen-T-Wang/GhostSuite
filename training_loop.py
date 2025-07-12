"""Main training loop implementation."""

import time
import numpy as np
import torch
import os

from training_utils import (
    get_learning_rate,
    update_learning_rate,
    estimate_loss,
    save_training_results
)

from ghostEngines.graddotprod_engine import GradDotProdEngine
# from ghostEngines.gradnorm_engine import GradNormEngine


class Trainer: 
    """Main trainer class that orchestrates the training process."""
    
    def __init__(self, model, optimizer, scaler, config, ddp_info, 
                 get_batch_fn, get_val_batch_fn, ctx, trainable_layers=None):
        
        print("Initializing Trainer ...")

        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.ddp_info = ddp_info
        self.get_batch = get_batch_fn
        self.get_val_batch = get_val_batch_fn
        self.ctx = ctx
        self.trainable_layers = trainable_layers
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9
        
        # Timing
        self.time_list = []

        # Initialize engines based on the selected method
        self.grad_norm_engine = None
        self.grad_dot_prod_engine = None

        if self.config.method == 'GradDotProd':
            print("Initializing GradDotProdEngine ...")

            self.grad_dot_prod_engine = GradDotProdEngine(
                module=self.model,
                val_batch_size=self.config.val_batch_size,
                loss_reduction='mean',
                average_grad=True,
                origin_params=None,
            )

            self.grad_dot_prod_engine.attach(self.optimizer)
            
            self.dot_prod_save_path = os.path.join(self.config.result_folder, "grad_dot_products")
            if self.ddp_info['master_process']:
                os.makedirs(self.dot_prod_save_path, exist_ok=True)

            print("GradDotProdEngine initialized successfully.")
        
        # Get validation data once
        self.X_val, self.Y_val = self.get_val_batch(
            config.val_batch_size, 
            config.args.val_set
        )
        # Move validation data to the correct device
        self.X_val = self.X_val.to(ddp_info['device'])
        self.Y_val = self.Y_val.to(ddp_info['device'])

    def run_training(self):
        """Run the main training loop."""
        print("Starting training...")
        
        result_file = self.config.get_result_file_path()
        
        try:
            t0_total = time.time()
            
            while self.iter_num < self.config.max_steps:
                t_iter_start = time.time()
                
                # Get training batch
                X, Y, batch_idx = self.get_batch(
                    'train', 
                    batch_size=self.config.batch_size, 
                    return_idx=True
                )
                
                # Update learning rate
                lr = get_learning_rate(self.iter_num, self.config) if self.config.decay_lr else self.config.learning_rate
                update_learning_rate(self.optimizer, lr)
                
                if self.iter_num > 0 and self.ddp_info['master_process']:
                    # Run evaluation at its own interval
                    if self.iter_num % self.config.eval_interval == 0:
                        self._run_evaluation(result_file)
                    
                    # Save dot products at their own interval
                    if (self.config.method == 'GradDotProd' and self.iter_num % self.config.dot_prod_save_interval == 0):
                        self._save_dot_products()
                
                if self.config.args.eval_only:
                    print('Eval only mode, exiting now')
                    break
                
                # Training step
                self._training_step(X, Y, batch_idx, lr)
                
                # Timing
                torch.cuda.synchronize()
                t_iter_end = time.time()
                dt = t_iter_end - t_iter_start
                self.time_list.append(dt)
                
                print(f"iter {self.iter_num}: time {dt*1000:.2f}ms")
                
                self.iter_num += 1
            
            # Training completed
            t1_total = time.time()
            dt_total = t1_total - t0_total
            avg_time = np.mean(self.time_list) * 1000 if self.time_list else 0
            print(f"Training completed in {dt_total:.2f}s, avg iter time: {avg_time:.2f}ms")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup(result_file)
    
    def _training_step(self, X, Y, batch_idx, lr):
        """Perform one training step."""
        
        loss = None

        # # Debug: print the first few entries of each layer's parameters
        # if self.ddp_info['master_process']:
        #     print("[Debug] Inspecting model parameters before training step:")
        #     for name, param in self.model.named_parameters():
        #         if param.initially_requires_grad:
        #             first_vals = param.view(-1)[:5].detach().cpu()
        #             print(f"[Debug] {name} first entries: {first_vals}")

        # Forward and backward pass with gradient accumulation
        for micro_step in range(self.config.gradient_accumulation_steps):
            if self.ddp_info['ddp']:
                self.model.require_backward_grad_sync = (
                    micro_step == self.config.gradient_accumulation_steps - 1
                )
            
            with self.ctx:
                if self.config.method == 'Regular':
                    start_time = time.time()
                    outputs = self.model(input_ids=X, labels=Y)
                    logits, loss = outputs.logits, outputs.loss
                    torch.cuda.synchronize()
                    print(f"Forward pass time: {time.time() - start_time:.4f}s")

                elif self.config.method == 'GradNorm':
                    start_time = time.time()
                    outputs = self.model(input_ids=X, labels=Y)
                    logits, loss = outputs.logits, outputs.loss
                    torch.cuda.synchronize()
                    print(f"Forward pass time: {time.time() - start_time:.4f}s")

                elif self.config.method == 'GradDotProd':
                    # Concatenate train and val batches
                    X_cat = torch.cat((X, self.X_val), dim=0)
                    Y_cat = torch.cat((Y, self.Y_val), dim=0)

                    start_time = time.time()
                    # The loss here will be from the concatenated batch
                    outputs = self.model(input_ids=X_cat, labels=Y_cat)
                    logits, loss = outputs.logits, outputs.loss
                    torch.cuda.synchronize()
                    print(f"Forward pass time: {time.time() - start_time:.4f}s")
                
                # Scale loss for gradient accumulation
                if loss is not None:
                    loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if loss is not None:
                start_time = time.time()
                self.scaler.scale(loss).backward()
                torch.cuda.synchronize()
                print(f"Backward pass time: {time.time() - start_time:.4f}s")
        
        # Gradient clipping and optimization step
        self.scaler.unscale_(self.optimizer)
        
        # if self.config.grad_clip != 0.0:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # This will call the custom engine's step() if it's enabled, which computes values
        # before calling the original optimizer step.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Print loss value for debugging
        if loss is not None:
            print(f"Loss at iter {self.iter_num}: {loss.item():.4f}")

        self.optimizer.zero_grad(set_to_none=True)
    

    def _run_evaluation(self, result_file):

        if self.grad_dot_prod_engine: self.grad_dot_prod_engine.detach()
        if self.grad_norm_engine: self.grad_norm_engine.disable_hooks()

        losses = estimate_loss(self.model, self.get_batch, self.config, self.ctx)

        if self.grad_dot_prod_engine: self.grad_dot_prod_engine.attach(self.optimizer)
        if self.grad_norm_engine: self.grad_norm_engine.enable_hooks()

        train_loss, val_loss, test_loss = losses['train'], losses['val'], losses['test']
        
        print(f"step {self.iter_num}: train loss {train_loss:.4f}, "
              f"val loss {val_loss:.4f}, test loss {test_loss:.4f}")
        
        save_training_results(result_file, train_loss, val_loss, test_loss, self.iter_num)


    def _save_dot_products(self):
        """
        Triggers the engine to save its entire log of dot products to a file.
        """
        if self.grad_dot_prod_engine is None:
            raise RuntimeError("GradDotProdEngine is not initialized.")

        # The engine handles the file path and logging internally
        self.grad_dot_prod_engine.save_dot_product_log(
            save_path=self.dot_prod_save_path,
            iter_num=self.iter_num
        )

