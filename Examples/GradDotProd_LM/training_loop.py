"""Main training loop implementation."""

import time
import torch
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
from shared.training_utils import (
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
        self.wandb_run = None
        
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
            self.dynamic_val_batch = getattr(self.config, "dynamic_val_batch", False)
        else:
            self.dynamic_val_batch = False

        # Initialize ghost engine manager
        self.ghost_engine = GhostEngineManager(
            config=self.config,
            model=self.model, 
            optimizer=self.optimizer,
            ddp_info=self.ddp_info,
            val_data=val_data
        )

        # Initialize Weights & Biases logging if requested
        self._init_wandb()


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
                try:
                    self._training_step(self.iter_num)
                except StopIteration:
                    print("[INFO] Replay data exhausted; terminating training loop.")
                    break
                                
                self.iter_num += 1
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup(result_file)
    

    def _training_step(self, iter_num):
        """
        Perform one complete training step including data loading, LR update, and ghost engine operations.
        """

        start_time = time.time()

        # Refresh validation batch if configured (for GradDotProd)
        if self.dynamic_val_batch:
            self._refresh_validation_batch()
        
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
                outputs = self.model(X_forward, Y_forward)
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
        
        if self.config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # This will call the custom engine's step() if it's enabled, which computes values
        # before calling the original optimizer step.
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Aggregate metrics and clear gradients using ghost engine
        self.ghost_engine.aggregate_and_log()
        self.ghost_engine.clear_gradients()

        self.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Time taken for training step: {end_time - start_time:.4f} seconds")
        metrics = {
            "train/lr": lr,
            "train/step_time": end_time - start_time
        }
        if loss is not None:
            metrics["train/loss"] = loss.item()
        self._log_metrics(metrics, step=iter_num)
    

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
        self._log_metrics({
            "eval/train_loss": float(train_loss),
            "eval/val_loss": float(val_loss),
            "eval/test_loss": float(test_loss)
        }, step=self.iter_num)


    def _cleanup(self, result_file):
        """Cleanup training resources and run a final evaluation."""

        print("Running cleanup ...")

        # Run evaluation at the end of training
        self._run_evaluation(result_file)

        # Cleanup ghost engines
        self.ghost_engine.cleanup()
        if self.wandb_run is not None and self.ddp_info['master_process']:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"[WARN] Failed to finalize Weights & Biases run: {e}")


    def _init_wandb(self):
        """Initialize Weights & Biases logging on the master process."""
        if not getattr(self.config, "use_wandb", False):
            return
        if not self.ddp_info['master_process']:
            return
        try:
            import wandb
        except ImportError:
            print("[WARN] Weights & Biases is not installed; skipping wandb logging.")
            return

        run_name = self.config.wandb_run_name
        if not run_name:
            timestamp = int(time.time())
            run_name = f"{self.config.method}_{self.config.architecture}_bs{self.config.batch_size}_lr{self.config.learning_rate}_{timestamp}"

        config_payload = {
            "method": self.config.method,
            "architecture": self.config.architecture,
            "train_set": self.config.args.train_set,
            "val_set": self.config.args.val_set,
            "batch_size": self.config.batch_size,
            "val_batch_size": self.config.val_batch_size,
            "learning_rate": self.config.learning_rate,
            "optimizer": self.config.optimizer,
            "max_steps": self.config.max_steps,
            "seed": self.config.seed,
            "eval_interval": self.config.eval_interval,
            "eval_iters": self.config.eval_iters,
            "eval_bs": self.config.eval_bs,
            "dot_prod_save_interval": self.config.dot_prod_save_interval,
            "model_dtype": self.config.model_dtype,
            "train_dtype": self.config.train_dtype,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps
        }

        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                mode=self.config.wandb_mode,
                dir=self.config.wandb_dir,
                config=config_payload
            )
            print(f"[INFO] Weights & Biases logging enabled (project: {self.config.wandb_project}, run: {run_name}).")
        except Exception as e:
            print(f"[WARN] Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None


    def _log_metrics(self, metrics, step=None):
        """Log metrics to Weights & Biases if enabled."""
        if self.wandb_run is None or not self.ddp_info['master_process']:
            return
        try:
            self.wandb_run.log(metrics, step=step if step is not None else self.iter_num)
        except Exception as e:
            print(f"[WARN] Failed to log metrics to Weights & Biases: {e}")


    def _refresh_validation_batch(self):
        """Fetch and attach a new validation batch for GradDotProd."""
        X_val, Y_val = self.get_val_batch(
            self.config.val_batch_size, return_idx=False
        )
        X_val = to_device(X_val, self.ddp_info['device'])
        Y_val = to_device(Y_val, self.ddp_info['device'])
        self.ghost_engine.update_validation_batch(X_val, Y_val)
