"""Training utilities and helper functions."""

import os
import time
import math
import json
import pickle
import threading
import queue
from contextlib import nullcontext

import torch
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Setup distributed training if available."""
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
        device = 'cuda'
    
    return {
        'ddp': ddp,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'device': device,
        'master_process': master_process,
        'seed_offset': seed_offset
    }


def setup_torch_backend(config):
    """Setup PyTorch backend settings."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config.dtype]
    
    # Create autocast context
    ctx = nullcontext()  # Disabled for gradient calculation accuracy
    
    return device_type, ptdtype, ctx


def get_learning_rate(iteration, config):
    """Get learning rate for current iteration using cosine schedule with warmup."""
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    
    if iteration > config.lr_decay_iters:
        return config.min_lr
    
    # Cosine decay
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def update_learning_rate(optimizer, lr):
    """Update learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def estimate_loss(model, get_batch_fn, config, ctx):
    """Estimate loss on train/val/test splits."""
    model.eval()
    
    out = {}
    for split in ['train', 'val', 'test']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch_fn(split, batch_size=config.eval_bs)
            
            with ctx:
                outputs = model(input_ids=X, labels=Y)
                logits, loss = outputs.logits, outputs.loss

            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def save_training_results(file_path, train_loss, val_loss, test_loss, step):
    """Save training results to JSON file."""
    # Read existing record
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as file:
            record = json.load(file)
    else:
        record = []
    
    # Add new entry
    new_entry = {
        "train_loss": train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss,
        "eval_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
        "test_loss": test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss,
        "step": step
    }
    
    record.append(new_entry)
    
    # Write back to file
    with open(file_path, "w") as file:
        json.dump(record, file, indent=4)


def shapley_value_processor(q, value_record, lock):
    """Process Shapley values asynchronously."""
    while True:
        try:
            item = q.get(timeout=1.0)
            
            if item is None:  # Sentinel value to stop
                print("Shapley processor received shutdown signal")
                q.task_done()
                break
            
            first_order_score_gpu, batch_idx, lr = item
            
            try:
                first_order_value = first_order_score_gpu.cpu().numpy()
            except Exception as e:
                print(f"Error converting tensor to numpy: {e}")
                q.task_done()
                continue
            
            # Store values with thread safety
            with lock:
                value_record['index'].append(batch_idx)
                value_record['First-order In-Run Data Shapley'].append(first_order_value)
            
            q.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in Shapley value processor: {e}")
            try:
                q.task_done()
            except ValueError:
                pass


class ShapleyProcessor:
    """Handles asynchronous Shapley value processing."""
    
    def __init__(self, queue_size=10):
        self.value_record = {
            'index': [],
            'First-order In-Run Data Shapley': []
        }
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.thread = None
    
    def start(self):
        """Start the processing thread."""
        self.thread = threading.Thread(
            target=shapley_value_processor,
            args=(self.data_queue, self.value_record, self.lock)
        )
        self.thread.start()
    
    def add_values(self, first_order_score, batch_idx, lr, timeout=5.0):
        """Add values to processing queue."""
        data_to_queue = (
            first_order_score.detach(),
            batch_idx,
            lr
        )
        try:
            self.data_queue.put(data_to_queue, timeout=timeout)
        except queue.Full:
            print("Warning: Shapley processing queue is full, skipping this batch")
    
    def shutdown(self, timeout=10.0):
        """Shutdown the processing thread."""
        print("Shutting down Shapley value processing...")
        
        try:
            self.data_queue.put(None, timeout=2.0)
        except queue.Full:
            print("Warning: Could not send shutdown signal to queue")
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print("Warning: Worker thread did not shut down gracefully")
            else:
                print("Shapley value processing finished.")
    
    def save_values(self, file_path):
        """Save collected values to file."""
        try:
            with self.lock:
                pickle.dump(self.value_record, open(file_path + '.value', 'wb'))
                print(f"Shapley values saved to {file_path}.value")
        except Exception as e:
            print(f"Error saving Shapley values: {e}")


def cleanup_distributed():
    """Cleanup distributed training."""
    destroy_process_group()


def print_training_info(config, tokens_per_iter):
    """Print training configuration information."""
    print(f"Tokens Per Iteration: {tokens_per_iter:,}")
    print(f"Architecture: {config.args.architecture}")
    print(f"Method: {config.method}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max steps: {config.max_steps}")
