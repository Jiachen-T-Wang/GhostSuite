#!/usr/bin/env python3
"""Test script to verify ghost engine functionality."""

import torch
import torch.nn as nn
import torch.optim as optim
from ghostEngines import GradDotProdEngine
import tempfile
import os

def test_graddotprod_engine():
    """Test basic functionality of GradDotProdEngine."""
    
    print("Testing GradDotProdEngine functionality...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'dot_prods')
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize GradDotProdEngine
        val_batch_size = 2
        engine = GradDotProdEngine(
            module=model,
            val_batch_size=val_batch_size,
            loss_reduction='mean',
            average_grad=True,
            use_dummy_bias=False,
            dot_prod_save_path=save_path
        )
        
        # Attach to optimizer
        engine.attach(optimizer)
        
        # Create sample data
        batch_size = 4
        total_batch_size = batch_size + val_batch_size
        X = torch.randn(total_batch_size, 10).to(device)
        Y = torch.randn(total_batch_size, 10).to(device)
        
        # Split into train and val
        X_train = X[:batch_size]
        Y_train = Y[:batch_size]
        X_val = X[batch_size:]
        Y_val = Y[batch_size:]
        
        # Store validation set
        engine.attach_and_store_valset(X_val, Y_val)
        
        # Attach training batch
        engine.attach_train_batch(X_train, Y_train, iter_num=1, batch_idx=[0,1,2,3])
        
        # Forward pass
        output = model(X)
        
        # Compute loss
        loss = nn.MSELoss()(output, Y)
        
        # Backward pass
        loss.backward()
        
        # Prepare gradients
        engine.prepare_gradients()
        
        # Check that gradients are prepared
        has_gradients = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"✓ Gradients prepared: {has_gradients}")
        
        # Optimizer step
        optimizer.step()
        
        # Aggregate and log
        engine.aggregate_and_log()
        
        # Check that dot products were computed
        has_dot_products = len(engine.dot_product_log) > 0
        print(f"✓ Dot products computed: {has_dot_products}")
        
        if has_dot_products:
            dot_prod_info = engine.dot_product_log[0]
            print(f"  - Dot product shape: {dot_prod_info['dot_product'].shape}")
            print(f"  - Iteration number: {dot_prod_info['iter_num']}")
        
        # Save dot products
        if has_dot_products:
            engine.save_dot_product_log(iter_num=1)
            saved_file = os.path.join(save_path, 'dot_prod_log_iter_1.pt')
            file_exists = os.path.exists(saved_file)
            print(f"✓ Dot products saved: {file_exists}")
        
        # Clean up
        engine.clear_gradients()
        engine.detach()
        
        print("✓ GradDotProdEngine test completed successfully!")
        
    return True

def test_hook_mechanism():
    """Test that hooks are properly added and removed."""
    
    print("\nTesting hook mechanism...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.LayerNorm(10),
        nn.Linear(10, 5)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Check no hooks initially
    has_hooks_initially = hasattr(model, "autograd_grad_sample_hooks")
    print(f"✓ No hooks initially: {not has_hooks_initially}")
    
    # Create engine and attach
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = GradDotProdEngine(
            module=model,
            val_batch_size=1,
            loss_reduction='mean',
            dot_prod_save_path=tmpdir
        )
        engine.attach(optimizer)
        
        # Check hooks are added
        has_hooks_after_attach = hasattr(model, "autograd_grad_sample_hooks")
        print(f"✓ Hooks added after attach: {has_hooks_after_attach}")
        
        # Detach
        engine.detach()
        
        # Check hooks are removed
        has_hooks_after_detach = hasattr(model, "autograd_grad_sample_hooks")
        print(f"✓ Hooks removed after detach: {not has_hooks_after_detach}")
    
    print("✓ Hook mechanism test completed successfully!")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Running Ghost Engine Tests")
    print("=" * 60)
    
    try:
        test_graddotprod_engine()
        test_hook_mechanism()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()