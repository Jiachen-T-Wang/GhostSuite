"""
Example demonstrating gradient projection on a simple MLP.
Includes validation modes for non-interference and naive equality checks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine
from ghostEngines.gradProjection.projection_utils import choose_ki_ko, init_projection_matrix_gaussian


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def compute_naive_projection(model, inputs, targets, engine_config):
    """
    Compute projected gradients using naive method (materialize then project).
    This is for validation only - not efficient for large models.
    """
    model.zero_grad()
    
    batch_size = inputs.shape[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Get projection configuration from engine
    engine = GradProjLoraEngine(model, **engine_config)
    
    all_projections = []
    
    for b in range(batch_size):
        model.zero_grad()
        
        # Forward pass for single sample
        output = model(inputs[b:b+1])
        loss = criterion(output, targets[b:b+1]).mean()
        
        # Backward to get gradients
        loss.backward()
        
        # Collect and project gradients for each layer
        sample_projections = []
        
        for layer_name in sorted(engine.matched_layers.keys()):
            layer = engine.matched_layers[layer_name]
            P_i, P_o = engine.projection_matrices[layer_name]
            
            # Get full gradient
            if hasattr(layer, 'weight'):
                grad_full = layer.weight.grad.clone()  # [n_o, n_i]
                
                # Project using Kronecker structure
                # vec(P_o @ grad @ P_i.T) = (P_i ⊗ P_o) @ vec(grad)
                grad_proj = P_o @ grad_full @ P_i.t()  # [k_o, k_i]
                grad_flat = grad_proj.reshape(-1)
                
                sample_projections.append(grad_flat)
                
        # Concatenate all layers
        sample_proj = torch.cat(sample_projections)
        all_projections.append(sample_proj)
        
    # Stack all samples
    naive_projections = torch.stack(all_projections).to(torch.float32)
    
    # Clean up
    engine.detach()
    
    return naive_projections


def test_non_interference(args):
    """Test that engine doesn't interfere with normal training."""
    print("\n=== Testing Non-Interference ===")
    
    # Set deterministic mode
    torch.manual_seed(args.seed)
    # Only use deterministic algorithms on CPU (CUDA requires special env vars)
    device = torch.device('cpu')  # Force CPU for deterministic tests
    if not torch.cuda.is_available() or device.type == 'cpu':
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    # Create model and data
    model1 = SimpleMLP().to(device)
    model2 = SimpleMLP().to(device)
    
    # Copy weights to ensure identical initialization
    model2.load_state_dict(model1.state_dict())
    
    # Generate data
    torch.manual_seed(args.seed)
    inputs = torch.randn(args.batch_size, 10, device=device)
    targets = torch.randint(0, 5, (args.batch_size,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train without engine
    print("\nTraining without engine...")
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    losses1 = []
    
    for step in range(args.test_steps):
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer1.step()
        losses1.append(loss.item())
        
    # Train with engine
    print("Training with engine...")
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    
    # Create engine
    engine_config = {
        'proj_layers': 'fc',
        'proj_rank_total': args.proj_rank_total,
        'proj_rank_min': args.proj_rank_min,
        'proj_seed': args.proj_seed,
        'proj_dtype': args.proj_dtype,
        'proj_dir': args.proj_dir,
    }
    engine = GradProjLoraEngine(model2, **engine_config)
    engine.attach()
    
    losses2 = []
    
    for step in range(args.test_steps):
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Collect projections (but don't use them)
        _ = engine.collect_batch()
        
        optimizer2.step()
        losses2.append(loss.item())
        
    engine.detach()
    
    # Compare results
    print("\nComparing training trajectories:")
    all_equal = True
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        equal = np.allclose(l1, l2, rtol=1e-5, atol=1e-6)
        print(f"  Step {i}: loss_no_engine={l1:.6f}, loss_with_engine={l2:.6f}, equal={equal}")
        if not equal:
            all_equal = False
            
    # Compare final parameters
    print("\nComparing final parameters:")
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        equal = torch.allclose(p1, p2, rtol=1e-5, atol=1e-6)
        print(f"  {n1}: equal={equal}")
        if not equal:
            all_equal = False
            
    if all_equal:
        print("\n✓ Non-interference test PASSED")
    else:
        print("\n✗ Non-interference test FAILED")
        
    return all_equal


def test_naive_equality(args):
    """Test that engine projections match naive computation."""
    print("\n=== Testing Naive Equality ===")
    
    torch.manual_seed(args.seed)
    device = torch.device('cpu')  # Use CPU for deterministic tests
    
    # Create model and data
    model = SimpleMLP().to(device)
    
    # Small batch for naive test
    batch_size = min(4, args.batch_size)
    inputs = torch.randn(batch_size, 10, device=device)
    targets = torch.randint(0, 5, (batch_size,), device=device)
    
    # Engine configuration
    engine_config = {
        'proj_layers': 'fc',
        'proj_rank_total': args.proj_rank_total,
        'proj_rank_min': args.proj_rank_min,
        'proj_seed': args.proj_seed,
        'proj_dtype': args.proj_dtype,
        'proj_dir': args.proj_dir,
    }
    
    # Compute with engine
    print("\nComputing with engine...")
    engine = GradProjLoraEngine(model, **engine_config)
    engine.attach()
    
    model.zero_grad()
    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    loss.backward()
    
    engine_projections = engine.collect_batch()
    engine.detach()
    
    # Compute naive projections
    print("Computing naive projections...")
    naive_projections = compute_naive_projection(model, inputs, targets, engine_config)
    
    # Compare
    print("\nComparing projections:")
    print(f"  Engine shape: {engine_projections.shape}")
    print(f"  Naive shape: {naive_projections.shape}")
    
    # Convert to same dtype for comparison
    engine_proj_float = engine_projections.to(torch.float32)
    
    # Check equality
    equal = torch.allclose(engine_proj_float, naive_projections, rtol=1e-4, atol=1e-5)
    
    if equal:
        print("\n✓ Naive equality test PASSED")
    else:
        print("\n✗ Naive equality test FAILED")
        max_diff = (engine_proj_float - naive_projections).abs().max().item()
        print(f"  Max difference: {max_diff}")
        
    return equal


def run_projection(args):
    """Run normal projection mode."""
    print("\n=== Running Gradient Projection ===")
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SimpleMLP().to(device)
    
    # Create engine
    engine_config = {
        'proj_layers': args.proj_layers,
        'proj_rank_total': args.proj_rank_total,
        'proj_rank_min': args.proj_rank_min,
        'proj_seed': args.proj_seed,
        'proj_dtype': args.proj_dtype,
        'proj_dir': args.proj_dir,
        'proj_save_interval': args.proj_save_interval,
    }
    
    engine = GradProjLoraEngine(model, **engine_config)
    engine.attach()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for step in range(args.num_steps):
        # Generate random batch
        inputs = torch.randn(args.batch_size, 10, device=device)
        targets = torch.randint(0, 5, (args.batch_size,), device=device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Collect projections
        projections = engine.collect_batch(batch_indices=list(range(step * args.batch_size, 
                                                                   (step + 1) * args.batch_size)))
        
        # Update model
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, proj_shape={projections.shape}")
            
    engine.detach()
    print(f"\nProjections saved to: {args.proj_dir}")


def main():
    parser = argparse.ArgumentParser(description='MLP Gradient Projection Example')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='project',
                       choices=['project', 'non_interf', 'naive_check'],
                       help='Execution mode')
    
    # Projection configuration (required)
    parser.add_argument('--proj_layers', type=str, default='fc',
                       help='Comma-separated layer name patterns')
    parser.add_argument('--proj_rank_total', type=int, default=64,
                       help='Target total projection dimension per layer')
    parser.add_argument('--proj_rank_min', type=int, default=4,
                       help='Minimum dimension for k_i and k_o')
    parser.add_argument('--proj_seed', type=int, default=42,
                       help='Random seed for projection matrices')
    parser.add_argument('--proj_dtype', type=str, default='float32',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='Data type for storing projections')
    parser.add_argument('--proj_dir', type=str, default='./Examples/grad_proj_mlp',
                       help='Directory to save projections')
    parser.add_argument('--proj_save_interval', type=int, default=1,
                       help='Save projections every N iterations')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of training steps')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')
    parser.add_argument('--test_steps', type=int, default=10,
                       help='Number of steps for testing')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.proj_layers:
        raise ValueError("--proj_layers is required")
    if args.proj_rank_total <= 0:
        raise ValueError("--proj_rank_total must be positive")
    if args.proj_rank_min <= 0:
        raise ValueError("--proj_rank_min must be positive")
        
    # Run selected mode
    if args.mode == 'project':
        run_projection(args)
    elif args.mode == 'non_interf':
        success = test_non_interference(args)
        sys.exit(0 if success else 1)
    elif args.mode == 'naive_check':
        success = test_naive_equality(args)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()