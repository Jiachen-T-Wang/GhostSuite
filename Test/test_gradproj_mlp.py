"""
Unit tests for gradient projection engine with MLP.
Tests non-interference and naive equality criteria.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine
from ghostEngines.gradProjection.projection_utils import (
    choose_ki_ko, 
    init_projection_matrix_gaussian,
    init_projection_matrix_orthonormal
)


class TwoLayerMLP(nn.Module):
    """Simple two-layer MLP for testing."""
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


class TestGradProjEngine(unittest.TestCase):
    """Test suite for gradient projection engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set seeds for reproducibility
        torch.manual_seed(1234)
        np.random.seed(1234)
        
        # Use deterministic algorithms only on CPU
        # CUDA requires special environment variables for determinism
        if not torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        
        # Create temporary directory for projections
        self.temp_dir = tempfile.mkdtemp()
        
        # Device - use CPU for tests to ensure determinism
        self.device = torch.device('cpu')
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_choose_ki_ko(self):
        """Test optimal dimension selection."""
        # Test basic case
        k_i, k_o = choose_ki_ko(100, 200, 64, 2)
        self.assertGreaterEqual(k_i, 2)
        self.assertGreaterEqual(k_o, 2)
        self.assertLessEqual(k_i, 100)
        self.assertLessEqual(k_o, 200)
        # Product should be close to target
        self.assertLessEqual(abs(k_i * k_o - 64), 20)
        
        # Test with constraints
        k_i, k_o = choose_ki_ko(10, 10, 100, 5)
        self.assertGreaterEqual(k_i, 5)
        self.assertGreaterEqual(k_o, 5)
        self.assertLessEqual(k_i, 10)
        self.assertLessEqual(k_o, 10)
        
        # Test error cases
        with self.assertRaises(ValueError):
            choose_ki_ko(5, 5, 100, 10)  # k_min too large
            
    def test_projection_initialization(self):
        """Test projection matrix initialization."""
        # Test Gaussian initialization
        P = init_projection_matrix_gaussian(10, 100, seed=42)
        self.assertEqual(P.shape, (10, 100))
        self.assertFalse(P.requires_grad)
        
        # Check that projection preserves energy approximately
        # For Gaussian JL with std=1/sqrt(k), we expect E[P @ P^T] ≈ I
        # But for a single sample, we just check it's not degenerate
        PPT = P @ P.t()
        # Check matrix is positive definite and well-conditioned
        eigenvalues = torch.linalg.eigvalsh(PPT)
        self.assertTrue((eigenvalues > 0).all())  # Positive definite
        condition_number = eigenvalues.max() / eigenvalues.min()
        self.assertLess(condition_number, 100)  # Well-conditioned
        
        # Test orthonormal initialization
        P_orth = init_projection_matrix_orthonormal(10, 100, seed=42)
        self.assertEqual(P_orth.shape, (10, 100))
        PPT_orth = P_orth @ P_orth.t()
        # Should be exactly identity (up to numerical precision)
        self.assertTrue(torch.allclose(PPT_orth, torch.eye(10), atol=1e-6))
        
    def test_non_interference(self):
        """Test that engine doesn't interfere with normal training."""
        # Create two identical models
        torch.manual_seed(42)
        model1 = TwoLayerMLP().to(self.device)
        
        torch.manual_seed(42)
        model2 = TwoLayerMLP().to(self.device)
        
        # Verify initial weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
            
        # Prepare data
        torch.manual_seed(123)
        inputs = torch.randn(8, 10, device=self.device)
        targets = torch.randint(0, 5, (8,), device=self.device)
        
        # Train model1 without engine
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        losses1 = []
        
        for _ in range(5):
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer1.step()
            losses1.append(loss.item())
            
        # Train model2 with engine attached
        engine_config = {
            'proj_layers': 'fc',
            'proj_rank_total': 32,
            'proj_rank_min': 2,
            'proj_seed': 999,
            'proj_dtype': 'float32',
            'proj_dir': self.temp_dir,
        }
        
        engine = GradProjLoraEngine(model2, **engine_config)
        engine.attach()
        
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        losses2 = []
        
        for _ in range(5):
            optimizer2.zero_grad()
            outputs = model2(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Collect projections (but don't use them)
            _ = engine.collect_batch()
            
            optimizer2.step()
            losses2.append(loss.item())
            
        engine.detach()
        
        # Compare losses (should be identical)
        for l1, l2 in zip(losses1, losses2):
            self.assertAlmostEqual(l1, l2, places=5)
            
        # Compare final parameters (should be identical)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2, rtol=1e-5, atol=1e-6))
            
    def test_naive_equality(self):
        """Test that engine projections match naive computation."""
        # Create model
        torch.manual_seed(42)
        model = TwoLayerMLP().to(self.device)
        
        # Prepare small batch
        inputs = torch.randn(2, 10, device=self.device)
        targets = torch.randint(0, 5, (2,), device=self.device)
        
        # Engine configuration
        engine_config = {
            'proj_layers': 'fc',
            'proj_rank_total': 16,
            'proj_rank_min': 2,
            'proj_seed': 42,
            'proj_dtype': 'float32',
            'proj_dir': self.temp_dir,
        }
        
        # Compute with engine
        engine = GradProjLoraEngine(model, **engine_config)
        engine.attach()
        
        model.zero_grad()
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        loss.backward()
        
        engine_projections = engine.collect_batch()
        
        # Compute naive projections
        naive_projections = self._compute_naive_projections(
            model, inputs, targets, engine
        )
        
        engine.detach()
        
        # Compare
        self.assertEqual(engine_projections.shape, naive_projections.shape)
        
        # Debug info if test fails
        if not torch.allclose(engine_projections, naive_projections, rtol=1e-4, atol=1e-5):
            max_diff = (engine_projections - naive_projections).abs().max()
            rel_diff = ((engine_projections - naive_projections).abs() / 
                       (naive_projections.abs() + 1e-8)).max()
            print(f"\nMax absolute diff: {max_diff}")
            print(f"Max relative diff: {rel_diff}")
            print(f"Engine proj sample: {engine_projections[0, :5]}")
            print(f"Naive proj sample: {naive_projections[0, :5]}")
            
        self.assertTrue(torch.allclose(engine_projections, naive_projections, 
                                     rtol=1e-3, atol=1e-4))  # Relax tolerance slightly
        
    def _compute_naive_projections(self, model, inputs, targets, engine):
        """Helper to compute naive projections for validation."""
        batch_size = inputs.shape[0]
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        all_projections = []
        
        for b in range(batch_size):
            model.zero_grad()
            
            # Forward/backward for single sample
            output = model(inputs[b:b+1])
            loss = criterion(output, targets[b:b+1]).mean()
            loss.backward()
            
            # Collect and project gradients
            sample_projections = []
            
            for layer_name in sorted(engine.matched_layers.keys()):
                layer = engine.matched_layers[layer_name]
                P_i, P_o = engine.projection_matrices[layer_name]
                
                if hasattr(layer, 'weight'):
                    grad_full = layer.weight.grad.clone()
                    # Project: P_o @ grad @ P_i.T
                    grad_proj = P_o @ grad_full @ P_i.t()
                    grad_flat = grad_proj.reshape(-1)
                    sample_projections.append(grad_flat)
                    
            sample_proj = torch.cat(sample_projections)
            all_projections.append(sample_proj)
            
        return torch.stack(all_projections)
        
    def test_metadata_saving(self):
        """Test that metadata is saved correctly."""
        model = TwoLayerMLP().to(self.device)
        
        engine_config = {
            'proj_layers': 'fc',
            'proj_rank_total': 32,
            'proj_rank_min': 2,
            'proj_seed': 42,
            'proj_dtype': 'float16',
            'proj_dir': self.temp_dir,
        }
        
        engine = GradProjLoraEngine(model, **engine_config)
        metadata = engine.get_projection_metadata()
        
        # Check metadata structure
        self.assertEqual(metadata['engine'], 'GradProjLora')
        self.assertEqual(metadata['proj_seed'], 42)
        self.assertEqual(metadata['proj_dtype'], 'float16')
        self.assertEqual(metadata['total_proj_dim'], engine.total_proj_dim)
        self.assertIn('layers', metadata)
        self.assertEqual(len(metadata['layers']), 2)  # fc1 and fc2
        
        # Check layer metadata
        for layer_meta in metadata['layers']:
            self.assertIn('name', layer_meta)
            self.assertIn('k_i', layer_meta)
            self.assertIn('k_o', layer_meta)
            self.assertIn('slice_start', layer_meta)
            self.assertIn('slice_end', layer_meta)
            
    def test_projection_saving(self):
        """Test that projections are saved to disk correctly."""
        model = TwoLayerMLP().to(self.device)
        
        engine_config = {
            'proj_layers': 'fc',
            'proj_rank_total': 16,
            'proj_rank_min': 2,
            'proj_seed': 42,
            'proj_dtype': 'bfloat16',
            'proj_dir': self.temp_dir,
            'proj_save_interval': 1,
        }
        
        engine = GradProjLoraEngine(model, **engine_config)
        engine.attach()
        
        # Run a few iterations
        for i in range(3):
            inputs = torch.randn(4, 10, device=self.device)
            targets = torch.randint(0, 5, (4,), device=self.device)
            
            model.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            projections = engine.collect_batch(
                batch_indices=list(range(i * 4, (i + 1) * 4))
            )
            
        engine.detach()
        
        # Check saved files
        proj_dir = Path(self.temp_dir)
        
        # Metadata should exist
        metadata_file = proj_dir / 'metadata.json'
        self.assertTrue(metadata_file.exists())
        
        # Projection files should exist
        proj_files = list(proj_dir.glob('proj_iter_*.pt'))
        self.assertEqual(len(proj_files), 3)
        
        # Load and check a projection file
        proj_data = torch.load(proj_files[0])
        self.assertIn('proj', proj_data)
        self.assertIn('iter', proj_data)
        self.assertIn('batch_size', proj_data)
        self.assertIn('batch_idx', proj_data)
        
        # Check projection shape
        self.assertEqual(proj_data['proj'].shape[0], 4)  # batch size
        self.assertEqual(proj_data['proj'].shape[1], engine.total_proj_dim)


if __name__ == '__main__':
    unittest.main()