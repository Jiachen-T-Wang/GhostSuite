"""
Minimal MLP training example using GhostEngine (gradient dot-product).
"""

import os
import sys
from typing import Tuple

# Ensure repo root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.nn import functional as F

from ghostEngines import GradDotProdEngine


class TwoLayerMLP(nn.Module):
    """A minimal two-layer MLP for classification."""

    def __init__(self, in_dim: int = 10, hidden_dim: int = 16, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def seed_everything(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_synth_data(
    n_train: int,
    n_val: int,
    in_dim: int,
    num_classes: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small synthetic classification set with a linear groundtruth."""

    X_train = torch.randn(n_train, in_dim, device=device)
    X_val = torch.randn(n_val, in_dim, device=device)
    W = torch.randn(in_dim, num_classes, device=device)
    with torch.no_grad():
        Y_train = (X_train @ W).argmax(dim=-1)
        Y_val = (X_val @ W).argmax(dim=-1)
    return X_train, Y_train, X_val, Y_val


def demo_train_with_engine() -> None:
    """Run a tiny training loop with GradDotProdEngine attached.

    Steps:
    1) Build data/model/optimizer.
    2) Attach GhostEngine with validation batch size.
    3) Train for multiple steps on concatenated (train + val) batch.
    4) Replace .grad with accumulated training gradients and step.
    5) Access and print per-parameter gradient dot products.
    """

    seed_everything(100)
    device = "cpu"

    in_dim, hidden_dim, out_dim = 10, 8, 4
    n_train, n_val = 8, 4

    X_tr, Y_tr, X_val, Y_val = make_synth_data(n_train, n_val, in_dim, out_dim, device=device)

    model = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    engine = GradDotProdEngine(
        module=model,
        val_batch_size=n_val,
        loss_reduction="mean",
        average_grad=True,
        use_dummy_bias=False,
    )
    engine.attach(optimizer)

    # Concatenate train and val once; reuse across steps
    X_cat = torch.cat([X_tr, X_val], dim=0)
    Y_cat = torch.cat([Y_tr, Y_val], dim=0)

    # Train for 10 steps with per-iteration logging
    steps = 10
    for step in range(steps):
        # Optional: store current train batch metadata for aggregation utilities
        engine.attach_train_batch(X_train=X_tr, Y_train=Y_tr, iter_num=step, batch_idx=0)

        logits = model(X_cat)
        loss = F.cross_entropy(logits, Y_cat, reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Print per-parameter gradient dot products this iteration (before aggregation clears them)
        print(f"\n[Iter {step}] Per-parameter gradient dot products (val ⋅ train):")
        for name, p in model.named_parameters():
            if hasattr(p, "grad_dot_prod"):
                vec = p.grad_dot_prod.detach().cpu()
                print(f"  {name:20s} shape={tuple(vec.shape)} values={vec.tolist()}")

        # Aggregate across parameters and log this iteration
        engine.aggregate_and_log()
        if engine.dot_product_log:
            agg = engine.dot_product_log[-1]["dot_product"].detach().cpu()
            print(f"[Iter {step}] Aggregated dot product across parameters: {agg}")

        # Move accumulated training gradients into .grad so the optimizer can update
        engine.prepare_gradients()
        optimizer.step()
        engine.clear_gradients()

    # Show validation loss to confirm the model runs end-to-end
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = F.cross_entropy(val_logits, Y_val, reduction="mean")
    print(f"\nValidation loss after {steps} steps: {val_loss.item():.6f}")
    

if __name__ == "__main__":
    demo_train_with_engine()
