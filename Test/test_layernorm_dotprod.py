import os
import sys
import copy
from typing import Dict

# Ensure repo root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.nn import functional as F

from ghostEngines.graddotprod_engine import GradDotProdEngine


class LNNet(nn.Module):
    """Tiny network with a LayerNorm block and a linear head."""
    def __init__(self, in_dim: int = 12, hidden_dim: int = 8, out_dim: int = 5):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.ln(x)
        x = self.act(x)
        return self.head(x)


def _seed_everything(seed: int = 1234):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_synth_data(n_train: int, n_val: int, in_dim: int, out_dim: int, device: str = "cpu"):
    X_train = torch.randn(n_train, in_dim, device=device)
    X_val = torch.randn(n_val, in_dim, device=device)
    # Stable labels from a fixed random linear rule
    W = torch.randn(in_dim, out_dim, device=device)
    with torch.no_grad():
        logits_train = X_train @ W
        logits_val = X_val @ W
    Y_train = logits_train.argmax(dim=-1)
    Y_val = logits_val.argmax(dim=-1)
    return X_train, Y_train, X_val, Y_val


def _autograd_dp_for_params(model: nn.Module,
                            X_train: torch.Tensor,
                            Y_train: torch.Tensor,
                            X_val: torch.Tensor,
                            Y_val: torch.Tensor,
                            param_fullnames: Dict[str, None]) -> Dict[str, torch.Tensor]:
    """Compute per-sample gradient dot-products for specific parameters via autograd.

    - Compute validation gradient once with reduction='sum' and scale by 1/(n_train+n_val).
    - For each training sample i, compute its gradient with reduction='sum' and the same scale.
    - Return a dict mapping param full name to a vector of length n_train.
    """
    n_train = X_train.size(0)
    n_total = n_train + X_val.size(0)
    scale = 1.0 / float(n_total)

    # Clone model to avoid interference
    model = copy.deepcopy(model)
    device = X_train.device

    # Validation gradient (sum) then scale
    model.zero_grad(set_to_none=True)
    val_logits = model(X_val)
    val_loss = F.cross_entropy(val_logits, Y_val, reduction='sum')
    val_loss.backward()

    val_grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if name in param_fullnames and p.grad is not None:
            val_grads[name] = p.grad.detach().clone() * scale

    # Prepare outputs
    results: Dict[str, torch.Tensor] = {name: torch.zeros(n_train, dtype=torch.float32, device=device)
                                        for name in param_fullnames}

    # Per-training-sample grads (sum) and dot with validation
    for i in range(n_train):
        model.zero_grad(set_to_none=True)
        tr_logits = model(X_train[i:i+1])
        tr_loss = F.cross_entropy(tr_logits, Y_train[i:i+1], reduction='sum')
        tr_loss.backward()
        for name, p in model.named_parameters():
            if name in param_fullnames and p.grad is not None:
                dp = torch.dot(val_grads[name].reshape(-1), (p.grad * scale).reshape(-1))
                results[name][i] = dp

    return results


def test_layernorm_dotprod_shape_and_value_mismatch():
    """LayerNorm weight/bias dot-products should be per-sample scalars [B_train].

    This test asserts the correct semantics. With the current implementation,
    we expect it to FAIL due to a shape bug (engine returns [B_train, F]).
    """
    _seed_everything(0)
    device = 'cpu'

    in_dim, hidden_dim, out_dim = 12, 8, 5
    n_train, n_val = 6, 3
    X_tr, Y_tr, X_val, Y_val = _make_synth_data(n_train, n_val, in_dim, out_dim, device=device)
    X_cat = torch.cat([X_tr, X_val], dim=0)
    Y_cat = torch.cat([Y_tr, Y_val], dim=0)

    # Model with engine to get engine outputs
    _seed_everything(42)
    model_eng = LNNet(in_dim, hidden_dim, out_dim).to(device)
    opt = torch.optim.SGD(model_eng.parameters(), lr=1e-2)
    engine = GradDotProdEngine(module=model_eng, val_batch_size=n_val, loss_reduction='mean', use_dummy_bias=False)
    engine.attach(opt)

    logits = model_eng(X_cat)
    loss = F.cross_entropy(logits, Y_cat, reduction='mean')
    loss.backward()

    # Collect engine LN param dot products
    eng_dp = {}
    for name, p in model_eng.named_parameters():
        if name.endswith('ln.weight') or name.endswith('ln.bias'):
            eng_dp[name] = getattr(p, 'grad_dot_prod', None)

    # Naive autograd baseline for LN params only
    _seed_everything(42)
    model_ref = LNNet(in_dim, hidden_dim, out_dim).to(device)
    model_ref.load_state_dict(copy.deepcopy(model_eng.state_dict()))

    wanted = {n: None for n in eng_dp.keys()}
    auto_dp = _autograd_dp_for_params(model_ref, X_tr, Y_tr, X_val, Y_val, wanted)

    # Assert correct semantics: engine should return 1D [n_train] vectors matching autograd
    for name in eng_dp:
        assert eng_dp[name] is not None, f"Engine did not produce grad_dot_prod for {name}"
        # Expect a 1D vector; current buggy code returns 2D [B, F]
        assert eng_dp[name].dim() == 1 and eng_dp[name].numel() == n_train, \
            f"BUG: {name} grad_dot_prod has shape {tuple(eng_dp[name].shape)}, expected [{n_train}]"
        # Value closeness to autograd baseline
        assert torch.allclose(eng_dp[name].float(), auto_dp[name].float(), atol=5e-4, rtol=0), \
            f"BUG: {name} grad_dot_prod values differ from autograd baseline.\nengine={eng_dp[name]}\nautograd={auto_dp[name]}"


def test_aggregator_runs_with_layernorm():
    """Aggregator runs and returns a per-sample vector when LayerNorm is present.

    This validates integration despite the current LN value bug.
    """
    _seed_everything(1)
    device = 'cpu'

    in_dim, hidden_dim, out_dim = 10, 6, 4
    n_train, n_val = 5, 3
    X_tr = torch.randn(n_train, in_dim, device=device)
    Y_tr = torch.randint(0, out_dim, (n_train,), device=device)
    X_val = torch.randn(n_val, in_dim, device=device)
    Y_val = torch.randint(0, out_dim, (n_val,), device=device)
    X_cat = torch.cat([X_tr, X_val], dim=0)
    Y_cat = torch.cat([Y_tr, Y_val], dim=0)

    model = LNNet(in_dim, hidden_dim, out_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    engine = GradDotProdEngine(module=model, val_batch_size=n_val, loss_reduction='mean', use_dummy_bias=False)
    engine.attach(opt)

    logits = model(X_cat)
    loss = F.cross_entropy(logits, Y_cat, reduction='mean')
    loss.backward()

    # Provide batch metadata expected by aggregator
    engine.attach_train_batch(X_tr, Y_tr, iter_num=0, batch_idx=0)
    # Should not raise
    engine.aggregate_and_log()
    assert len(engine.dot_product_log) == 1
    dp = engine.dot_product_log[0]['dot_product']
    assert isinstance(dp, torch.Tensor) and dp.dim() == 1 and dp.numel() == n_train


if __name__ == "__main__":
    # Run as a simple script
    test_layernorm_dotprod_shape_and_value_mismatch()
    print("[PASS] LayerNorm dot-products match autograd baseline")
    test_aggregator_runs_with_layernorm()
    print("[PASS] Aggregator runs and returns per-sample vector")
