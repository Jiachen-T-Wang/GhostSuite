import math
import os
import sys
import copy
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch import nn
from torch.nn import functional as F

from ghostEngines.graddotprod_engine import GradDotProdEngine


class TwoLayerMLP(nn.Module):
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


def _seed_everything(seed: int = 1234):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_synth_data(n_train: int, n_val: int, in_dim: int, num_classes: int, device: str = "cpu"):
    X_train = torch.randn(n_train, in_dim, device=device)
    X_val = torch.randn(n_val, in_dim, device=device)
    # Create labels by a simple linear rule for stability
    W = torch.randn(in_dim, num_classes, device=device)
    with torch.no_grad():
        logits_train = X_train @ W
        logits_val = X_val @ W
    Y_train = logits_train.argmax(dim=-1)
    Y_val = logits_val.argmax(dim=-1)
    return X_train, Y_train, X_val, Y_val


def _capture_AB_linear(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Run a forward+backward on a copy of the model and capture (A,B) for each Linear."""
    handles: List[torch.utils.hooks.RemovableHandle] = []
    cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def fwd_hook(layer, inp, out):
        cache[id(layer)]['A'] = inp[0].detach()

    def bwd_hook(layer, grad_input, grad_output):
        cache[id(layer)]['B'] = grad_output[0].detach()

    # Register hooks on Linear layers
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            cache[id(m)] = {}
            handles.append(m.register_forward_hook(fwd_hook))
            handles.append(m.register_full_backward_hook(bwd_hook))

    # Forward/backward
    logits = model(X)
    loss = F.cross_entropy(logits, y, reduction='mean')
    loss.backward()

    # Build map by module name for deterministic lookup
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and id(m) in cache and 'A' in cache[id(m)] and 'B' in cache[id(m)]:
            result[name] = (cache[id(m)]['A'], cache[id(m)]['B'])

    for h in handles:
        h.remove()

    return result


def _naive_linear_dotprod(A: torch.Tensor, B: torch.Tensor, train_bs: int, val_bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naively compute per-sample dot products for weight and bias for a Linear layer.

    Supports A of shape [B, ..., d] and B of shape [B, ..., p].
    """
    total_bs = A.size(0)
    assert total_bs == train_bs + val_bs

    # Match engine's dtype behavior (bf16 for dot-product path)
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    A_train, A_val = A.split([train_bs, val_bs], dim=0)
    B_train, B_val = B.split([train_bs, val_bs], dim=0)

    # Materialize gradients
    grad_train = torch.einsum('b...p,b...d->bpd', B_train, A_train)  # [train_bs, p, d] (bf16)

    # Correct aggregated validation gradient: sum over (token) positions, no cross terms
    p = B.size(-1)
    d = A.size(-1)
    grad_val = torch.einsum('np,nd->pd', B_val.reshape(-1, p), A_val.reshape(-1, d))  # [p, d] (bf16)

    # Frobenius inner product for each training sample (accumulate in fp32)
    weight_dot = torch.einsum('pd,bpd->b', grad_val.float(), grad_train.float())

    # Bias: per-sample is sum over non-feature dims of B
    sum_dims_train = list(range(1, B_train.dim() - 1))
    grad_bias_train = B_train.sum(dim=sum_dims_train) if B_train.dim() > 2 else B_train  # [train_bs, p]
    grad_bias_val = B_val.sum(dim=list(range(B_val.dim() - 1)))  # [p]
    bias_dot = torch.einsum('p,bp->b', grad_bias_val.float(), grad_bias_train.float())

    return weight_dot, bias_dot


def _naive_autograd_per_sample_dotprod(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extremely naive baseline using autograd gradients with batch_size=1 for train samples.

    - Compute validation gradient once using reduction='sum' (sum over val set).
    - For each training sample i, compute its gradient with reduction='sum'.
    - For every Linear layer param (weight/bias), compute dot(param_grad_val, param_grad_train_i).

    Returns a dict mapping layer base name (e.g., 'fc1') to a tuple of tensors:
    (weight_dot_products[n_train], bias_dot_products[n_train or None]).
    """

    model = copy.deepcopy(model)
    model.zero_grad(set_to_none=True)

    n_train = X_train.size(0)

    # 1) Validation gradient (sum over val set), then scale to match single-mean backward on train+val
    val_logits = model(X_val)
    val_loss = F.cross_entropy(val_logits, Y_val, reduction='sum')
    val_loss.backward()

    # Collect per-parameter validation grads by layer base name
    val_grads: Dict[str, Dict[str, torch.Tensor]] = {}
    n_total = n_train + X_val.size(0)
    scale = 1.0 / float(n_total)

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name.endswith('.weight'):
            lname = name[:-7]
            val_grads.setdefault(lname, {})['weight'] = (p.grad.detach().clone() * scale)
        elif name.endswith('.bias'):
            lname = name[:-5]
            val_grads.setdefault(lname, {})['bias'] = (p.grad.detach().clone() * scale)

    # Prepare result containers
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for lname, grads in val_grads.items():
        w_len = n_train if 'weight' in grads else 0
        b_len = n_train if 'bias' in grads else 0
        w_vec = torch.zeros(w_len, dtype=torch.float32, device=X_train.device) if w_len else None
        b_vec = torch.zeros(b_len, dtype=torch.float32, device=X_train.device) if b_len else None
        results[lname] = (w_vec, b_vec)

    # 2) For each training sample, compute per-sample gradient and dot with val grads
    for i in range(n_train):
        model.zero_grad(set_to_none=True)
        tr_logits = model(X_train[i:i+1])
        tr_loss = F.cross_entropy(tr_logits, Y_train[i:i+1], reduction='sum')
        tr_loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name.endswith('.weight'):
                lname = name[:-7]
                if lname in val_grads and 'weight' in val_grads[lname]:
                    dp = torch.dot(val_grads[lname]['weight'].reshape(-1), (p.grad * scale).reshape(-1))
                    results[lname][0][i] = dp
            elif name.endswith('.bias'):
                lname = name[:-5]
                if lname in val_grads and 'bias' in val_grads[lname] and results[lname][1] is not None:
                    dp = torch.dot(val_grads[lname]['bias'].reshape(-1), (p.grad * scale).reshape(-1))
                    results[lname][1][i] = dp

    return results


def test_grad_dotprod_linear_correctness():
    """Validate that engine-computed dot products match naive materialization on a 2-layer MLP."""

    _seed_everything(0)
    device = 'cpu'

    in_dim, hidden_dim, out_dim = 12, 8, 5
    n_train, n_val = 7, 3

    X_tr, Y_tr, X_val, Y_val = _make_synth_data(n_train, n_val, in_dim, out_dim, device=device)
    X_cat = torch.cat([X_tr, X_val], dim=0)
    Y_cat = torch.cat([Y_tr, Y_val], dim=0)

    # Model A: with engine, just to get engine's per-layer dot products
    _seed_everything(42)
    model_A = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    opt_A = torch.optim.SGD(model_A.parameters(), lr=1e-2)
    engine = GradDotProdEngine(module=model_A, val_batch_size=n_val, loss_reduction='mean', average_grad=True, use_dummy_bias=False)
    engine.attach(opt_A)

    logits = model_A(X_cat)
    loss = F.cross_entropy(logits, Y_cat, reduction='mean')
    loss.backward()

    # Collect engine-computed dot products
    engine_dps: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, p in model_A.named_parameters():
        if hasattr(p, 'grad_dot_prod'):
            if name.endswith('.weight'):
                lname = name[:-7]
                engine_dps.setdefault(lname, [None, None])[0] = p.grad_dot_prod.detach().clone()
            elif name.endswith('.bias'):
                lname = name[:-5]
                engine_dps.setdefault(lname, [None, None])[1] = p.grad_dot_prod.detach().clone()

    # Model B: no engine, capture A/B and compute naive dot products
    _seed_everything(42)
    model_B = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    model_B.load_state_dict(copy.deepcopy(model_A.state_dict()))

    AB_map = _capture_AB_linear(model_B, X_cat, Y_cat)

    # Compare per-layer
    for lname, (A, B) in AB_map.items():
        w_dp_naive, b_dp_naive = _naive_linear_dotprod(A, B, n_train, n_val)
        w_dp_engine, b_dp_engine = engine_dps[lname]

        if not torch.allclose(w_dp_naive, w_dp_engine, atol=5e-4, rtol=0):
            # Try reproducing engine's ghost formula directly on captured A/B for debugging
            A_bf16 = A.to(torch.bfloat16)
            B_bf16 = B.to(torch.bfloat16)
            total_bs = A_bf16.size(0)
            d = A_bf16.size(-1)
            p = B_bf16.size(-1)
            T = int(A_bf16.numel() // (total_bs * d))
            A_bf16 = A_bf16.reshape(total_bs, T, d)
            B_bf16 = B_bf16.reshape(total_bs, T, p)
            A_tr = A_bf16[:n_train].reshape(-1, d)
            A_vl = A_bf16[n_train:].reshape(-1, d)
            B_tr = B_bf16[:n_train].reshape(-1, p)
            B_vl = B_bf16[n_train:].reshape(-1, p)
            a_dot = A_tr @ A_vl.T
            b_dot = B_tr @ B_vl.T
            token_contrib = (a_dot * b_dot).sum(dim=1, dtype=torch.float32)
            w_dp_engine_alt = token_contrib.reshape(n_train, T).sum(dim=1)

            raise AssertionError(
                f"Weight DP mismatch in layer {lname}\n"
                f"naive (mat-grad): {w_dp_naive}\n"
                f"engine saved:     {w_dp_engine}\n"
                f"engine formula:   {w_dp_engine_alt}\n"
            )

        if b_dp_engine is not None:  # bias exists
            assert torch.allclose(b_dp_naive, b_dp_engine, atol=5e-4, rtol=0), f"Bias DP mismatch in layer {lname}"

    # Also compare with an even more naive autograd baseline (batch_size=1 per train sample)
    autograd_dp = _naive_autograd_per_sample_dotprod(model_B, X_tr, Y_tr, X_val, Y_val)
    for lname, (w_vec, b_vec) in autograd_dp.items():
        w_dp_engine, b_dp_engine = engine_dps[lname]
        if w_vec is not None:
            assert torch.allclose(w_vec, w_dp_engine, atol=5e-4, rtol=0), f"Autograd weight DP mismatch in layer {lname}"
        if b_vec is not None and b_dp_engine is not None:
            assert torch.allclose(b_vec, b_dp_engine, atol=5e-4, rtol=0), f"Autograd bias DP mismatch in layer {lname}"


def test_training_equivalence_validation_loss():
    """Ensure hooks and engine-based gradient application do not change training behavior.

    Train 2 steps on synthetic data with and without engine; the validation loss must be exactly equal.
    """
    _seed_everything(1)
    device = 'cpu'

    in_dim, hidden_dim, out_dim = 10, 8, 4
    n_train, n_val = 8, 4
    steps = 5

    X_tr, Y_tr, X_val, Y_val = _make_synth_data(n_train, n_val, in_dim, out_dim, device=device)

    # Baseline (no engine): train only on training subset
    _seed_everything(100)
    model_ref = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=1e-1)

    for _ in range(steps):
        logits = model_ref(X_tr)
        loss = F.cross_entropy(logits, Y_tr, reduction='mean')
        opt_ref.zero_grad(set_to_none=True)
        loss.backward()
        opt_ref.step()

    with torch.no_grad():
        val_logits_ref = model_ref(X_val)
        val_loss_ref = F.cross_entropy(val_logits_ref, Y_val, reduction='mean')

    # Engine run: concatenate train+val for forward/backward, but update must equal baseline
    _seed_everything(100)
    model_eng = TwoLayerMLP(in_dim, hidden_dim, out_dim).to(device)
    opt_eng = torch.optim.SGD(model_eng.parameters(), lr=1e-1)
    engine = GradDotProdEngine(module=model_eng, val_batch_size=n_val, loss_reduction='mean', average_grad=True, use_dummy_bias=False)
    engine.attach(opt_eng)

    X_cat = torch.cat([X_tr, X_val], dim=0)
    Y_cat = torch.cat([Y_tr, Y_val], dim=0)

    for _ in range(steps):
        logits = model_eng(X_cat)
        loss = F.cross_entropy(logits, Y_cat, reduction='mean')
        opt_eng.zero_grad(set_to_none=True)
        loss.backward()
        # Replace .grad with engine-accumulated training gradients
        engine.prepare_gradients()
        opt_eng.step()
        # Clear for next iteration
        engine.clear_gradients()

    with torch.no_grad():
        val_logits_eng = model_eng(X_val)
        val_loss_eng = F.cross_entropy(val_logits_eng, Y_val, reduction='mean')

    # Exact match
    assert torch.equal(val_loss_ref, val_loss_eng), f"Validation losses differ: {val_loss_ref.item()} vs {val_loss_eng.item()}"


if __name__ == "__main__":
    # Run as a simple script
    test_grad_dotprod_linear_correctness()
    print("[PASS] Gradient dot-product is correct.")
    test_training_equivalence_validation_loss()
    print("[PASS] Training equivalence: validation loss matches exactly.")
