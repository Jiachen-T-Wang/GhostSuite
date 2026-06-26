"""Direct-on-model correctness tests for Data Value Embedding.

Complements validate_dve.py (which tests the recursion + projection on abstract tensors)
with two checks that exercise the real machinery on real models:

  Test A (ghost-capture correctness, on GPT2): the engine's per-sample *projected*
      gradient must equal a brute-force projected gradient assembled from per-sample
      backward passes -> torch.allclose. Validates the P_i (x) P_o ghost trick, the
      token-sum, the batch-size rescale, and the per-layer dispatch/slicing directly on
      a GPT2 model.

  Test B (exact unrolled-SGD influence, on a small MLP): the semantic ground truth that
      the reference repo lacks. The TRUE first-order influence dL_test/dw_{s,b} is
      obtained by autodiff *through the SGD trajectory* (double-backprop, no Gauss-Newton,
      no projection). DVE values (which use the Gauss-Newton outer-product Hessian) must
      track it -- and track it at least as well as a plain gradient-dot (TracIn-style)
      baseline, showing the reverse recursion adds signal.

Run:  python validate_dve_model.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ghostEngines.gradProjection.gradproj_engine import GradProjLoraEngine
from ghostEngines.gradProjection.dve_embedding import dve_recursion


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# ======================================================================================
# Test A: ghost-capture correctness on GPT2
# ======================================================================================
def test_ghost_capture_gpt2(seed=0, device=None):
    print("\n[Test A] Ghost capture vs brute-force projected per-sample grad (GPT2)...")
    from shared.gpt2 import GPT, GPTConfig
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    V, T, B = 256, 16, 4
    cfg = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=T,
                    bias=False, vocab_size=V, dropout=0.0)
    model = GPT(cfg).to(device).to(torch.float32)
    model.eval()
    model.config.use_cache = False
    try:
        from ghostEngines import transformers_support
        transformers_support.forward_swapper(model)
    except ImportError:
        pass

    tmp = tempfile.mkdtemp()
    engine = GradProjLoraEngine(model, proj_layers='mlp,attn', proj_rank_total=32,
                                proj_rank_min=4, proj_seed=0, proj_dtype='float32',
                                proj_dir=tmp, proj_save_interval=10 ** 12)
    engine.proj_save_interval = 10 ** 12
    engine.attach()

    g = torch.Generator(device='cpu').manual_seed(seed + 1)
    X = torch.randint(0, V, (B, T), generator=g).to(device)
    Y = torch.randint(0, V, (B, T), generator=g).to(device)

    # Engine pass: combined batch -> per-sample projected grads [B, total].
    model.zero_grad(set_to_none=True)
    loss = model(X, Y).loss
    loss.backward()
    engine_out = engine.collect_batch().float().cpu()  # [B, total]
    engine.clear_gradients()

    # Brute force: each sample alone, full per-sample layer grad -> project -> compare.
    names = sorted(engine.matched_layers.keys())
    max_rel = 0.0
    for b in range(B):
        brute_blocks = []
        for name in names:
            layer = engine.matched_layers[name]
            P_i, P_o = engine.projection_matrices[name]  # [k_i,n_i], [k_o,n_o]
            model.zero_grad(set_to_none=True)
            lb = model(X[b:b + 1], Y[b:b + 1]).loss  # CE mean over this sample's T tokens
            lb.backward()
            W_grad = layer.weight.grad.detach().float()  # nn.Linear: [n_o, n_i]
            proj = P_o.float() @ W_grad @ P_i.float().t()  # [k_o, k_i]
            brute_blocks.append(proj.reshape(-1).cpu())    # row-major == engine reshape
        brute = torch.cat(brute_blocks)
        eng = engine_out[b]
        rel = (brute - eng).norm().item() / (eng.norm().item() + 1e-12)
        max_rel = max(max_rel, rel)

    engine.detach()
    print(f"  matched layers: {len(names)}, total_proj_dim={engine.total_proj_dim}")
    print(f"  max per-sample relative L2 error: {max_rel:.3e}")
    assert max_rel < 1e-4, f"ghost capture mismatch: {max_rel}"
    print("  PASS")


# ======================================================================================
# Test B: exact unrolled-SGD influence on a small MLP
# ======================================================================================
def _flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
    return torch.cat([g.reshape(-1) for g in gs])


def test_unrolled_influence_mlp(seed=0):
    print("\n[Test B] DVE vs exact unrolled-SGD influence (small MLP, autodiff)...")
    from torch.func import functional_call
    torch.manual_seed(seed)

    D_in, H, D_out = 8, 16, 3
    K, B, eta = 6, 8, 0.2
    n_test = 6

    model = nn.Sequential(nn.Linear(D_in, H), nn.Tanh(), nn.Linear(H, D_out))
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    pkeys = list(base.keys())

    Xtr = torch.randn(K, B, D_in)
    Ytr = torch.randint(0, D_out, (K, B))
    Xte = torch.randn(n_test, D_in)
    Yte = torch.randint(0, D_out, (n_test,))

    def fwd(p, x):
        return functional_call(model, p, (x,))

    # ---- Ground-truth influence via differentiable trajectory wrt per-sample weights ----
    W = torch.ones(K, B, requires_grad=True)
    # Initial params need requires_grad so the per-step update gradient (and thus the
    # whole trajectory's dependence on W) stays in the autograd graph.
    p = {k: base[k].clone().requires_grad_(True) for k in pkeys}
    for s in range(K):
        outs = fwd(p, Xtr[s])                                   # [B, D_out]
        losses = F.cross_entropy(outs, Ytr[s], reduction='none')  # [B]
        batch_loss = (W[s] * losses).mean()
        grads = torch.autograd.grad(batch_loss, [p[k] for k in pkeys], create_graph=True)
        p = {k: p[k] - eta * gi for k, gi in zip(pkeys, grads)}
    pK = p

    infl = torch.zeros(n_test, K * B)
    for j in range(n_test):
        lj = F.cross_entropy(fwd(pK, Xte[j:j + 1]), Yte[j:j + 1])
        gW = torch.autograd.grad(lj, W, retain_graph=True)[0]   # [K, B]
        infl[j] = gW.reshape(-1)

    # ---- DVE side: per-sample grads along the SAME (detached) trajectory ----
    p = {k: base[k].clone().requires_grad_(True) for k in pkeys}
    per_step_grads = []  # list of [B, D_flat]
    for s in range(K):
        gs = []
        for b in range(B):
            lb = F.cross_entropy(fwd(p, Xtr[s, b:b + 1]), Ytr[s, b:b + 1])
            gs.append(_flat_grad(lb, [p[k] for k in pkeys]))
        G = torch.stack(gs)              # [B, D_flat]
        per_step_grads.append(G.detach())
        gmean = G.mean(0)               # batch-mean grad (W=1) == ground-truth step
        with torch.no_grad():
            off = 0
            for k in pkeys:
                n = base[k].numel()
                p[k] -= eta * gmean[off:off + n].reshape(base[k].shape)
                off += n

    # test gradients at theta_K
    g_test = []
    for j in range(n_test):
        lj = F.cross_entropy(fwd(p, Xte[j:j + 1]), Yte[j:j + 1])
        g_test.append(_flat_grad(lj, [p[k] for k in pkeys]).detach())
    g_test = torch.stack(g_test)        # [n_test, D_flat]

    # DVE embeddings (single block = full params, no projection -> isolates the
    # first-order + Gauss-Newton approximations from projection error).
    embs = dve_recursion(per_step_grads, lrs=[eta] * K, lr_mode='scaled')
    E = torch.cat(embs, dim=0)          # [K*B, D_flat]
    dve_vals = g_test @ E.t()           # [n_test, K*B]

    # Plain gradient-dot baseline (TracIn-style: M=0, no recursion).
    G_all = torch.cat([g for g in per_step_grads], dim=0)  # [K*B, D_flat]
    tracin = (eta) * (g_test @ G_all.t())

    # DVE value ~= -B * influence (see README math); compare to -influence.
    target = (-infl).reshape(-1)
    pe_dve = _pearson(dve_vals.reshape(-1), target)
    pe_tracin = _pearson(tracin.reshape(-1), target)
    print(f"  Pearson(DVE,  -influence) = {pe_dve:+.4f}")
    print(f"  Pearson(grad-dot, -infl)  = {pe_tracin:+.4f}  (baseline, no recursion)")
    assert pe_dve > 0.85, f"DVE should track exact influence (got {pe_dve})"
    assert pe_dve >= pe_tracin - 1e-3, "recursion should not hurt vs plain grad-dot"
    print("  PASS (DVE tracks exact unrolled influence, >= grad-dot baseline)")


def main():
    test_ghost_capture_gpt2()
    test_unrolled_influence_mlp()
    print("\nAll direct-on-model DVE tests passed.")


if __name__ == '__main__':
    main()
