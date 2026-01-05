"""Benchmark training throughput and peak GPU memory with vs without Ghost dot-product computation.

Usage (from repo root):
  python tests/benchmark_ghost_dotprod.py --device auto

Notes:
- The ghost path uses a fixed validation batch appended to each train batch.
- Total batch size is kept constant between baseline and ghost; the ghost
  train batch is smaller by `val_batch_size`.
- Logging of dot products is optional; disable it to focus on compute overhead.
- Model is Llama3-like (pre-norm, RoPE, SwiGLU) but uses LayerNorm for
  ghost-engine compatibility.
"""

import argparse
import os
import statistics
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ghostEngines import GradDotProdEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ghost dot-product benchmark on a Llama-like Transformer LM")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run on (auto chooses cuda if available).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Total batch size per step (ghost uses train+val = batch-size).")
    parser.add_argument("--val-batch-size", type=int, default=8,
                        help="Validation batch size used for ghost dot products.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length.")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Vocabulary size.")
    parser.add_argument("--d-model", type=int, default=128, help="Model width.")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--mlp-ratio", type=int, default=4, help="MLP expansion ratio.")
    parser.add_argument("--rope-base", type=int, default=500000, help="RoPE base (theta).")
    parser.add_argument("--steps", type=int, default=10, help="Measured steps.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--log-dotprods", action="store_true",
                        help="Aggregate and log dot products (adds CPU transfers).")
    parser.add_argument("--check-correctness", action="store_true",
                        help="Validate ghost dot products against naive per-sample gradients.")
    parser.add_argument("--check-batch-size", type=int, default=16,
                        help="Total batch size for correctness check (default: min(4, batch_size)).")
    parser.add_argument("--check-val-batch-size", type=int, default=8,
                        help="Validation batch size for correctness check (default: min(1, val_batch_size)).")
    parser.add_argument("--check-grad-norms", action="store_true",
                        help="Validate logged train/val gradient norms against naive autograd.")
    parser.add_argument("--check-rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check.")
    parser.add_argument("--check-atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return requested


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lm_batch(batch_size: int, seq_len: int, vocab_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


def build_rope_cache(seq_len: int, head_dim: int, base: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype, device=x.device)
    sin = sin.to(dtype=x.dtype, device=x.device)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return x_rot.flatten(-2)


class LlamaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, rope_base: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        rope_cos, rope_sin = build_rope_cache(seq_len, self.head_dim, rope_base)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden_dim = mlp_ratio * d_model
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, seq_len: int, rope_base: int) -> None:
        super().__init__()
        # LayerNorm is used as a stand-in for RMSNorm for hook compatibility.
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = LlamaAttention(d_model, n_heads, seq_len, rope_base)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = LlamaMLP(d_model, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: int,
        rope_base: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [LlamaBlock(d_model, n_heads, mlp_ratio, seq_len, rope_base) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.seq_len:
            raise ValueError(f"input seq_len={seq_len} exceeds model seq_len={self.seq_len}")
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def run_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    warmup: int,
    engine: Optional[GradDotProdEngine] = None,
    val_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    log_dotprods: bool = False,
    track_memory: bool = False,
) -> Tuple[List[float], Optional[int]]:
    times: List[float] = []
    model.train()
    baseline_alloc = 0
    peak_bytes: Optional[int] = None

    for step, (x_train, y_train) in enumerate(batches):
        if step == warmup and device == "cuda" and track_memory:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_alloc = torch.cuda.memory_allocated()

        optimizer.zero_grad(set_to_none=True)

        if engine is not None:
            if val_batch is None:
                raise ValueError("val_batch is required when using ghost engine")
            x_val, y_val = val_batch
            x_forward = torch.cat([x_train, x_val], dim=0)
            y_forward = torch.cat([y_train, y_val], dim=0)
            if log_dotprods:
                engine.attach_train_batch(x_train, y_train, iter_num=step, batch_idx=step)
        else:
            x_forward, y_forward = x_train, y_train

        if device == "cuda" and step >= warmup:
            torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(x_forward)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_forward.reshape(-1))
        loss.backward()

        if engine is not None:
            engine.prepare_gradients()

        optimizer.step()

        if engine is not None:
            if log_dotprods:
                engine.aggregate_and_log()
            engine.clear_gradients()

        if device == "cuda" and step >= warmup:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if step >= warmup:
            times.append(elapsed)

    if device == "cuda" and track_memory:
        peak_bytes = max(torch.cuda.max_memory_allocated(), baseline_alloc)

    return times, peak_bytes


def summarize(times: List[float], tokens_per_step: int) -> dict:
    if not times:
        raise ValueError("No measured steps recorded; increase --steps or reduce --warmup.")
    mean_s = statistics.mean(times)
    median_s = statistics.median(times)
    throughput = tokens_per_step / mean_s
    return {
        "mean_ms": mean_s * 1000.0,
        "median_ms": median_s * 1000.0,
        "tokens_per_sec": throughput,
    }


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "N/A"
    return f"{num_bytes / (1024 ** 2):.1f} MiB"


def compute_ghost_dot_products(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    val_batch_size: int,
) -> torch.Tensor:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    engine = GradDotProdEngine(
        module=model,
        val_batch_size=val_batch_size,
        loss_reduction="mean",
        use_dummy_bias=False,
        dot_prod_save_path=None,
    )
    engine.attach(optimizer)
    engine.attach_train_batch(x_train, y_train, iter_num=0, batch_idx=0)

    optimizer.zero_grad(set_to_none=True)
    x_forward = torch.cat([x_train, x_val], dim=0)
    y_forward = torch.cat([y_train, y_val], dim=0)
    logits = model(x_forward)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_forward.reshape(-1))
    loss.backward()

    engine.aggregate_and_log()
    if not engine.dot_product_log:
        raise RuntimeError("Ghost engine did not log any dot products.")
    ghost_dot = engine.dot_product_log[-1]["dot_product"].clone()
    engine.dot_product_log.clear()
    engine.clear_gradients()
    engine.detach()
    return ghost_dot


def compute_ghost_grad_norms(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    val_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    engine = GradDotProdEngine(
        module=model,
        val_batch_size=val_batch_size,
        loss_reduction="mean",
        use_dummy_bias=False,
        dot_prod_save_path=None,
        log_grad_norms=True,
    )
    engine.attach(optimizer)
    engine.attach_train_batch(x_train, y_train, iter_num=0, batch_idx=0)

    optimizer.zero_grad(set_to_none=True)
    x_forward = torch.cat([x_train, x_val], dim=0)
    y_forward = torch.cat([y_train, y_val], dim=0)
    logits = model(x_forward)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_forward.reshape(-1))
    loss.backward()

    engine.aggregate_and_log()
    if not engine.dot_product_log:
        raise RuntimeError("Ghost engine did not log any gradient norms.")
    info = engine.dot_product_log[-1]
    if "train_grad_norm" not in info or "val_grad_norm" not in info:
        raise RuntimeError("Gradient norm entries missing from ghost engine log.")
    train_norm = info["train_grad_norm"].clone()
    val_norm = torch.tensor(info["val_grad_norm"])

    engine.dot_product_log.clear()
    engine.clear_gradients()
    engine.detach()
    return train_norm, val_norm


def compute_naive_dot_products(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    total_batch_size: int,
) -> torch.Tensor:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    scale = 1.0 / float(total_batch_size)

    val_sum = [torch.zeros_like(p, device=p.device) for p in params]
    for idx in range(x_val.size(0)):
        logits = model(x_val[idx:idx + 1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_val[idx:idx + 1].reshape(-1))
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        for i, g in enumerate(grads):
            val_sum[i].add_(g, alpha=scale)

    train_dots = []
    for idx in range(x_train.size(0)):
        logits = model(x_train[idx:idx + 1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_train[idx:idx + 1].reshape(-1))
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        dot = torch.tensor(0.0, device=x_train.device)
        for i, g in enumerate(grads):
            dot = dot + (g * scale * val_sum[i]).sum()
        train_dots.append(dot)

    return torch.stack(train_dots).detach().cpu()


def compute_naive_grad_norms(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    train_norms = []
    for idx in range(x_train.size(0)):
        logits = model(x_train[idx:idx + 1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_train[idx:idx + 1].reshape(-1))
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        norm_sq = torch.tensor(0.0, device=x_train.device)
        for g in grads:
            norm_sq = norm_sq + (g.float() ** 2).sum()
        train_norms.append(norm_sq.sqrt())

    val_sum = [torch.zeros_like(p, device=p.device) for p in params]
    for idx in range(x_val.size(0)):
        logits = model(x_val[idx:idx + 1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_val[idx:idx + 1].reshape(-1))
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        for i, g in enumerate(grads):
            val_sum[i].add_(g)

    val_norm_sq = torch.tensor(0.0, device=x_train.device)
    for g in val_sum:
        val_norm_sq = val_norm_sq + (g.float() ** 2).sum()

    return torch.stack(train_norms).detach().cpu(), val_norm_sq.sqrt().detach().cpu()


def run_correctness_check(args: argparse.Namespace, init_state: dict, device: str) -> None:
    check_total_bs = args.check_batch_size or min(4, args.batch_size)
    check_val_bs = args.check_val_batch_size or min(1, args.val_batch_size)
    if check_val_bs <= 0:
        raise ValueError("check_val_batch_size must be > 0")
    if check_val_bs >= check_total_bs:
        raise ValueError("check_val_batch_size must be smaller than check_batch_size")

    train_bs = check_total_bs - check_val_bs

    set_seed(args.seed)
    x_train, y_train = make_lm_batch(train_bs, args.seq_len, args.vocab_size, device)
    x_val, y_val = make_lm_batch(check_val_bs, args.seq_len, args.vocab_size, device)

    ghost_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)
    ghost_model.load_state_dict(init_state)
    ghost_dot = compute_ghost_dot_products(
        ghost_model, x_train, y_train, x_val, y_val, check_val_bs
    )

    naive_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)
    naive_model.load_state_dict(init_state)
    naive_dot = compute_naive_dot_products(
        naive_model, x_train, y_train, x_val, y_val, check_total_bs
    )

    if ghost_dot.shape != naive_dot.shape:
        raise ValueError(f"Shape mismatch: ghost={ghost_dot.shape} naive={naive_dot.shape}")

    diff = (ghost_dot - naive_dot).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (naive_dot.abs() + args.check_atol)).max().item()
    ok = torch.allclose(ghost_dot, naive_dot, rtol=args.check_rtol, atol=args.check_atol)

    print("\n=== Correctness Check ===")
    print(f"Batch size: total={check_total_bs} train={train_bs} val={check_val_bs}")
    print(f"allclose={ok} max_abs_diff={max_abs:.6g} max_rel_diff={max_rel:.6g}")
    if not ok:
        raise AssertionError("Ghost dot products do not match naive computation.")


def run_grad_norm_check(args: argparse.Namespace, init_state: dict, device: str) -> None:
    check_total_bs = args.check_batch_size or min(4, args.batch_size)
    check_val_bs = args.check_val_batch_size or min(1, args.val_batch_size)
    if check_val_bs <= 0:
        raise ValueError("check_val_batch_size must be > 0")
    if check_val_bs >= check_total_bs:
        raise ValueError("check_val_batch_size must be smaller than check_batch_size")

    train_bs = check_total_bs - check_val_bs

    set_seed(args.seed)
    x_train, y_train = make_lm_batch(train_bs, args.seq_len, args.vocab_size, device)
    x_val, y_val = make_lm_batch(check_val_bs, args.seq_len, args.vocab_size, device)

    ghost_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)
    ghost_model.load_state_dict(init_state)
    ghost_train_norm, ghost_val_norm = compute_ghost_grad_norms(
        ghost_model, x_train, y_train, x_val, y_val, check_val_bs
    )

    naive_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)
    naive_model.load_state_dict(init_state)
    naive_train_norm, naive_val_norm = compute_naive_grad_norms(
        naive_model, x_train, y_train, x_val, y_val
    )

    if ghost_train_norm.shape != naive_train_norm.shape:
        raise ValueError(f"Shape mismatch: ghost={ghost_train_norm.shape} naive={naive_train_norm.shape}")

    train_diff = (ghost_train_norm - naive_train_norm).abs()
    train_max_abs = train_diff.max().item()
    train_max_rel = (train_diff / (naive_train_norm.abs() + args.check_atol)).max().item()
    train_ok = torch.allclose(ghost_train_norm, naive_train_norm, rtol=args.check_rtol, atol=args.check_atol)

    val_diff = abs(float(ghost_val_norm) - float(naive_val_norm))
    val_den = float(abs(naive_val_norm) + args.check_atol)
    val_rel = val_diff / val_den if val_den != 0 else float("inf")
    val_ok = torch.isclose(
        torch.tensor(float(ghost_val_norm)),
        torch.tensor(float(naive_val_norm)),
        rtol=args.check_rtol,
        atol=args.check_atol,
    )

    print("\n=== Gradient Norm Check ===")
    print(f"Batch size: total={check_total_bs} train={train_bs} val={check_val_bs}")
    print(f"train allclose={train_ok} max_abs_diff={train_max_abs:.6g} max_rel_diff={train_max_rel:.6g}")
    print(f"val   allclose={bool(val_ok)} abs_diff={val_diff:.6g} rel_diff={val_rel:.6g}")
    if not (train_ok and val_ok):
        raise AssertionError("Ghost gradient norms do not match naive computation.")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.val_batch_size <= 0:
        raise ValueError("val_batch_size must be > 0")
    if args.val_batch_size >= args.batch_size:
        raise ValueError("val_batch_size must be smaller than batch_size")

    train_batch_size = args.batch_size - args.val_batch_size
    total_steps = args.steps + args.warmup

    set_seed(args.seed)

    init_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    )
    init_state = init_model.state_dict()
    del init_model

    # Pre-generate batches to avoid timing randomness from data creation.
    base_batches = [
        make_lm_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        for _ in range(total_steps)
    ]
    ghost_batches = [
        make_lm_batch(train_batch_size, args.seq_len, args.vocab_size, device)
        for _ in range(total_steps)
    ]
    val_batch = make_lm_batch(args.val_batch_size, args.seq_len, args.vocab_size, device)

    tokens_per_step = args.batch_size * args.seq_len

    print("\n=== Benchmark Configuration ===")
    print(f"Device: {device}")
    print(f"Model: Llama-like | d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"RoPE base: {args.rope_base}")
    print(f"Seq len: {args.seq_len}, Vocab: {args.vocab_size}")
    print(f"Batch size (total): {args.batch_size} | ghost train: {train_batch_size} | val: {args.val_batch_size}")
    print(f"Warmup: {args.warmup} steps | Measured: {args.steps} steps")
    print(f"Log dot-products: {args.log_dotprods}")

    print("\nRunning baseline (no ghost)...")
    base_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)
    base_model.load_state_dict(init_state)
    base_opt = torch.optim.AdamW(base_model.parameters(), lr=args.lr)
    base_times, base_peak_bytes = run_loop(
        model=base_model,
        optimizer=base_opt,
        batches=base_batches,
        device=device,
        warmup=args.warmup,
        track_memory=True,
    )

    print("Running ghost dot-product...")
    del base_model, base_opt
    if device == "cuda":
        torch.cuda.empty_cache()

    ghost_model = LlamaLM(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        rope_base=args.rope_base,
    ).to(device)

    ghost_model.load_state_dict(init_state)
    ghost_opt = torch.optim.AdamW(ghost_model.parameters(), lr=args.lr)
    ghost_engine = GradDotProdEngine(
        module=ghost_model,
        val_batch_size=args.val_batch_size,
        loss_reduction="mean",
        use_dummy_bias=True,
        dot_prod_save_path=None,
    )
    ghost_engine.attach(ghost_opt)
    ghost_times, ghost_peak_bytes = run_loop(
        model=ghost_model,
        optimizer=ghost_opt,
        batches=ghost_batches,
        device=device,
        warmup=args.warmup,
        engine=ghost_engine,
        val_batch=val_batch,
        log_dotprods=args.log_dotprods,
        track_memory=True,
    )

    base_stats = summarize(base_times, tokens_per_step)
    ghost_stats = summarize(ghost_times, tokens_per_step)

    slowdown = ghost_stats["mean_ms"] / base_stats["mean_ms"]

    print("\n=== Results (mean over measured steps) ===")
    print(
        f"Baseline: {base_stats['mean_ms']:.2f} ms/step | "
        f"{base_stats['tokens_per_sec']:.1f} tok/s | "
        f"peak_mem={format_bytes(base_peak_bytes)}"
    )
    print(
        f"Ghost:    {ghost_stats['mean_ms']:.2f} ms/step | "
        f"{ghost_stats['tokens_per_sec']:.1f} tok/s | "
        f"peak_mem={format_bytes(ghost_peak_bytes)}"
    )
    print(f"Slowdown: {slowdown:.2f}x")

    del base_batches, ghost_batches, val_batch
    del ghost_model, ghost_opt, ghost_engine
    if device == "cuda":
        torch.cuda.empty_cache()

    run_correctness_check(args, init_state, device)
    run_grad_norm_check(args, init_state, device)


if __name__ == "__main__":
    main()
