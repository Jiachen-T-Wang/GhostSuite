# Tests

## Ghost Dot-Product Benchmark

Script: `tests/benchmark_ghost_dotprod.py`

This benchmark compares training throughput and peak GPU memory for a Llama-like
Transformer LM with and without Ghost dot-product computation. It also supports
an optional correctness check against a naive per-sample gradient implementation.

### Quick start

```bash
python tests/benchmark_ghost_dotprod.py --device auto
```

### Common options

- `--device {auto,cpu,cuda}`: Select device (default: auto).
- `--batch-size N`: Total batch size per step (ghost uses train+val = N).
- `--val-batch-size N`: Validation batch size used by Ghost.
- `--steps N` and `--warmup N`: Measured and warmup steps.
- `--log-dotprods`: Aggregate and log dot products (adds CPU transfer overhead).
- `--rope-base N`: RoPE base (theta) for the Llama-like model.

Example:

```bash
python tests/benchmark_ghost_dotprod.py --device cuda --batch-size 16 --val-batch-size 2 --steps 50
```

### Correctness check

Enable the correctness check to validate Ghost dot products against a naive
per-sample gradient dot-product computation. This runs after the benchmark.

```bash
python tests/benchmark_ghost_dotprod.py --device cuda --check-correctness
```

You can control the check size and tolerances:

```bash
python tests/benchmark_ghost_dotprod.py \
  --check-correctness \
  --check-batch-size 4 \
  --check-val-batch-size 1 \
  --check-rtol 1e-2 \
  --check-atol 1e-2
```

Notes:
- The correctness check is intentionally small since naive per-sample gradients
  are expensive.
- If you see small mismatches due to reduced precision in the Ghost path,
  relax tolerances with `--check-rtol`/`--check-atol`.

### Model assumptions

The benchmark uses a Llama-like block with:
- pre-norm transformer blocks,
- RoPE positional encoding,
- SwiGLU MLPs,
- `LayerNorm` in place of RMSNorm for Ghost engine compatibility.

This keeps the architecture close to modern Llama3-style models while staying
within the supported layer types for Ghost dot-product hooks.
