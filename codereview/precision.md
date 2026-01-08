# Mixed precision dtype analysis for `run_train_with_ghost.sh`

## Context: what mixed precision mode is active

- The debug config sets `training.dtype = "float32"` and `training.mixed_precision_param = "bfloat16"` with no FSDP/TP/PP (`parallelism.*_degree = 1`). In this setup, `maybe_enable_amp(...)` chooses AMP autocast (bfloat16) rather than FSDP mixed precision. Parameters stay in fp32; autocast selectively runs compute-heavy ops (e.g., `nn.Linear`, matmuls) in bf16 while keeping numerically sensitive ops (e.g., norms, softmax) in fp32.
  - Config defaults: `tests/torchtitan/torchtitan/config/job_config.py`
  - AMP path: `tests/torchtitan/torchtitan/distributed/utils.py`
  - AMP is used in `train_with_ghost.py` around the forward + loss.

## What the log lines represent

- The hook prints `A dtype` and `B dtype` inside `_compute_linear_dot_product`. Here:
  - `A` = the layer input activation captured by the forward hook (`layer.activations`).
  - `B` = the layer output gradient (`grad_output[0]`) captured by the full backward hook.
  - If `A` and `B` differ, `_prepare_sample_grad_or_dotprod` promotes both to a common dtype before calling `_compute_linear_dot_product`, so the printed dtype may be the promoted dtype, not the original.
  - `_compute_linear_dot_product` immediately casts both tensors to bf16 for its dot-product computation, regardless of the printed dtype.
  - Hook path: `ghostEngines/autograd_grad_sample_dotprod.py`, `ghostEngines/supported_layers_grad_samplers_dotprod.py`

## Activation and gradient dtypes in the log

### Attention path

- **`Attention forward: x dtype: torch.float32`**
  - `x` is the output of `self.attention_norm(x)` from `TransformerBlock`. RMSNorm runs in fp32 under autocast, and the residual stream is fp32 (float32 + bf16 promotes to float32), so the attention input stays fp32.
  - Code: `tests/torchtitan/torchtitan/models/llama3/model/model.py` (`TransformerBlock.forward`, `Attention.forward`)

- **`Attention forward: xq/xk/xv dtype: torch.bfloat16`**
  - `wq/wk/wv` are `nn.Linear` ops. Under autocast they execute in bf16, so their outputs are bf16.
  - Code: `tests/torchtitan/torchtitan/models/llama3/model/model.py` (`Attention.forward`)

- **`attention after rotary: xq/xk dtype: torch.bfloat16`**
  - `apply_rotary_emb(...)` explicitly converts `xq/xk` to fp32 for the complex math, then casts back to the original dtype via `type_as(xq/xk)`. So the *returned* tensors are bf16 even though the internal math is fp32.
  - Code: `tests/torchtitan/torchtitan/models/llama3/model/model.py` (`apply_rotary_emb`)

- **`[hook] ... attention.wq/wk/wv: A dtype: torch.float32, B dtype: torch.float32`**
  - `A` is fp32 because it is the attention-norm output. `B` prints as fp32 because the hook promotes `A` and `B` to a common type; even if the raw backprop were bf16, the promoted dtype is fp32 when `A` is fp32.
  - This is consistent with the residual stream being fp32 and with AMP promotion in the hook.
  - Hook code: `ghostEngines/autograd_grad_sample_dotprod.py`

- **`[hook] ... attention.wo: A dtype: torch.bfloat16, B dtype: torch.bfloat16`**
  - The attention kernel consumes bf16 q/k/v and returns bf16 output, so the input to `wo` is bf16 and its grad_output also stays bf16.

### Feed-forward path

- **`[hook] ... feed_forward.w1/w3: A dtype: torch.float32, B dtype: torch.float32`**
  - `A` is fp32 because `ffn_norm(h)` runs in fp32 and feeds both `w1` and `w3`.
  - The hook prints fp32 because of dtype promotion with fp32 activations; the raw backprop may have been bf16, but it is promoted to fp32 before `_compute_linear_dot_product` runs.

- **`[hook] ... feed_forward.w2: A dtype: torch.bfloat16, B dtype: torch.bfloat16`**
  - `w2` takes the bf16 intermediate `silu(w1(x)) * w3(x)` under autocast, so its input activation and its grad_output remain bf16.

### Output head

- **`[hook] _compute_linear_dot_product for output: A dtype: torch.float32, B dtype: torch.float32`**
  - The final RMSNorm keeps the residual stream in fp32, so the output projection receives fp32 activations.
  - The loss explicitly casts logits to fp32 via `pred.flatten(...).float()`, so the gradient w.r.t. logits is fp32, matching the hook print.
  - Code: `tests/torchtitan/torchtitan/models/llama3/model/model.py` (final `norm`, `output`), loss in `tests/torchtitan/torchtitan/components/loss.py`.

## Takeaway

- The mixed precision behavior in this run is: fp32 residual stream and norms; bf16 compute for linear/matmul-heavy blocks; rotary uses fp32 internally but returns bf16 tensors. The hook outputs reflect these choices plus an extra fp32 promotion step in the ghost hook before dot-product computation.
