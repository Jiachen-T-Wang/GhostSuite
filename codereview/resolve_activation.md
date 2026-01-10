# resolve_activation review (autograd_grad_sample_dotprod.py)

## What the saved-tensor log shows
- `tok_embeddings`: one non-param tensor, shape `(16, 32)` int64 (token ids).
- RMSNorm blocks (`layers.*.attention_norm`, `layers.*.ffn_norm`, `norm`):
  - input activation `(16, 32, 16)` fp32
  - weight parameter `(16,)` fp32
  - inv_rms `(16, 32, 1)` fp32
- Linear blocks (`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`, `output`):
  - weight.T view `(in, out)` bf16
  - flattened input activation `(batch*seq, in)` bf16
- SDPA/rotary/cross-entropy tensors are captured under scope `<none>` and are not part of any supported layer’s scope.

## Correctness check against the log
- Embedding: input shape `(16, 32)` matches `input_shape`, so `resolve_activation` selects the token-id activation correctly.
- RMSNorm: input shape `(16, 32, 16)` has a unique match in the per-layer capture list, so the function selects the input activation, not the weight or `inv_rms`.
- Linear/Conv1D-style layers: input shape `(16, 32, d)` does not match any saved tensor, but `flat_shape = (512, d)` does. `resolve_activation` selects the flattened input, and `_prepare_sample_grad_or_dotprod` reshapes it back to `(16, 32, d)` before the dot-product/grad logic. This lines up with the observed `(512, d)` saved tensors in the log.
- No evidence in the log that the fallback to `captured_all` is used for any supported layer; all supported layers have scoped captures.

Overall, for the recorded TorchTitan run, the selection logic appears consistent with the saved tensors and should return the intended activation for each supported layer.

## Potential correctness risks (not exercised by this log)
- Shared input tensor objects across layers: `_used_ids` is global. If two supported layers save the exact same tensor object (e.g., two `nn.Linear` calls on the same 2D input without a view/reshape), the first call will mark it used and the second call will skip it, potentially returning `None` or the wrong tensor.
- Reused modules (weight sharing) with identical input shapes: `resolve_activation` does not track call order; it picks the first matching tensor in the per-layer list. If the same module is invoked multiple times in a single forward pass with the same shape, backward order may not align with this choice.
- Param filtering relies on `tensor._base`: if autograd saves a parameter copy (not a view), it will not be filtered out. In a degenerate case where a parameter and the activation share the same shape (e.g., square linear with batch size equal to out_features), the wrong tensor could be selected.
- The `captured[name]` vs `captured_all` fallback is only taken when the per-layer list is empty. If a layer’s per-scope list contains only parameter tensors (no non-params), `resolve_activation` returns `None` rather than falling back to global captures. This can fail if scope tracking misses the activation but still captures weights.
- Ambiguous same-shape intermediates: if a layer saves multiple non-param tensors of identical shape, the selection will choose the first non-leaf tensor, which may not be the true input activation.

## Simplification opportunities
- Factor the repeated shape-matching logic into a small helper:
  - `pick_by_shape(non_param, shape)` that prefers non-leaf tensors and returns the first match.
- Iterate over candidate shapes in a loop (`input_shape`, then `flat_shape`) instead of duplicating the same “match + prefer non-leaf + mark used” block.
- Centralize the “mark used and return” step into a small helper to avoid repeated `self._used_ids.add(id(t))` blocks.
- Clarify the capture-pool fallback with an explicit `if not capture_pool: capture_pool = self._captured_all` (same behavior, easier to read).

These changes would reduce duplication while keeping the existing behavior intact.
