# Code Review: GhostSuite

## Findings
- **Critical** `ghostEngines/graddotprod_engine.py:64-73` `GradDotProdEngine` forcibly sets `requires_grad` on **all** parameters (all True when `use_dummy_bias=False`, all False when `use_dummy_bias=True`). The original flags are never restored in `detach`, so frozen weights can be unintentionally unfrozen or the model can remain permanently frozen after detaching the engine.
- **Major** `ghostEngines/engine_manager.py:216-221` `GhostEngineManager.reattach_after_evaluation()` always calls `engine.attach(self.optimizer)`. `GradProjLoraEngine.attach()` takes no optimizer, so this path raises `TypeError` when eval toggling is used with GradProj.
- **Major** `ghostEngines/gradProjection/supported_layers_gradproj.py:11-57` + `ghostEngines/gradProjection/autograd_gradproj.py:256-275` Conv1d is treated as supported and `include_conv2d=True` enables Conv2d selection, but hook creation raises `NotImplementedError` for both. This yields runtime failures for matching patterns or configs that explicitly enable Conv2d.
- **Major** `ghostEngines/engine_manager.py:189-196` Dict input concatenation is hardcoded to `input_ids`, `pixel_values`, `attention_mask`. Any other required keys (e.g., `token_type_ids`, `position_ids`, `labels`, `image_sizes`) are dropped or cause `KeyError`, breaking many HF models or silently changing behavior.
- **Medium** `ghostEngines/autograd_grad_sample_dotprod.py:51-52` uses exact `type(layer)` matching; subclasses (e.g., LoRA/PEFT linear layers, quantized linear wrappers) are skipped. This leads to missing `train_grad` attributes and runtime errors in `GradDotProdEngine.prepare_gradients()` or partial coverage without warning.
- **Minor** `ghostEngines/autograd_grad_sample_dotprod.py:58-68` `_apply_train_grad` is called without forwarding `loss_reduction`, so bias gradients always assume `'mean'`. If `loss_reduction='sum'` is used, dot-product scaling and bias grads become inconsistent.

## Questions / Assumptions
- Is `GradDotProdEngine` intended to be used directly with `use_dummy_bias=False`? If so, the `requires_grad` mutation/restoration behavior needs to be fixed; if not, it should still restore flags on `detach` to avoid surprising state.
- Are Conv1d/Conv2d projections supposed to be supported? If not, the selection helpers should exclude them and the CLI flags should be clarified.

## Testing Gaps
- No automated tests cover `detach/reattach` flows, dictionary input concatenation, or subclassed layers. These are the highest-risk integration surfaces.
