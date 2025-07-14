# Developing version of the GhostSuite codebase.

## Codebase revision notes
- 6/26: There is an issue when upgrading transformers to 4.50. Now falling back to 4.36.2
- 7/10: Removed gradient clipping and I realize that gradient clipping should happen inside the engine. **TODO:** if no more elegant way to handle gradient clipping then just manually handle gradient clipping inside Engine. 
- 7/10: The dropout rates need to be set to 0 at the moment as per-sample activation dropouts are dependent on batch size, but it's okey. Can try customized dropout later if that does not increase runtime.


# "Ghost" Suites for fast gradient information calculation

## Introduction
This repository demonstrates how to compute gradient based metrics, such as In-Run Data Shapley values, while training language models. The "ghost" engines compute gradient dot product trick so that the validation gradients can be compared against training samples without fully materialising intermediate gradients. The code is built around the
`transformers` library and currently targets GPT-2 style architectures.

## Installation
```bash
pip install -r requirements.txt
```

## Getting started
Prepare dataset binaries as expected by `dataloader.py` and adjust paths in `config_file.py` if necessary.  Training can then be launched with
```bash
python main.py --method "GradDotProd" --batch_size 16 --val_batch_size 1
```
During training the engine logs gradient dot products and saves results under the directory specified by `TrainingConfig.result_folder`.

When integrating `GradDotProdEngine` manually the update step now looks like:

```python
loss.backward()
engine.prepare_gradients()
# Can add any additional gradient operations such as gradient clipping here. 
optimizer.step()
engine.aggregate_and_log()
engine.clear_gradients()
```
