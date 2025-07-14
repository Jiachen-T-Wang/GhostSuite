# "Ghost" Suites for fast gradient information calculation

## Introduction
This repository demonstrates how to compute gradient based metrics, such as [In-Run Data Shapley](https://openreview.net/pdf?id=HD6bWcj87Y), while training language models. The "ghost" engines currently support first-order In-Run Data Shapley which computes the gradient dot-product between validation loss and each individual training sample within a batch without fully materialising intermediate gradients. The code is built around the `transformers` library and currently targets GPT-2 style architectures.

## Installation
```bash
pip install -r requirements.txt
```

## Quick start
Prepare dataset binaries as expected by `dataloader.py` and adjust paths in `config_file.py` if necessary. Training can then be launched with
```bash
python main.py --method "GradDotProd" --batch_size 16 --val_batch_size 1
```
During training the engine logs gradient dot products and saves results under the directory specified by `TrainingConfig.result_folder`.

When integrating `GradDotProdEngine` with your own training loop, you can first wrap your model with `GradDotProdEngine` and the update step now looks like:
```python
loss.backward()
engine.prepare_gradients()
# Can add any additional gradient operations, such as gradient clipping here. 
optimizer.step()
engine.aggregate_and_log()
engine.clear_gradients()
```
