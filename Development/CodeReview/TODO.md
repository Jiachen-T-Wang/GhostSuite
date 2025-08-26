# Issue

API inconsistency: `average_grad` parameter is unused.
  - Files: `graddotprod_engine.py`, `autograd_grad_sample_dotprod.py`, samplers
  - Issue: Engine initializer accepts `average_grad=True` but samplers always compute average gradients. The flag is documented but has no effect.
  - Fix: Remove this parameter throughout the entire codebase. Make sure the code revision passes tests in 'Test/' and 'train.sh' does not change `./Scripts/train.sh --batch_size 2` does not change anything. 


# Revision
Finished