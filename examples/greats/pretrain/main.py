#!/usr/bin/env python3
"""Entry point for the GREATS online-batch-selection pretraining example.

Reuses the shared/ model + data utilities under examples/lm/ and the GradDotProd
ghost engine for per-sample scoring.
"""

import os
import sys

# Put examples/lm/ (for `shared`), the repo root (for `ghostEngines`), and this
# directory (for config_file / training_loop) on the path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)
for _p in (_THIS_DIR, os.path.join(_EXAMPLES_DIR, "lm"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_file import parse_arguments, TrainingConfig
from shared.training_utils import (
    setup_distributed,
    setup_torch_backend,
    print_training_info,
    setup_data_functions,
    load_dataset_main,
)
from shared.model_setup import setup_model_and_optimizer
from shared.utils import set_seed
from training_loop import GreatsTrainer


def main():
    args = parse_arguments()
    config = TrainingConfig(args)

    ddp_info = setup_distributed()
    set_seed(config.seed + ddp_info["seed_offset"])
    ctx = setup_torch_backend(config)
    print_training_info(config)

    model, optimizer, scaler = setup_model_and_optimizer(
        config, ddp_info["device"], ddp_info
    )

    dataset = load_dataset_main(args.train_set, args.val_set)
    get_batch_fn, get_val_batch_fn = setup_data_functions(
        dataset, config, ddp_info["device"], ddp_info=ddp_info
    )

    trainer = GreatsTrainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        ddp_info=ddp_info,
        get_batch_fn=get_batch_fn,
        get_val_batch_fn=get_val_batch_fn,
        ctx=ctx,
    )
    trainer.run_training()


if __name__ == "__main__":
    main()
