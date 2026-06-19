# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import os
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]

# Phase 2 (ghost torch.compile): F.cross_entropy / F.nll_loss compute the mean over the
# *count of non-ignored tokens*, a data-dependent scalar that Inductor codegens via
# aten::_local_scalar_dense and crashes the compiled backward with "found type 'int'". With
# GHOST_COMPILE_LOSS=1 we swap in a manual log_softmax + gather whose normalizer is a
# python-float constant (1/N), so the compiled graph has no data-dependent scalar. The llama3
# dataloader does not emit ignore_index (-100) tokens, so this is numerically equal to the
# default mean cross-entropy. Default path (flag off) is untouched.
_COMPILE_LOSS = os.getenv("GHOST_COMPILE_LOSS", "0") == "1"


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def compile_friendly_cross_entropy_loss(
    pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Manual log_softmax + gather cross-entropy with a constant (1/N) normalizer.

    Numerically equal to ``cross_entropy_loss`` when no labels are ignore_index, but emits no
    data-dependent scalar (the token count), so the compiled backward avoids the
    ``aten::_local_scalar_dense ... found type 'int'`` Inductor error. See _COMPILE_LOSS.
    """
    logits = pred.flatten(0, 1).float()
    target = labels.flatten(0, 1)
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # Constant python-float normalizer instead of .mean() (which divides by a traced count).
    return nll.sum() * (1.0 / nll.shape[0])


def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = compile_friendly_cross_entropy_loss if _COMPILE_LOSS else cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn


class RescaleAccumulatedLoss:
    def __init__(self, unwrapped_loss_fn, accumulation_steps):
        self.unwrapped_loss_fn = unwrapped_loss_fn
        self.accumulation_steps = accumulation_steps
        self.skip_rescale = False

        # Copy over attributes from the original function, but don't
        # copy the dict, which interferes with nested wrapping.
        functools.update_wrapper(self, unwrapped_loss_fn, updated=tuple())

    def __call__(self, *args, **kwargs):
        loss = self.unwrapped_loss_fn(*args, **kwargs)
        if self.skip_rescale:
            return loss
        # Phase 2: under the compile-loss flag with a single accumulation step the divide is a
        # no-op (loss / 1) but still emits a scalar op that can trip compiled codegen; skip it.
        if _COMPILE_LOSS and self.accumulation_steps == 1:
            return loss
        return loss / self.accumulation_steps

    @contextlib.contextmanager
    def no_rescale(self):
        """Context manager for disabling rescaling"""
        previous = self.skip_rescale
        self.skip_rescale = True
        try:
            yield
        finally:
            self.skip_rescale = previous


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """
    return RescaleAccumulatedLoss(unwrapped_loss_fn, accumulation_steps)


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function for Transformer models training."""
    return torch.nn.functional.mse_loss(pred.float(), labels.float().detach())


def build_mse_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn
