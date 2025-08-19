"""
Ghost Engines Package - Unified interface for gradient computation engines.

This package provides engines for computing gradient-based metrics during training,
such as gradient dot products, with minimal integration overhead.
"""

from .graddotprod_engine import GradDotProdEngine
from .engine_manager import GhostEngineManager

__all__ = ['GradDotProdEngine', 'GhostEngineManager']
