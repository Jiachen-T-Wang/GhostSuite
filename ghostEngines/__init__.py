"""
Ghost Engines Package - Unified interface for gradient computation engines.

This package provides engines for computing gradient-based metrics during training,
such as gradient dot products, with minimal integration overhead.
"""

from .graddotprod_engine import GradDotProdEngine
from .gradProjection.gradproj_engine import GradProjLoraEngine, create_gradproj_engine
from .engine_manager import GhostEngineManager
from .engine_protocol import GhostEngine
from .selection import (
    SelectionPolicy, UpdateAll, TopK, BottomK, Threshold, NoSelection, greedy_selection,
    stochastic_greedy_selection,
)
from .selection_driver import online_selection_step

__all__ = [
    'GhostEngine', 'GradDotProdEngine', 'GradProjLoraEngine', 'create_gradproj_engine',
    'GhostEngineManager',
    'SelectionPolicy', 'UpdateAll', 'TopK', 'BottomK', 'Threshold', 'NoSelection',
    'greedy_selection', 'stochastic_greedy_selection', 'online_selection_step',
]
