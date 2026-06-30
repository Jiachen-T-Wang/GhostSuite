"""Gradient-projection engine subpackage (LoRA-style per-sample projected gradients)."""

from .gradproj_engine import GradProjLoraEngine, create_gradproj_engine

__all__ = ['GradProjLoraEngine', 'create_gradproj_engine']
