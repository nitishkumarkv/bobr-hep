"""BOBR binners subpackage.

This module exposes the binner implementations. Classes are provided using
the new lowercase names requested by the maintainer while preserving the
original uppercase names as aliases for backwards compatibility.
"""
from .base import BOBRBase
from .equidistant import equidistant
from .bobr_1d import bobr_1d
from .bobr_gmm import bobr_gmm

__all__ = ["BOBRBase", "equidistant", "bobr_1d", "bobr_gmm"]