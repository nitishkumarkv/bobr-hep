"""Top-level BOBR API shim.

This module re-exports the classes from the `binners` subpackage so users
can continue importing from `bobr_hep.bobr` as before.
"""

from .binners.base import bobr_base
from .binners.equidistant import equidistant
from .binners.bobr_1d import bobr_1d
from .binners.bobr_gmm import bobr_gmm

__all__ = ["bobr_base", "equidistant", "bobr_1d", "bobr_gmm"]
