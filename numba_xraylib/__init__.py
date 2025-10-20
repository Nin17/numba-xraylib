"""numba-xraylib: Numba overloads of xraylib functions."""

__version__ = "0.2.1"

__all__ = ["_init", "config"]

from .config import config
from .utils import _init
