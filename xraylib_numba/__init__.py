"""_summary_."""

__version__ = "0.1.0"

# from . import xraylib_overloads
from .config import config


def _init() -> None:
    from . import xraylib_overloads


# __all__ = ["xraylib_overloads"]

__all__ = ["config", "xraylib_overloads"]
