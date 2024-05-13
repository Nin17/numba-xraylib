"""_summary_."""

__version__ = "0.1.0"

from .config import config


def _init() -> None:
    from . import xraylib_overloads  # noqa: F401


__all__ = ["config", "xraylib_overloads"]
