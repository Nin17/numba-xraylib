"""Configure jit options for numba-xraylib."""

from __future__ import annotations

__all__ = ("config",)

import sys
from typing import Any

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

from pathlib import Path

with (Path(__file__).parent / "config.toml").open("rb") as f:
    toml_config = tomllib.load(f)


class Config:
    """Configuration for jit options."""

    __slots__ = ("allow_nd", "xrl", "xrl_np")

    def __init__(
        self: Config,
        **toml_config: dict[str, dict[str, Any]],
    ) -> None:
        self.xrl = toml_config["xrl"]
        self.xrl_np = toml_config["xrl_np"]
        self.allow_nd = False


config = Config(**toml_config)
