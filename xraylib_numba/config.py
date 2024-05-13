"""Configure jit options for xraylib_numba."""

from __future__ import annotations

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

    __slots__ = ["xrl", "xrl_np", "allow_nd"]

    def __init__(  # noqa: D107
        self: Config,
        **toml_config: dict[str, dict[str, Any]],
    ) -> None:
        self.xrl = toml_config["xrl"]
        self.xrl_np = toml_config["xrl_np"]
        self.allow_nd = False


config = Config(**toml_config)
