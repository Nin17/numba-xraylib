# numba-xraylib

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI - Version](https://img.shields.io/pypi/v/numba-xraylib?color=%23785EF0)](https://pypi.org/project/numba-xraylib/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FNin17%2Fnumba-xraylib%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&color=%23785EF0)](https://pypi.org/project/numba-xraylib/)
[![GitHub License](https://img.shields.io/github/license/Nin17/numba-xraylib?color=%23785EF0)](https://choosealicense.com/licenses/mit/)

Use [xraylib](https://github.com/tschoonj/xraylib/tree/master) in [numba](https://numba.pydata.org) nopython functions.

## Installation

```shell
pip install numba-xraylib
```

## Usage

Simply install `numba-xraylib` in your environment to use `xraylib` and `xraylib_np` in nopython mode:

```python
import xraylib
import xraylib_np
from numba import njit
import numpy as np

@njit
def AtomicWeight(Z):
    return xraylib.AtomicWeight(Z), xraylib_np.AtomicWeight(np.array([Z]))

print(AtomicWeight(1))  # (1.01, array([1.01]))
```

Currently, functions that have non-numeric returns, or take c structs as arguments are unsupported.
If you know how to pass strings from numba to c and vice-versa please let me know.

## Documentation

See the [docs](https://nin17.github.io/numba-xraylib/).
