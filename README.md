# Xraylib_numba

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](
https://github.com/psf/black)

Use [xraylib](https://github.com/tschoonj/xraylib/tree/master) in [numba](https://numba.pydata.org) nopython functions.

## Installation

```text
pip install xraylib_numba
```

## Usage

Simply install `xraylib_numba` in your environment to use `xraylib` and `xraylib_np` in nopython mode:

```python
import xraylib
import xraylib_np
from numba import njit
import numpy as np

# %pip install xraylib_numba

@njit
def AtomicWeight(Z):
    return xraylib.AtomicWeight(Z), xraylib_np.AtomicWeight(np.array([Z]))

print(AtomicWeight(1))  # (1.01, array([1.01]))
```

Currently, functions that have non-numeric arguments or returns are unsupported.
If you know how to pass strings from numba to c please let me know.
