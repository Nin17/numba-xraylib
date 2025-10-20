"""Utility functions for numba-xraylib."""

import inspect
import warnings
from collections.abc import Callable, Sequence
from ctypes.util import find_library
from itertools import chain, repeat
from pathlib import Path
from types import EllipsisType

import _xraylib
import xraylib
import xraylib_np
from llvmlite.binding import load_library_permanently
from numba import errors, extending, types
from numpy import byte, zeros
from numpy.typing import NDArray

from .config import config

ND_ERROR = "N-dimensional arrays (N > 1) are not allowed if config.allow_nd is False"


def get_extension_path(lib_name: str) -> str | None:
    """Get the path to the library with the given name in the parent directory.

    Parameters
    ----------
    lib_name : str
        The name of the library to search for.

    Returns
    -------
    str
        The path to the library.

    """
    search_path = Path(__file__).parent.parent
    ext_path = f"{lib_name}*.*"
    matches = filter(lambda x: not x.name.endswith(".pc"), search_path.rglob(ext_path))
    try:
        return str(next(matches))
    except StopIteration:
        return None


def _init() -> None:
    _path = get_extension_path("libxrl")
    if _path is None:
        msg = "Could not find libxrl library bundled with numba_xraylib"
        warnings.warn(msg, stacklevel=2)
        _path = find_library("xrl")
    load_library_permanently(_path)
    from . import overload_xraylib, overload_xraylib_np  # noqa: PLC0415 F401


def _get_name() -> str:
    return inspect.stack()[2].function.removeprefix("_").removesuffix("_np")


def external_fcn(*, ns: int = 0, ni: int = 0, nf: int = 0) -> types.ExternalFunction:
    """External function with ns string args, ni integer args & nf double args.

    fcn(*[string]*ns, *[integer]*ni, *[double]*nf)

    Parameters
    ----------
    ns   : int, optional
        the number of string args, by default 0
    ni   : int, optional
        the number of integer args, by default 0
    nf   : int, optional
        the number of double args, by default 0

    Returns
    -------
    types.ExternalFunction
        the external function

    """
    name = _get_name()
    ptr_char = types.CPointer(types.char)
    argtypes = (*[ptr_char] * ns, *[types.int32] * ni, *[types.float64] * nf)
    sig = types.float64(*argtypes, types.voidptr)
    return types.ExternalFunction(name, sig)


def check_types_xrl(args: Sequence, *, ns: int = 0, ni: int = 0, nf: int = 0) -> None:
    """Check the arguments have the correct type for the signature.

    Function with signature fcn(*[string]*ns, *[integer]*ni, *[double]*nf)

    Parameters
    ----------
    args : Sequence
        arguments to check
    ns : int, optional
        number of string arguments, by default 0
    ni : int, optional
        number of integer arguments, by default 0
    nf : int, optional
        number of double arguments, by default 0

    Raises
    ------
    errors.NumbaTypeError
        If the argument types don't match the signature

    """
    msg = "Expected {0} got {1}"
    for i in range(ns):
        if not isinstance(args[i], types.UnicodeType):
            raise errors.NumbaTypeError(msg.format(types.UnicodeType, args[i]))
    for i in range(ns, ns + ni):
        if not isinstance(args[i], types.Integer):
            raise errors.NumbaTypeError(msg.format(types.Integer, args[i]))
    for i in range(ns + ni, ns + ni + nf):
        if args[i] is not types.float64:
            raise errors.NumbaTypeError(msg.format(types.float64, args[i]))


def check_types_xrl_np(args: Sequence, *, ni: int = 0, nf: int = 0) -> None:
    """Check the arguments have the correct types for the signature.

    Function with signature fcn(*[integer]*ni, *[double]*nf)

    Parameters
    ----------
    args : Sequence
        arguments to check
    ni : int, optional
        number of integer arguments, by default 0
    nf : int, optional
        number of double arguments, by default 0

    Raises
    ------
    errors.NumbaTypeError
        If the argument types don't match the signature

    """
    msg = "Expected array({0}, ...) got {1}"
    for i in range(ni):
        if not isinstance(args[i].dtype, types.Integer):
            raise errors.NumbaTypeError(msg.format("int32|int64", args[i]))
    for i in range(ni, ni + nf):
        if args[i].dtype is not types.float64:
            raise errors.NumbaTypeError(msg.format("float64", args[i]))


def check_ndim(*args: types.Array) -> None:
    """Check that the arguments are 1d unless config.allow_nd is True.

    Raises
    ------
    errors.NumbaValueError
        If n > 1 dimensional arrays are passed and config.allow_nd = False

    """
    if not config.allow_nd and any(arg.ndim > 1 for arg in args):
        raise errors.NumbaValueError(ND_ERROR)


def indices(*args: types.Array) -> list[tuple[None | EllipsisType, ...]]:
    """Generate indices to broadcast arrays whilst concatenating their shapes.

    Returns
    -------
    list[tuple[None | EllipsisType, ...]]
        indices

    """
    return [
        tuple(
            chain.from_iterable(
                [repeat(None, n.ndim) if m != i else [...] for m, n in enumerate(args)],
            ),
        )
        for i, _ in enumerate(args)
    ]


@extending.register_jitable
def convert_str(s: str) -> NDArray[byte]:
    """Convert a string to a zero terminated byte array.

    Parameters
    ----------
    s : str
        string

    Returns
    -------
    NDArray[byte]
        byte array terminated by 0.

    """
    len_s = len(s)
    out = zeros(len(s) + 1, dtype=byte)
    for i in range(len_s):
        out[i] = ord(s[i])
    return out


def overload_xrl(fcn: Callable) -> None:
    """Overload a function in the xraylib namespace.

    Parameters
    ----------
    fcn : Callable
        The function that implements the overload.
        Should have the same name as the xraylib function to be overloaded with an
        underscore prefixed.

    """
    fname = fcn.__name__.removeprefix("_")
    jit_options = config.xrl.get(fname, {})

    _xrl_fcn = getattr(_xraylib, fname)
    extending.overload(_xrl_fcn, jit_options)(fcn)
    extending.register_jitable(_xrl_fcn)

    xrl_fcn = getattr(xraylib, fname)
    extending.overload(xrl_fcn, jit_options)(fcn)
    extending.register_jitable(xrl_fcn)


def overload_xrl_np(fcn: Callable) -> None:
    """Overload a function in the xraylib_np namespace.

    Parameters
    ----------
    fcn : Callable
        The function that implements the overload.
        Should have the same name as the xraylib_np function to be overloaded with an
        underscore prefixed and "_np" suffixed.

    """
    fname = fcn.__name__.removeprefix("_").removesuffix("_np")
    jit_options = config.xrl_np.get(fname, {})
    extending.overload(getattr(xraylib_np, fname), jit_options)(fcn)
