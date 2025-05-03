"""Overloads for xraylib functions to work with numba."""

from __future__ import annotations

import inspect
from ctypes.util import find_library
from itertools import chain, repeat
from typing import TYPE_CHECKING

import _xraylib
import xraylib
import xraylib_np
from llvmlite import binding
from numba import errors, extending, types, vectorize
from numpy import array, broadcast_to, byte, int32, zeros

from .config import config

if TYPE_CHECKING:
    import sys
    from collections.abc import Sequence

    from numpy.typing import Array

    if sys.version_info >= (3, 10):
        from types import EllipsisType
    else:
        EllipsisType = "builtins.Ellipsis"

binding.load_library_permanently(find_library("xrl"))

# Error messages
Z_OUT_OF_RANGE = "Z out of range"
NEGATIVE_ENERGY = "Energy must be strictly positive"
NEGATIVE_DENSITY = "Density must be strictly positive"
NEGATIVE_Q = "q must be positive"
NEGATIVE_PZ = "pz must be positive"
INVALID_SHELL = "Invalid shell for this atomic number"
INVALID_LINE = "Invalid line for this atomic number"
INVALID_CK = "Invalid Coster-Kronig transition for this atomic number"
INVALID_AUGER = "Invalid Auger transition macro for this atomic number"
UNKNOWN_SHELL = "Unknown shell macro provided"
UNKNOWN_LINE = "Unknown line macro provided"
UNKNOWN_CK = "Unknown Coster-Kronig transition macro provided"
UNKNOWN_AUGER = "Unknown Auger transition macro provided"
NEGATIVE_PZ = "pz must be strictly positive"
SPLINE_EXTRAPOLATION = "Spline extrapolation is not allowed"
UNAVALIABLE_PHOTO_CS = (
    "Photoionization cross section unavailable for atomic number and energy"
)

ND_ERROR = "N-dimensional arrays (N > 1) are not allowed if config.allow_nd is False"


def _check_ndim(*args: types.Array) -> None:
    if not config.allow_nd and any(arg.ndim > 1 for arg in args):
        raise errors.NumbaValueError(ND_ERROR)


def _indices(*args: types.Array) -> list[tuple[None | EllipsisType, ...]]:
    return [
        tuple(
            chain.from_iterable(
                [repeat(None, n.ndim) if m != i else [...] for m, n in enumerate(args)],
            ),
        )
        for i, _ in enumerate(args)
    ]


@extending.register_jitable
def _convert_str(s: str) -> Array[byte]:
    len_s = len(s)
    out = zeros(len(s) + 1, dtype=byte)
    for i in range(len_s):
        out[i] = ord(s[i])
    return out


def _get_name():
    return inspect.stack()[2].function.removeprefix("_").removesuffix("_np")


def _check_types(
    args: Sequence,
    *,
    nstr: int = 0,
    ni32: int = 0,
    nf64: int = 0,
    _np: bool = False,
) -> None:
    if _np:
        if nstr:
            msg = "No string arguments allowed in xrayslib_np"
            raise errors.NumbaValueError(msg)

        msg = "Expected array({0}, ...) got {1}"
        for i in range(ni32):
            if not isinstance(args[i].dtype, types.Integer):
                raise errors.NumbaTypeError(msg.format("int32|int64", args[i]))
        for i in range(ni32, ni32 + nf64):
            if args[i].dtype is not types.float64:
                raise errors.NumbaTypeError(msg.format("float64", args[i]))
    else:
        msg = "Expected {0} got {1}"
        for i in range(nstr):
            if not isinstance(args[i], types.UnicodeType):
                raise errors.NumbaTypeError(msg.format(types.UnicodeType, args[i]))
        for i in range(nstr, nstr + ni32):
            if not isinstance(args[i], types.Integer):
                raise errors.NumbaTypeError(msg.format(types.Integer, args[i]))
        for i in range(nstr + ni32, nstr + ni32 + nf64):
            if args[i] is not types.float64:
                raise errors.NumbaTypeError(msg.format(types.float64, args[i]))


numba_type_dict = {"double": types.float64, "xrlComplex": types.int64}

for i in dir(_xraylib):
    if callable(getattr(_xraylib, i)):
        try:
            signature = getattr(_xraylib, i).__doc__.split("\n")[1]
        except (AttributeError, IndexError):
            print(i)
            continue
        try:
            return_type = signature.split("->")[1].strip()
            numba_return = numba_type_dict[return_type]
        except (IndexError, KeyError):
            print(i)
            continue
        args = signature.split("(")[1].split(")")[0]
        nstr = args.count("char const []")
        ni32 = args.count("int")
        nf64 = args.count("double")

        numba_args = tuple(
            chain.from_iterable(
                [
                    repeat(types.CPointer(types.char), nstr),
                    repeat(types.int64, ni32),
                    repeat(types.float64, nf64),
                ],
            ),
        )
        numba_signature = numba_return(*numba_args, types.voidptr)

        def _overload(i, numba_signature, nstr, ni32, nf64):
            if nstr:

                def idk(c, *args):
                    _check_types((c, *args), nstr=nstr, ni32=ni32, nf64=nf64)
                    xrl_fcn = types.ExternalFunction(i, numba_signature)

                    def impl(c, *args):
                        d = _convert_str(c)
                        error = array([0, 0], dtype=int32)
                        result = xrl_fcn(d.ctypes, *args, error.ctypes)
                        if error.any():
                            print(error[1])
                            raise ValueError("")
                        return result

                    return impl

            else:

                def idk(*args):
                    _check_types(args, nstr=nstr, ni32=ni32, nf64=nf64)
                    xrl_fcn = types.ExternalFunction(i, numba_signature)

                    def impl(*args):
                        error = array([0, 0], dtype=int32)
                        result = xrl_fcn(*args, error.ctypes)
                        if error.any():
                            raise ValueError("")
                        return result

                    return impl

            jit_options = config.xrl.get(i, {})
            extending.overload(getattr(_xraylib, i), jit_options=jit_options)(idk)
            extending.register_jitable(getattr(xraylib, i))

        _overload(i, numba_signature, nstr, ni32, nf64)

        if hasattr(xraylib_np, i):

            def _overload_np(i, numba_signature, ni32, nf64):
                nargs = ni32 + nf64  # nstr is always 0
                if nargs == 1:

                    def idk(a):
                        _check_types((a,), ni32=ni32, nf64=nf64, _np=True)
                        _check_ndim(a)
                        xrl_fcn = types.ExternalFunction(i, numba_signature)

                        @vectorize
                        def _impl(a):
                            return xrl_fcn(a, 0)

                        def impl(a):
                            return _impl(a)

                        return impl

                elif nargs == 2:

                    def idk(*args):
                        _check_types(args, ni32=ni32, nf64=nf64, _np=True)
                        _check_ndim(*args)
                        xrl_fcn = types.ExternalFunction(i, numba_signature)
                        i0, i1 = _indices(*args)

                        @vectorize
                        def _impl(a, b):
                            return xrl_fcn(a, b, 0)

                        def impl(*args):
                            shape = args[0].shape + args[1].shape
                            _a = broadcast_to(args[0][i0], shape)
                            _b = broadcast_to(args[1][i1], shape)
                            return _impl(_a, _b)

                        return impl

                elif nargs == 3:

                    def idk(a, b, c):
                        _check_types((a, b, c), ni32=ni32, nf64=nf64, _np=True)
                        _check_ndim(a, b, c)
                        xrl_fcn = types.ExternalFunction(i, numba_signature)
                        i0, i1, i2 = _indices(a, b, c)

                        @vectorize
                        def _impl(a, b, c):
                            return xrl_fcn(a, b, c, 0)

                        def impl(a, b, c):
                            shape = a.shape + b.shape + c.shape
                            _a = broadcast_to(a[i0], shape)
                            _b = broadcast_to(b[i1], shape)
                            _c = broadcast_to(c[i2], shape)
                            return _impl(_a, _b, _c)

                        return impl

                elif nargs == 4:

                    def idk(a, b, c, d):
                        _check_types((a, b, c, d), ni32=ni32, nf64=nf64, _np=True)
                        _check_ndim(a, b, c, d)
                        xrl_fcn = types.ExternalFunction(i, numba_signature)
                        i0, i1, i2, i3 = _indices(a, b, c, d)

                        @vectorize
                        def _impl(a, b, c, d):
                            return xrl_fcn(a, b, c, d, 0)

                        def impl(a, b, c, d):
                            shape = a.shape + b.shape + c.shape + d.shape
                            _a = broadcast_to(a[i0], shape)
                            _b = broadcast_to(b[i1], shape)
                            _c = broadcast_to(c[i2], shape)
                            _d = broadcast_to(d[i3], shape)
                            return _impl(_a, _b, _c, _d)

                        return impl

                else:
                    return
                jit_options = config.xrl_np.get(i, {})
                extending.overload(getattr(xraylib_np, i), jit_options=jit_options)(idk)

            _overload_np(i, numba_signature, ni32, nf64)
