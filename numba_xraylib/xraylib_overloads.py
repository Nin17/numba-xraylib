"""Overloads for xraylib functions to work with numba."""
# TODO(nin17): check variable names are the same as in the xraylib functions

# ??? why is this necessary. should work in pyproject.toml
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import inspect
from ctypes.util import find_library
from itertools import chain, repeat
from typing import TYPE_CHECKING

import _xraylib
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


def _get_name():
    return inspect.stack()[2].function.removeprefix("_").removesuffix("_np")


def _external_func(
    *,
    nstr: int = 0,
    ni32: int = 0,
    nf64: int = 0,
) -> types.ExternalFunction:
    """External function with nstr string args, ni32 integer args & nf64 double args.

    Parameters
    ----------
    nstr : int, optional
        the number of string args, by default 0
    ni32 : int, optional
        the number of integer args, by default 0
    nf64 : int, optional
        the number of double args, by default 0

    Returns
    -------
    types.ExternalFunction
        the external function

    """
    name = _get_name()
    # TODO(nin17): make this look nicer
    argtypes = (
        [types.CPointer(types.char) for _ in range(nstr)]
        + [types.int64 for _ in range(ni32)]
        + [types.float64 for _ in range(nf64)]
    )
    sig = types.float64(*argtypes, types.voidptr)
    return types.ExternalFunction(name, sig)


def _check_type(arg, numba_type, msg, msgformat):
    if not isinstance(arg, numba_type):
        raise errors.NumbaTypeError(msg.format(msgformat, arg))


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


def _overload(func: callable) -> None:
    fname = func.__name__.removeprefix("_")
    jit_options = config.xrl.get(fname, {})
    xrl_func = getattr(_xraylib, fname)
    extending.overload(xrl_func, jit_options)(func)
    extending.register_jitable(xrl_func)


def _overload_np(func: callable) -> None:
    fname = func.__name__.removeprefix("_").removesuffix("_np")
    jit_options = config.xrl_np.get(fname, {})
    extending.overload(getattr(xraylib_np, fname), jit_options)(func)


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


# --------------------------------------- 1 int -------------------------------------- #


@_overload
def _AtomicWeight(Z):
    _check_types((Z,), ni32=1)
    xrl_fcn = _external_func(ni32=1)

    def impl(Z):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, error.ctypes)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


@_overload_np
def _AtomicWeight_np(Z):
    _check_types((Z,), ni32=1, _np=True)
    _check_ndim(Z)
    xrl_fcn = _external_func(ni32=1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


@_overload
def _ElementDensity(Z):
    _check_types((Z,), ni32=1)
    xrl_fcn = _external_func(ni32=1)

    def impl(Z):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, error.ctypes)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


@_overload_np
def _ElementDensity_np(Z):
    _check_types((Z,), ni32=1, _np=True)
    _check_ndim(Z)
    xrl_fcn = _external_func(ni32=1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


# ------------------------------------- 1 double ------------------------------------- #


@_overload
def _CS_KN(E):
    _check_types((E,), nf64=1)
    xrl_fcn = _external_func(nf64=1)

    def impl(E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


@_overload_np
def _CS_KN_np(E):
    _check_types((E,), nf64=1, _np=True)
    _check_ndim(E)
    xrl_fcn = _external_func(nf64=1)

    @vectorize
    def _impl(E):
        return xrl_fcn(E, 0)

    return lambda E: _impl(E)


@_overload
def _DCS_Thoms(theta):
    _check_types((theta,), nf64=1)
    xrl_fcn = _external_func(nf64=1)

    def impl(theta):
        return xrl_fcn(theta, 0)

    return impl


@_overload_np
def _DCS_Thoms_np(theta):
    _check_types((theta,), nf64=1, _np=True)
    _check_ndim(theta)
    xrl_fcn = _external_func(nf64=1)

    @vectorize
    def _impl(theta):
        return xrl_fcn(theta, 0)

    return lambda theta: _impl(theta)


# --------------------------------------- 2 int -------------------------------------- #


@_overload
def _AtomicLevelWidth(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _AtomicLevelWidth_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        return _impl(_Z, _shell)

    return impl


@_overload
def _AugerRate(Z, auger_trans):
    _check_types((Z, auger_trans), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_AUGER} | {INVALID_AUGER}"

    def impl(Z, auger_trans):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, auger_trans, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _AugerRate_np(Z, auger_trans):
    _check_types((Z, auger_trans), ni32=2, _np=True)
    _check_ndim(Z, auger_trans)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, auger_trans)

    @vectorize
    def _impl(Z, auger_trans):
        return xrl_fcn(Z, auger_trans, 0)

    def impl(Z, auger_trans):
        shape = Z.shape + auger_trans.shape

        _Z = broadcast_to(Z[i0], shape)
        _auger_trans = broadcast_to(auger_trans[i1], shape)

        return _impl(_Z, _auger_trans)

    return impl


@_overload
def _AugerYield(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _AugerYield_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@_overload
def _CosKronTransProb(Z, trans):
    _check_types((Z, trans), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_CK} | {INVALID_CK}"

    def impl(Z, trans):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, trans, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CosKronTransProb_np(Z, trans):
    _check_types((Z, trans), ni32=2, _np=True)
    _check_ndim(Z, trans)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, trans)

    @vectorize
    def _impl(Z, trans):
        return xrl_fcn(Z, trans, 0)

    def impl(Z, trans):
        shape = Z.shape + trans.shape

        _Z = broadcast_to(Z[i0], shape)
        _trans = broadcast_to(trans[i1], shape)

        return _impl(_Z, _trans)

    return impl


@_overload
def _EdgeEnergy(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _EdgeEnergy_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@_overload
def _ElectronConfig(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _ElectronConfig_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@_overload
def _FluorYield(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _FluorYield_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@_overload
def _JumpFactor(Z, shell):
    _check_types((Z, shell), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _JumpFactor_np(Z, shell):
    _check_types((Z, shell), ni32=2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@_overload
def _LineEnergy(Z, line):
    _check_types((Z, line), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _LineEnergy_np(Z, line):
    _check_types((Z, line), ni32=2, _np=True)
    _check_ndim(Z, line)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, line)

    @vectorize
    def _impl(Z, line):
        return xrl_fcn(Z, line, 0)

    def impl(Z, line):
        shape = Z.shape + line.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)

        return _impl(_Z, _line)

    return impl


@_overload
def _RadRate(Z, line):
    _check_types((Z, line), ni32=2)
    xrl_fcn = _external_func(ni32=2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _RadRate_np(Z, line):
    _check_types((Z, line), ni32=2, _np=True)
    _check_ndim(Z, line)
    xrl_fcn = _external_func(ni32=2)
    i0, i1 = _indices(Z, line)

    @vectorize
    def _impl(Z, line):
        return xrl_fcn(Z, line, 0)

    def impl(Z, line):
        shape = Z.shape + line.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)

        return _impl(_Z, _line)

    return impl


# ------------------------------------- 2 double ------------------------------------- #


@_overload
def _ComptonEnergy(E0, theta):
    _check_types((E0, theta), nf64=2)
    xrl_fcn = _external_func(nf64=2)
    msg = NEGATIVE_ENERGY

    def impl(E0, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E0, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _ComptonEnergy_np(E0, theta):
    _check_types((E0, theta), nf64=2, _np=True)
    _check_ndim(E0, theta)
    xrl_fcn = _external_func(nf64=2)
    i0, i1 = _indices(E0, theta)

    @vectorize
    def _impl(E0, theta):
        return xrl_fcn(E0, theta, 0)

    def impl(E0, theta):
        shape = E0.shape + theta.shape

        _E0 = broadcast_to(E0[i0], shape)
        _theta = broadcast_to(theta[i1], shape)

        return _impl(_E0, _theta)

    return impl


@_overload
def _DCS_KN(E, theta):
    _check_types((E, theta), nf64=2)
    xrl_fcn = _external_func(nf64=2)

    def impl(E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


@_overload_np
def _DCS_KN_np(E, theta):
    _check_types((E, theta), nf64=2, _np=True)
    _check_ndim(E, theta)
    xrl_fcn = _external_func(nf64=2)
    i0, i1 = _indices(E, theta)

    @vectorize
    def _impl(E, theta):
        return xrl_fcn(E, theta, 0)

    def impl(E, theta):
        shape = E.shape + theta.shape

        _E = broadcast_to(E[i0], shape)
        _theta = broadcast_to(theta[i1], shape)

        return _impl(_E, _theta)

    return impl


@_overload
def _DCSP_Thoms(theta, phi):
    _check_types((theta, phi), nf64=2)
    xrl_fcn = _external_func(nf64=2)

    def impl(theta, phi):
        return xrl_fcn(theta, phi, 0)

    return impl


@_overload_np
def _DCSP_Thoms_np(theta, phi):
    _check_types((theta, phi), nf64=2, _np=True)
    _check_ndim(theta, phi)
    xrl_fcn = _external_func(nf64=2)
    i0, i1 = _indices(theta, phi)

    @vectorize
    def _impl(theta, phi):
        return xrl_fcn(theta, phi, 0)

    def impl(theta, phi):
        shape = theta.shape + phi.shape

        _theta = broadcast_to(theta[i0], shape)
        _phi = broadcast_to(phi[i1], shape)

        return _impl(_theta, _phi)

    return impl


@_overload
def _MomentTransf(E, theta):
    _check_types((E, theta), nf64=2)
    xrl_fcn = _external_func(nf64=2)

    def impl(E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


@_overload_np
def _MomentTransf_np(E, theta):
    _check_types((E, theta), nf64=2, _np=True)
    _check_ndim(E, theta)
    xrl_fcn = _external_func(nf64=2)
    i0, i1 = _indices(E, theta)

    @vectorize
    def _impl(E, theta):
        return xrl_fcn(E, theta, 0)

    def impl(E, theta):
        shape = E.shape + theta.shape

        _E = broadcast_to(E[i0], shape)
        _theta = broadcast_to(theta[i1], shape)

        return _impl(_E, _theta)

    return impl


# ---------------------------------- 1 int, 1 double --------------------------------- #


@_overload
def _ComptonProfile(Z, p):
    _check_types((Z, p), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_PZ} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, p):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, p, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _ComptonProfile_np(Z, p):
    _check_types((Z, p), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, p)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, p)

    @vectorize
    def _impl(Z, p):
        return xrl_fcn(Z, p, 0)

    def impl(Z, p):
        shape = Z.shape + p.shape

        _Z = broadcast_to(Z[i0], shape)
        _p = broadcast_to(p[i1], shape)

        return _impl(_Z, _p)

    return impl


@_overload
def _CS_Compt(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Compt_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Energy(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Energy_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Photo(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Photo_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Photo_Total(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = (
        f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION} "
        f"| {UNAVALIABLE_PHOTO_CS}"
    )

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Photo_Total_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Rayl(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Rayl_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Total(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Total_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CS_Total_Kissel(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Total_Kissel_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Compt(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Compt_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Photo(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Photo_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Photo_Total(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Photo_Total_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Rayl(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Rayl_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Total(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Total_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _CSb_Total_Kissel(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Total_Kissel_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _FF_Rayl(Z, q):
    _check_types((Z, q), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_Q} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, q):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, q, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _FF_Rayl_np(Z, q):
    _check_types((Z, q), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, q)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, q)

    @vectorize
    def _impl(Z, q):
        return xrl_fcn(Z, q, 0)

    def impl(Z, q):
        shape = Z.shape + q.shape

        _Z = broadcast_to(Z[i0], shape)
        _q = broadcast_to(q[i1], shape)

        return _impl(_Z, _q)

    return impl


@_overload
def _SF_Compt(Z, q):
    _check_types((Z, q), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_Q} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, q):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, q, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _SF_Compt_np(Z, q):
    _check_types((Z, q), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, q)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, q)

    @vectorize
    def _impl(Z, q):
        return xrl_fcn(Z, q, 0)

    def impl(Z, q):
        shape = Z.shape + q.shape

        _Z = broadcast_to(Z[i0], shape)
        _q = broadcast_to(q[i1], shape)

        return _impl(_Z, _q)

    return impl


@_overload
def _Fi(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _Fi_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@_overload
def _Fii(Z, E):
    _check_types((Z, E), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _Fii_np(Z, E):
    _check_types((Z, E), ni32=1, nf64=1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    i0, i1 = _indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL1_pure_kissel(Z, energy):
    _check_types((Z, energy), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, energy):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, energy, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM1_pure_kissel(Z, energy):
    _check_types((Z, energy), ni32=1, nf64=1)
    xrl_fcn = _external_func(ni32=1, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, energy):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, energy, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 2 int, 1 double --------------------------------- #


@_overload
def _ComptonProfile_Partial(Z, shell, pz):
    _check_types((Z, shell, pz), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = (
        f"{Z_OUT_OF_RANGE} | {NEGATIVE_PZ} | {SPLINE_EXTRAPOLATION} | {UNKNOWN_SHELL}"
        f" | {INVALID_SHELL}"
    )

    def impl(Z, shell, pz):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, pz, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _ComptonProfile_Partial_np(Z, shell, pz):
    _check_types((Z, shell, pz), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, pz)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, pz)

    @vectorize
    def _impl(Z, shell, pz):
        return xrl_fcn(Z, shell, pz, 0)

    def impl(Z, shell, pz):
        shape = Z.shape + shell.shape + pz.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _pz = broadcast_to(pz[i2], shape)

        return _impl(_Z, _shell, _pz)

    return impl


@_overload
def _CS_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_FluorLine(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorLine_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CSb_FluorLine(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorLine_np(Z, line, E):
    _check_types((Z, line, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, line, E)

    @vectorize
    def _impl(Z, line, E):
        return xrl_fcn(Z, line, E, 0)

    def impl(Z, line, E):
        shape = Z.shape + line.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _line, _E)

    return impl


@_overload
def _CS_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CS_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CS_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)

        return _impl(_Z, _shell, _E)

    return impl


@_overload
def _CSb_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _CSb_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), ni32=2, nf64=1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _external_func(ni32=2, nf64=1)
    i0, i1, i2 = _indices(Z, shell, E)

    @vectorize
    def _impl(Z, shell, E):
        return xrl_fcn(Z, shell, E, 0)

    def impl(Z, shell, E):
        shape = Z.shape + shell.shape + E.shape
        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        _E = broadcast_to(E[i2], shape)
        return _impl(_Z, _shell, _E)

    return impl


# ---------------------------------- 1 int, 2 double --------------------------------- #


@_overload
def _DCS_Compt(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCS_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    i0, i1, i2 = _indices(Z, E, theta)

    @vectorize
    def _impl(Z, E, theta):
        return xrl_fcn(Z, E, theta, 0)

    def impl(Z, E, theta):
        shape = Z.shape + E.shape + theta.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        return _impl(_Z, _E, _theta)

    return impl


@_overload
def _DCS_Rayl(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCS_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    i0, i1, i2 = _indices(Z, E, theta)

    @vectorize
    def _impl(Z, E, theta):
        return xrl_fcn(Z, E, theta, 0)

    def impl(Z, E, theta):
        shape = Z.shape + E.shape + theta.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)

        return _impl(_Z, _E, _theta)

    return impl


@_overload
def _DCSb_Compt(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSb_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    i0, i1, i2 = _indices(Z, E, theta)

    @vectorize
    def _impl(Z, E, theta):
        return xrl_fcn(Z, E, theta, 0)

    def impl(Z, E, theta):
        shape = Z.shape + E.shape + theta.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        return _impl(_Z, _E, _theta)

    return impl


@_overload
def _DCSb_Rayl(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSb_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), ni32=1, nf64=2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    i0, i1, i2 = _indices(Z, E, theta)

    @vectorize
    def _impl(Z, E, theta):
        return xrl_fcn(Z, E, theta, 0)

    def impl(Z, E, theta):
        shape = Z.shape + E.shape + theta.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        return _impl(_Z, _E, _theta)

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL1_auger_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL1_full_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL1_rad_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL2_pure_kissel(Z, E, PL1):
    _check_types((Z, E, PL1), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM2_pure_kissel(Z, E, PM1):
    _check_types((Z, E, PM1), ni32=1, nf64=2)
    xrl_fcn = _external_func(ni32=1, nf64=2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PM1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PM1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 3 double --------------------------------- #


@_overload
def _DCSP_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSP_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    i0, i1, i2, i3 = _indices(Z, E, theta, phi)

    @vectorize
    def _impl(Z, E, theta, phi):
        return xrl_fcn(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        shape = Z.shape + E.shape + theta.shape + phi.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        _phi = broadcast_to(phi[i3], shape)
        return _impl(_Z, _E, _theta, _phi)

    return impl


@_overload
def _DCSP_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSP_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    i0, i1, i2, i3 = _indices(Z, E, theta, phi)

    @vectorize
    def _impl(Z, E, theta, phi):
        return xrl_fcn(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        shape = Z.shape + E.shape + theta.shape + phi.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        _phi = broadcast_to(phi[i3], shape)
        return _impl(_Z, _E, _theta, _phi)

    return impl


@_overload
def _DCSPb_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSPb_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    i0, i1, i2, i3 = _indices(Z, E, theta, phi)

    @vectorize
    def _impl(Z, E, theta, phi):
        return xrl_fcn(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        shape = Z.shape + E.shape + theta.shape + phi.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        _phi = broadcast_to(phi[i3], shape)
        return _impl(_Z, _E, _theta, _phi)

    return impl


@_overload
def _DCSPb_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSPb_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    i0, i1, i2, i3 = _indices(Z, E, theta, phi)

    @vectorize
    def _impl(Z, E, theta, phi):
        return xrl_fcn(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        shape = Z.shape + E.shape + theta.shape + phi.shape
        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)
        _theta = broadcast_to(theta[i2], shape)
        _phi = broadcast_to(phi[i3], shape)
        return _impl(_Z, _E, _theta, _phi)

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL2_auger_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL2_full_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL2_rad_cascade_kissel(Z, E, PK, PL1):
    _check_types((Z, E, PK, PL1), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL3_pure_kissel(Z, E, PL1, PL2):
    _check_types((Z, E, PL1, PL2), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PL1, PL2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PL1, PL2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM3_pure_kissel(Z, E, PM1, PM2):
    _check_types((Z, E, PM1, PM2), ni32=1, nf64=3)
    xrl_fcn = _external_func(ni32=1, nf64=3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 4 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PL3_auger_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), ni32=1, nf64=4)
    xrl_fcn = _external_func(ni32=1, nf64=4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL3_full_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), ni32=1, nf64=4)
    xrl_fcn = _external_func(ni32=1, nf64=4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2):
    _check_types((Z, E, PK, PL1, PL2), ni32=1, nf64=4)
    xrl_fcn = _external_func(ni32=1, nf64=4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM4_pure_kissel(Z, E, theta, phi, PM1):
    _check_types((Z, E, theta, phi, PM1), ni32=1, nf64=4)
    xrl_fcn = _external_func(ni32=1, nf64=4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 5 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PM1_auger_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), ni32=1, nf64=5)
    xrl_fcn = _external_func(ni32=1, nf64=5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM1_full_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), ni32=1, nf64=5)
    xrl_fcn = _external_func(ni32=1, nf64=5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM1_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3):
    _check_types((Z, E, PK, PL, PL2, PL3), ni32=1, nf64=5)
    xrl_fcn = _external_func(ni32=1, nf64=5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL, PL2, PL3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM5_pure_kissel(Z, E, theta, phi, PM1, PM2):
    _check_types((Z, E, theta, phi, PM1, PM2), ni32=1, nf64=5)
    xrl_fcn = _external_func(ni32=1, nf64=5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 6 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PM2_auger_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), ni32=1, nf64=6)
    xrl_fcn = _external_func(ni32=1, nf64=6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM2_full_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), ni32=1, nf64=6)
    xrl_fcn = _external_func(ni32=1, nf64=6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM2_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, PL4):
    _check_types((Z, E, PK, PL, PL2, PL3, PL4), ni32=1, nf64=6)
    xrl_fcn = _external_func(ni32=1, nf64=6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL, PL2, PL3, PL4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, PL4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 7 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni32=1, nf64=7)
    xrl_fcn = _external_func(ni32=1, nf64=7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni32=1, nf64=7)
    xrl_fcn = _external_func(ni32=1, nf64=7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni32=1, nf64=7)
    xrl_fcn = _external_func(ni32=1, nf64=7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 8 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni32=1, nf64=8)
    xrl_fcn = _external_func(ni32=1, nf64=8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni32=1, nf64=8)
    xrl_fcn = _external_func(ni32=1, nf64=8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni32=1, nf64=8)
    xrl_fcn = _external_func(ni32=1, nf64=8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 9 double --------------------------------- #


# !!! Not implemented in xraylib_np
@_overload
def _PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni32=1, nf64=9)
    xrl_fcn = _external_func(ni32=1, nf64=9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni32=1, nf64=9)
    xrl_fcn = _external_func(ni32=1, nf64=9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@_overload
def _PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni32=1, nf64=9)
    xrl_fcn = _external_func(ni32=1, nf64=9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# ------------------------------------- 3 double ------------------------------------- #


@_overload
def _DCSP_KN(E, theta, phi):
    _check_types((E, theta, phi), nf64=3)
    xrl_fcn = _external_func(nf64=3)
    msg = f"{NEGATIVE_ENERGY}"

    def impl(E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload_np
def _DCSP_KN_np(E, theta, phi):
    _check_types((E, theta, phi), nf64=3, _np=True)
    _check_ndim(E, theta, phi)
    xrl_fcn = _external_func(nf64=3)
    i0, i1, i2 = _indices(E, theta, phi)

    @vectorize
    def _impl(E, theta, phi):
        return xrl_fcn(E, theta, phi, 0)

    def impl(E, theta, phi):
        shape = E.shape + theta.shape + phi.shape
        _E = broadcast_to(E[i0], shape)
        _theta = broadcast_to(theta[i1], shape)
        _phi = broadcast_to(phi[i2], shape)
        return _impl(_E, _theta, _phi)

    return impl


# -------------------------------- 1 string, 1 double -------------------------------- #

# ??? How to pass a python string to an external function


@_overload
def _CS_Total_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""  # TODO: Error message

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CS_Photo_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""  # TODO: Error message

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CS_Rayl_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CS_Compt_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# #@_overload
# def _CS_Energy_CP(compound, E):
#     _check_types((compound, E), nstr=1, nf64=1)
#     xrl_fcn = _external_func(nstr=1, nf64=1)

#     msg = ""

#     def impl(compound, E):
#         c = _convert_str(compound)
#         error = array([0, 0], dtype=int32)
#         result = xrl_fcn(c.ctypes, E, error.ctypes)
#         if error.any():
#             raise ValueError(msg)
#         return result

#     return impl


@_overload
def _CS_Photo_Total_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CS_Total_Kissel_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Total_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Photo_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Rayl_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Compt_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Photo_Total_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _CSb_Total_Kissel_CP(compound, E):
    _check_types((compound, E), nstr=1, nf64=1)
    xrl_fcn = _external_func(nstr=1, nf64=1)

    msg = ""

    def impl(compound, E):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# -------------------------------- 1 string, 2 double -------------------------------- #


@_overload
def _DCS_Rayl_CP(compound, E, theta):
    _check_types((compound, E, theta), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, theta):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCS_Compt_CP(compound, E, theta):
    _check_types((compound, E, theta), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, theta):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCSb_Rayl_CP(compound, E, theta):
    _check_types((compound, E, theta), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, theta):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCSb_Compt_CP(compound, E, theta):
    _check_types((compound, E, theta), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, theta):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _Refractive_Index_Re(compound, E, density):
    _check_types((compound, E, density), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, density):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, density, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _Refractive_Index_Im(compound, E, density):
    _check_types((compound, E, density), nstr=1, nf64=2)
    xrl_fcn = _external_func(nstr=1, nf64=2)

    msg = ""

    def impl(compound, E, density):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, density, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# TODO(nin17): Refractive_Index
# TODO(nin17): complex return values

# -------------------------------- 1 string, 3 double -------------------------------- #


@_overload
def _DCSP_Rayl_CP(compound, E, theta, phi):
    _check_types((compound, E, theta, phi), nstr=1, nf64=3)
    xrl_fcn = _external_func(nstr=1, nf64=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCSP_Compt_CP(compound, E, theta, phi):
    _check_types((compound, E, theta, phi), nstr=1, nf64=3)
    xrl_fcn = _external_func(nstr=1, nf64=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCSPb_Rayl_CP(compound, E, theta, phi):
    _check_types((compound, E, theta, phi), nstr=1, nf64=3)
    xrl_fcn = _external_func(nstr=1, nf64=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


@_overload
def _DCSPb_Compt_CP(compound, E, theta, phi):
    _check_types((compound, E, theta, phi), nstr=1, nf64=3)
    xrl_fcn = _external_func(nstr=1, nf64=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = _convert_str(compound)
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(c.ctypes, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


# TODO(nin17): Other functions with string returns etc...
