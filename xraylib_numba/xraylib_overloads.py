"""Overloads for xraylib functions to work with numba."""
# TODO(nin17): check variable names are the same as in the xraylib functions

# ruff: noqa: ANN001, ANN202

from __future__ import annotations

from ctypes.util import find_library
from itertools import chain, repeat
from typing import TYPE_CHECKING

import _xraylib
import xraylib_np
from llvmlite import binding
from numba import errors, types, vectorize
from numba.extending import overload
from numpy import array, broadcast_to, int32

import xraylib

from .config import config

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from types import EllipsisType
    else:
        EllipsisType = "builtins.Ellipsis"

binding.load_library_permanently(find_library("xrl"))


def _nint_ndouble(name: str, nint: int = 0, ndouble: int = 0) -> types.ExternalFunction:
    """External function with nint integer args followed by ndouble double args.

    Parameters
    ----------
    name : str
        the name of the external function
    nint : int, optional
        the number of integer args, by default 0
    ndouble : int, optional
        the number of double args, by default 0

    Returns
    -------
    types.ExternalFunction
        the external function

    """
    argtypes = [types.int64 for _ in range(nint)] + [
        types.float64 for _ in range(ndouble)
    ]
    sig = types.float64(*argtypes, types.voidptr)
    return types.ExternalFunction(name, sig)


def _check_types(args, nint: int = 0, ndouble: int = 0, *, _np: bool = False) -> None:
    if _np:
        msg = "Expected array({0}, ...) got {1}"
        for i in range(nint):
            if not isinstance(args[i].dtype, types.Integer):
                raise errors.NumbaTypeError(msg.format("int32|int64", args[i]))
        for i in range(nint, nint + ndouble):
            if args[i].dtype is not types.float64:
                raise errors.NumbaTypeError(msg.format("float64", args[i]))
    else:
        msg = "Expected {0} got {1}"
        for i in range(nint):
            if not isinstance(args[i], types.Integer):
                raise errors.NumbaTypeError(msg.format(types.Integer, args[i]))
        for i in range(nint, nint + ndouble):
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


def _AtomicWeight(Z):
    _check_types((Z,), 1)
    xrl_fcn = _nint_ndouble("AtomicWeight", 1)

    def impl(Z):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, error.ctypes)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


overload(xraylib.AtomicWeight, jit_options=config.xrl.get("AtomicWeight", {}))(
    _AtomicWeight,
)
overload(_xraylib.AtomicWeight, jit_options=config.xrl.get("AtomicWeight", {}))(
    _AtomicWeight,
)


@overload(xraylib_np.AtomicWeight, jit_options=config.xrl_np.get("AtomicWeight", {}))
def _AtomicWeight_np(Z):
    _check_types((Z,), 1, _np=True)
    _check_ndim(Z)
    xrl_fcn = _nint_ndouble("AtomicWeight", 1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


def _ElementDensity(Z):
    _check_types((Z,), 1)
    xrl_fcn = _nint_ndouble("ElementDensity", 1)

    def impl(Z):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, error.ctypes)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


overload(xraylib.ElementDensity, jit_options=config.xrl.get("ElementDensity", {}))(
    _ElementDensity,
)
overload(_xraylib.ElementDensity, jit_options=config.xrl.get("ElementDensity", {}))(
    _ElementDensity,
)


@overload(
    xraylib_np.ElementDensity,
    jit_options=config.xrl_np.get("ElementDensity", {}),
)
def _ElementDensity_np(Z):
    _check_types((Z,), 1, _np=True)
    _check_ndim(Z)
    xrl_fcn = _nint_ndouble("ElementDensity", 1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


# ------------------------------------- 1 double ------------------------------------- #


def _CS_KN(E):
    _check_types((E,), 0, 1)
    xrl_fcn = _nint_ndouble("CS_KN", 0, 1)

    def impl(E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.CS_KN, jit_options=config.xrl.get("CS_KN", {}))(_CS_KN)
overload(_xraylib.CS_KN, jit_options=config.xrl.get("CS_KN", {}))(_CS_KN)


@overload(xraylib_np.CS_KN, jit_options=config.xrl_np.get("CS_KN", {}))
def _CS_KN_np(E):
    _check_types((E,), 0, 1, _np=True)
    _check_ndim(E)
    xrl_fcn = _nint_ndouble("CS_KN", 0, 1)

    @vectorize
    def _impl(E):
        return xrl_fcn(E, 0)

    return lambda E: _impl(E)


def _DCS_Thoms(theta):
    _check_types((theta,), 0, 1)
    xrl_fcn = _nint_ndouble("DCS_Thoms", 0, 1)

    def impl(theta):
        return xrl_fcn(theta, 0)

    return impl


overload(xraylib.DCS_Thoms, jit_options=config.xrl.get("DCS_Thoms", {}))(_DCS_Thoms)
overload(_xraylib.DCS_Thoms, jit_options=config.xrl.get("DCS_Thoms", {}))(_DCS_Thoms)


@overload(xraylib_np.DCS_Thoms, jit_options=config.xrl_np.get("DCS_Thoms", {}))
def _DCS_Thoms_np(theta):
    _check_types((theta,), 0, 1, _np=True)
    _check_ndim(theta)
    xrl_fcn = _nint_ndouble("DCS_Thoms", 0, 1)

    @vectorize
    def _impl(theta):
        return xrl_fcn(theta, 0)

    return lambda theta: _impl(theta)


# --------------------------------------- 2 int -------------------------------------- #


def _AtomicLevelWidth(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("AtomicLevelWidth", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AtomicLevelWidth, jit_options=config.xrl.get("AtomicLevelWidth", {}))(
    _AtomicLevelWidth,
)
overload(_xraylib.AtomicLevelWidth, jit_options=config.xrl.get("AtomicLevelWidth", {}))(
    _AtomicLevelWidth,
)


@overload(
    xraylib_np.AtomicLevelWidth,
    jit_options=config.xrl_np.get("AtomicLevelWidth", {}),
)
def _AtomicLevelWidth_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("AtomicLevelWidth", 2)
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


def _AugerRate(Z, auger_trans):
    _check_types((Z, auger_trans), 2)
    xrl_fcn = _nint_ndouble("AugerRate", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_AUGER} | {INVALID_AUGER}"

    def impl(Z, auger_trans):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, auger_trans, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AugerRate, jit_options=config.xrl.get("AugerRate", {}))(_AugerRate)
overload(_xraylib.AugerRate, jit_options=config.xrl.get("AugerRate", {}))(_AugerRate)


@overload(xraylib_np.AugerRate, jit_options=config.xrl_np.get("AugerRate", {}))
def _AugerRate_np(Z, auger_trans):
    _check_types((Z, auger_trans), 2, _np=True)
    _check_ndim(Z, auger_trans)
    xrl_fcn = _nint_ndouble("AugerRate", 2)
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


def _AugerYield(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("AugerYield", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AugerYield, jit_options=config.xrl.get("AugerYield", {}))(_AugerYield)
overload(_xraylib.AugerYield, jit_options=config.xrl.get("AugerYield", {}))(_AugerYield)


@overload(xraylib_np.AugerYield, jit_options=config.xrl_np.get("AugerYield", {}))
def _AugerYield_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("AugerYield", 2)
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


def _CosKronTransProb(Z, trans):
    _check_types((Z, trans), 2)
    xrl_fcn = _nint_ndouble("CosKronTransProb", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_CK} | {INVALID_CK}"

    def impl(Z, trans):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, trans, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CosKronTransProb, jit_options=config.xrl.get("CosKronTransProb", {}))(
    _CosKronTransProb,
)
overload(_xraylib.CosKronTransProb, jit_options=config.xrl.get("CosKronTransProb", {}))(
    _CosKronTransProb,
)


@overload(
    xraylib_np.CosKronTransProb,
    jit_options=config.xrl_np.get("CosKronTransProb", {}),
)
def _CosKronTransProb_np(Z, trans):
    _check_types((Z, trans), 2, _np=True)
    _check_ndim(Z, trans)
    xrl_fcn = _nint_ndouble("CosKronTransProb", 2)
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


def _EdgeEnergy(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("EdgeEnergy", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.EdgeEnergy, jit_options=config.xrl.get("EdgeEnergy", {}))(_EdgeEnergy)
overload(_xraylib.EdgeEnergy, jit_options=config.xrl.get("EdgeEnergy", {}))(_EdgeEnergy)


@overload(xraylib_np.EdgeEnergy, jit_options=config.xrl_np.get("EdgeEnergy", {}))
def _EdgeEnergy_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("EdgeEnergy", 2)
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


def _ElectronConfig(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("ElectronConfig", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ElectronConfig, jit_options=config.xrl.get("ElectronConfig", {}))(
    _ElectronConfig,
)
overload(_xraylib.ElectronConfig, jit_options=config.xrl.get("ElectronConfig", {}))(
    _ElectronConfig,
)


@overload(
    xraylib_np.ElectronConfig,
    jit_options=config.xrl_np.get("ElectronConfig", {}),
)
def _ElectronConfig_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("ElectronConfig", 2)
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


def _FluorYield(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("FluorYield", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.FluorYield, jit_options=config.xrl.get("FluorYield", {}))(_FluorYield)
overload(_xraylib.FluorYield, jit_options=config.xrl.get("FluorYield", {}))(_FluorYield)


@overload(xraylib_np.FluorYield, jit_options=config.xrl_np.get("FluorYield", {}))
def _FluorYield_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("FluorYield", 2)
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


def _JumpFactor(Z, shell):
    _check_types((Z, shell), 2)
    xrl_fcn = _nint_ndouble("JumpFactor", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.JumpFactor, jit_options=config.xrl.get("JumpFactor", {}))(_JumpFactor)
overload(_xraylib.JumpFactor, jit_options=config.xrl.get("JumpFactor", {}))(_JumpFactor)


@overload(xraylib_np.JumpFactor, jit_options=config.xrl_np.get("JumpFactor", {}))
def _JumpFactor_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    _check_ndim(Z, shell)
    xrl_fcn = _nint_ndouble("JumpFactor", 2)
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


def _LineEnergy(Z, line):
    _check_types((Z, line), 2)
    xrl_fcn = _nint_ndouble("LineEnergy", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.LineEnergy, jit_options=config.xrl.get("LineEnergy", {}))(_LineEnergy)
overload(_xraylib.LineEnergy, jit_options=config.xrl.get("LineEnergy", {}))(_LineEnergy)


@overload(xraylib_np.LineEnergy, jit_options=config.xrl_np.get("LineEnergy", {}))
def _LineEnergy_np(Z, line):
    _check_types((Z, line), 2, _np=True)
    _check_ndim(Z, line)
    xrl_fcn = _nint_ndouble("LineEnergy", 2)
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


def _RadRate(Z, line):
    _check_types((Z, line), 2)
    xrl_fcn = _nint_ndouble("RadRate", 2)
    msg = f"{Z_OUT_OF_RANGE} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.RadRate, jit_options=config.xrl.get("RadRate", {}))(_RadRate)
overload(_xraylib.RadRate, jit_options=config.xrl.get("RadRate", {}))(_RadRate)


@overload(xraylib_np.RadRate, jit_options=config.xrl_np.get("RadRate", {}))
def _RadRate_np(Z, line):
    _check_types((Z, line), 2, _np=True)
    _check_ndim(Z, line)
    xrl_fcn = _nint_ndouble("RadRate", 2)
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


def _ComptonEnergy(E0, theta):
    _check_types((E0, theta), 0, 2)
    xrl_fcn = _nint_ndouble("ComptonEnergy", 0, 2)
    msg = NEGATIVE_ENERGY

    def impl(E0, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E0, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ComptonEnergy, jit_options=config.xrl.get("ComptonEnergy", {}))(
    _ComptonEnergy,
)
overload(_xraylib.ComptonEnergy, jit_options=config.xrl.get("ComptonEnergy", {}))(
    _ComptonEnergy,
)


@overload(xraylib_np.ComptonEnergy, jit_options=config.xrl_np.get("ComptonEnergy", {}))
def _ComptonEnergy_np(E0, theta):
    _check_types((E0, theta), 0, 2, _np=True)
    _check_ndim(E0, theta)
    xrl_fcn = _nint_ndouble("ComptonEnergy", 0, 2)
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


def _DCS_KN(E, theta):
    _check_types((E, theta), 0, 2)
    xrl_fcn = _nint_ndouble("DCS_KN", 0, 2)

    def impl(E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.DCS_KN, jit_options=config.xrl.get("DCS_KN", {}))(_DCS_KN)
overload(_xraylib.DCS_KN, jit_options=config.xrl.get("DCS_KN", {}))(_DCS_KN)


@overload(xraylib_np.DCS_KN, jit_options=config.xrl_np.get("DCS_KN", {}))
def _DCS_KN_np(E, theta):
    _check_types((E, theta), 0, 2, _np=True)
    _check_ndim(E, theta)
    xrl_fcn = _nint_ndouble("DCS_KN", 0, 2)
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


def _DCSP_Thoms(theta, phi):
    _check_types((theta, phi), 0, 2)
    xrl_fcn = _nint_ndouble("DCSP_Thoms", 0, 2)

    def impl(theta, phi):
        return xrl_fcn(theta, phi, 0)

    return impl


overload(xraylib.DCSP_Thoms, jit_options=config.xrl.get("DCSP_Thoms", {}))(_DCSP_Thoms)
overload(_xraylib.DCSP_Thoms, jit_options=config.xrl.get("DCSP_Thoms", {}))(_DCSP_Thoms)


@overload(xraylib_np.DCSP_Thoms, jit_options=config.xrl_np.get("DCSP_Thoms", {}))
def _DCSP_Thoms_np(theta, phi):
    _check_types((theta, phi), 0, 2, _np=True)
    _check_ndim(theta, phi)
    xrl_fcn = _nint_ndouble("DCSP_Thoms", 0, 2)
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


def _MomentTransf(E, theta):
    _check_types((E, theta), 0, 2)
    xrl_fcn = _nint_ndouble("MomentTransf", 0, 2)

    def impl(E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, error.ctypes)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.MomentTransf, jit_options=config.xrl.get("MomentTransf", {}))(
    _MomentTransf,
)
overload(_xraylib.MomentTransf, jit_options=config.xrl.get("MomentTransf", {}))(
    _MomentTransf,
)


@overload(xraylib_np.MomentTransf, jit_options=config.xrl_np.get("MomentTransf", {}))
def _MomentTransf_np(E, theta):
    _check_types((E, theta), 0, 2, _np=True)
    _check_ndim(E, theta)
    xrl_fcn = _nint_ndouble("MomentTransf", 0, 2)
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


def _ComptonProfile(Z, p):
    _check_types((Z, p), 1, 1)
    xrl_fcn = _nint_ndouble("ComptonProfile", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_PZ} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, p):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, p, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ComptonProfile, jit_options=config.xrl.get("ComptonProfile", {}))(
    _ComptonProfile,
)
overload(_xraylib.ComptonProfile, jit_options=config.xrl.get("ComptonProfile", {}))(
    _ComptonProfile,
)


@overload(
    xraylib_np.ComptonProfile,
    jit_options=config.xrl_np.get("ComptonProfile", {}),
)
def _ComptonProfile_np(Z, p):
    _check_types((Z, p), 1, 1, _np=True)
    _check_ndim(Z, p)
    xrl_fcn = _nint_ndouble("ComptonProfile", 1, 1)
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


def _CS_Compt(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Compt", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Compt, jit_options=config.xrl.get("CS_Compt", {}))(_CS_Compt)
overload(_xraylib.CS_Compt, jit_options=config.xrl.get("CS_Compt", {}))(_CS_Compt)


@overload(xraylib_np.CS_Compt, jit_options=config.xrl_np.get("CS_Compt", {}))
def _CS_Compt_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Compt", 1, 1)
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


def _CS_Energy(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Energy", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Energy, jit_options=config.xrl.get("CS_Energy", {}))(_CS_Energy)
overload(_xraylib.CS_Energy, jit_options=config.xrl.get("CS_Energy", {}))(_CS_Energy)


@overload(xraylib_np.CS_Energy, jit_options=config.xrl_np.get("CS_Energy", {}))
def _CS_Energy(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Energy", 1, 1)
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


def _CS_Photo(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Photo", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Photo, jit_options=config.xrl.get("CS_Photo", {}))(_CS_Photo)
overload(_xraylib.CS_Photo, jit_options=config.xrl.get("CS_Photo", {}))(_CS_Photo)


@overload(xraylib_np.CS_Photo, jit_options=config.xrl_np.get("CS_Photo", {}))
def _CS_Photo_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Photo", 1, 1)
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


def _CS_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Photo_Total", 1, 1)
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


overload(xraylib.CS_Photo_Total, jit_options=config.xrl.get("CS_Photo_Total", {}))(
    _CS_Photo_Total,
)
overload(_xraylib.CS_Photo_Total, jit_options=config.xrl.get("CS_Photo_Total", {}))(
    _CS_Photo_Total,
)


@overload(
    xraylib_np.CS_Photo_Total,
    jit_options=config.xrl_np.get("CS_Photo_Total", {}),
)
def _CS_Photo_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Photo_Total", 1, 1)
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


def _CS_Rayl(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Rayl", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Rayl, jit_options=config.xrl.get("CS_Rayl", {}))(_CS_Rayl)
overload(_xraylib.CS_Rayl, jit_options=config.xrl.get("CS_Rayl", {}))(_CS_Rayl)


@overload(xraylib_np.CS_Rayl, jit_options=config.xrl_np.get("CS_Rayl", {}))
def _CS_Rayl_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Rayl", 1, 1)
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


def _CS_Total(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Total", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Total, jit_options=config.xrl.get("CS_Total", {}))(_CS_Total)
overload(_xraylib.CS_Total, jit_options=config.xrl.get("CS_Total", {}))(_CS_Total)


@overload(xraylib_np.CS_Total, jit_options=config.xrl_np.get("CS_Total", {}))
def _CS_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Total", 1, 1)
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


def _CS_Total_Kissel(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CS_Total_Kissel", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Total_Kissel, jit_options=config.xrl.get("CS_Total_Kissel", {}))(
    _CS_Total_Kissel,
)
overload(_xraylib.CS_Total_Kissel, jit_options=config.xrl.get("CS_Total_Kissel", {}))(
    _CS_Total_Kissel,
)


@overload(
    xraylib_np.CS_Total_Kissel,
    jit_options=config.xrl_np.get("CS_Total_Kissel", {}),
)
def _CS_Total_Kissel_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CS_Total_Kissel", 1, 1)
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


def _CSb_Compt(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Compt", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Compt, jit_options=config.xrl.get("CSb_Compt", {}))(_CSb_Compt)
overload(_xraylib.CSb_Compt, jit_options=config.xrl.get("CSb_Compt", {}))(_CSb_Compt)


@overload(xraylib_np.CSb_Compt, jit_options=config.xrl_np.get("CSb_Compt", {}))
def _CSb_Compt_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Compt", 1, 1)
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


def _CSb_Photo(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Photo", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Photo, jit_options=config.xrl.get("CSb_Photo", {}))(_CSb_Photo)
overload(_xraylib.CSb_Photo, jit_options=config.xrl.get("CSb_Photo", {}))(_CSb_Photo)


@overload(xraylib_np.CSb_Photo, jit_options=config.xrl_np.get("CSb_Photo", {}))
def _CSb_Photo_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Photo", 1, 1)
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


def _CSb_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Photo_Total", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Photo_Total, jit_options=config.xrl.get("CSb_Photo_Total", {}))(
    _CSb_Photo_Total,
)
overload(_xraylib.CSb_Photo_Total, jit_options=config.xrl.get("CSb_Photo_Total", {}))(
    _CSb_Photo_Total,
)


@overload(
    xraylib_np.CSb_Photo_Total,
    jit_options=config.xrl_np.get("CSb_Photo_Total", {}),
)
def _CSb_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Photo_Total", 1, 1)
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


def _CSb_Rayl(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Rayl", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Rayl, jit_options=config.xrl.get("CSb_Rayl", {}))(_CSb_Rayl)
overload(_xraylib.CSb_Rayl, jit_options=config.xrl.get("CSb_Rayl", {}))(_CSb_Rayl)


@overload(xraylib_np.CSb_Rayl, jit_options=config.xrl_np.get("CSb_Rayl", {}))
def _CSb_Rayl_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Rayl", 1, 1)
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


def _CSb_Total(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Total", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Total, jit_options=config.xrl.get("CSb_Total", {}))(_CSb_Total)
overload(_xraylib.CSb_Total, jit_options=config.xrl.get("CSb_Total", {}))(_CSb_Total)


@overload(xraylib_np.CSb_Total, jit_options=config.xrl_np.get("CSb_Total", {}))
def _CSb_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Total", 1, 1)
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


def _CSb_Total_Kissel(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("CSb_Total_Kissel", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Total_Kissel, jit_options=config.xrl.get("CSb_Total_Kissel", {}))(
    _CSb_Total_Kissel,
)
overload(_xraylib.CSb_Total_Kissel, jit_options=config.xrl.get("CSb_Total_Kissel", {}))(
    _CSb_Total_Kissel,
)


@overload(
    xraylib_np.CSb_Total_Kissel,
    jit_options=config.xrl_np.get("CSb_Total_Kissel", {}),
)
def _CSb_Total_Kissel_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("CSb_Total_Kissel", 1, 1)
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


def _FF_Rayl(Z, q):
    _check_types((Z, q), 1, 1)
    xrl_fcn = _nint_ndouble("FF_Rayl", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_Q} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, q):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, q, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.FF_Rayl, jit_options=config.xrl.get("FF_Rayl", {}))(_FF_Rayl)
overload(_xraylib.FF_Rayl, jit_options=config.xrl.get("FF_Rayl", {}))(_FF_Rayl)


@overload(xraylib_np.FF_Rayl, jit_options=config.xrl_np.get("FF_Rayl", {}))
def _FF_Rayl_np(Z, q):
    _check_types((Z, q), 1, 1, _np=True)
    _check_ndim(Z, q)
    xrl_fcn = _nint_ndouble("FF_Rayl", 1, 1)
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


def _SF_Compt(Z, q):
    _check_types((Z, q), 1, 1)
    xrl_fcn = _nint_ndouble("SF_Compt", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_Q} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, q):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, q, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.SF_Compt, jit_options=config.xrl.get("SF_Compt", {}))(_SF_Compt)
overload(_xraylib.SF_Compt, jit_options=config.xrl.get("SF_Compt", {}))(_SF_Compt)


@overload(xraylib_np.SF_Compt, jit_options=config.xrl_np.get("SF_Compt", {}))
def _SF_Compt_np(Z, q):
    _check_types((Z, q), 1, 1, _np=True)
    _check_ndim(Z, q)
    xrl_fcn = _nint_ndouble("SF_Compt", 1, 1)
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


def _Fi(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("Fi", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.Fi, jit_options=config.xrl.get("Fi", {}))(_Fi)
overload(_xraylib.Fi, jit_options=config.xrl.get("Fi", {}))(_Fi)


@overload(xraylib_np.Fi, jit_options=config.xrl_np.get("Fi", {}))
def _Fi_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("Fi", 1, 1)
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


def _Fii(Z, E):
    _check_types((Z, E), 1, 1)
    xrl_fcn = _nint_ndouble("Fii", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.Fii, jit_options=config.xrl.get("Fii", {}))(_Fii)
overload(_xraylib.Fii, jit_options=config.xrl.get("Fii", {}))(_Fii)


@overload(xraylib_np.Fii, jit_options=config.xrl_np.get("Fii", {}))
def _Fii_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    _check_ndim(Z, E)
    xrl_fcn = _nint_ndouble("Fii", 1, 1)
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
def _PL1_pure_kissel(Z, energy):
    _check_types((Z, energy), 1, 1)
    xrl_fcn = _nint_ndouble("PL1_pure_kissel", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, energy):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, energy, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL1_pure_kissel, jit_options=config.xrl.get("PL1_pure_kissel", {}))(
    _PL1_pure_kissel,
)
overload(_xraylib.PL1_pure_kissel, jit_options=config.xrl.get("PL1_pure_kissel", {}))(
    _PL1_pure_kissel,
)


# !!! Not implemented in xraylib_np
def _PM1_pure_kissel(Z, energy):
    _check_types((Z, energy), 1, 1)
    xrl_fcn = _nint_ndouble("PM1_pure_kissel", 1, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {SPLINE_EXTRAPOLATION}"

    def impl(Z, energy):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, energy, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM1_pure_kissel, jit_options=config.xrl.get("PM1_pure_kissel", {}))(
    _PM1_pure_kissel,
)
overload(_xraylib.PM1_pure_kissel, jit_options=config.xrl.get("PM1_pure_kissel", {}))(
    _PM1_pure_kissel,
)

# ---------------------------------- 2 int, 1 double --------------------------------- #


def _ComptonProfile_Partial(Z, shell, pz):
    _check_types((Z, shell, pz), 2, 1)
    xrl_fcn = _nint_ndouble("ComptonProfile_Partial", 2, 1)
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


overload(
    xraylib.ComptonProfile_Partial,
    jit_options=config.xrl.get("ComptonProfile_Partial", {}),
)(_ComptonProfile_Partial)
overload(
    _xraylib.ComptonProfile_Partial,
    jit_options=config.xrl.get("ComptonProfile_Partial", {}),
)(_ComptonProfile_Partial)


@overload(
    xraylib_np.ComptonProfile_Partial,
    jit_options=config.xrl_np.get("ComptonProfile_Partial", {}),
)
def _ComptonProfile_Partial_np(Z, shell, pz):
    _check_types((Z, shell, pz), 2, 1, _np=True)
    _check_ndim(Z, shell, pz)
    xrl_fcn = _nint_ndouble("ComptonProfile_Partial", 2, 1)
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


def _CS_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorLine_Kissel,
    jit_options=config.xrl.get("CS_FluorLine_Kissel", {}),
)(_CS_FluorLine_Kissel)
overload(
    _xraylib.CS_FluorLine_Kissel,
    jit_options=config.xrl.get("CS_FluorLine_Kissel", {}),
)(_CS_FluorLine_Kissel)


@overload(
    xraylib_np.CS_FluorLine_Kissel,
    jit_options=config.xrl_np.get("CS_FluorLine_Kissel", {}),
)
def _CS_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel", 2, 1)
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


def _CSb_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorLine_Kissel,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel", {}),
)(_CSb_FluorLine_Kissel)
overload(
    _xraylib.CSb_FluorLine_Kissel,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel", {}),
)(_CSb_FluorLine_Kissel)


@overload(
    xraylib_np.CSb_FluorLine_Kissel,
    jit_options=config.xrl_np.get("CSb_FluorLine_Kissel", {}),
)
def _CSb_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel", 2, 1)
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


def _CS_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorLine_Kissel_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Cascade", {}),
)(_CS_FluorLine_Kissel_Cascade)
overload(
    _xraylib.CS_FluorLine_Kissel_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Cascade", {}),
)(_CS_FluorLine_Kissel_Cascade)


@overload(
    xraylib_np.CS_FluorLine_Kissel_Cascade,
    jit_options=config.xrl_np.get("CS_FluorLine_Kissel_Cascade", {}),
)
def _CS_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Cascade", 2, 1)
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


def _CSb_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorLine_Kissel_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Cascade", {}),
)(_CSb_FluorLine_Kissel_Cascade)
overload(
    _xraylib.CSb_FluorLine_Kissel_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Cascade", {}),
)(_CSb_FluorLine_Kissel_Cascade)


@overload(
    xraylib_np.CSb_FluorLine_Kissel_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorLine_Kissel_Cascade", {}),
)
def _CSb_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Cascade", 2, 1)
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


def _CS_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_no_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_no_Cascade", {}),
)(_CS_FluorLine_Kissel_no_Cascade)
overload(
    _xraylib.CS_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_no_Cascade", {}),
)(_CS_FluorLine_Kissel_no_Cascade)


@overload(
    xraylib_np.CS_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl_np.get("CS_FluorLine_Kissel_no_Cascade", {}),
)
def _CS_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_no_Cascade", 2, 1)
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


def _CSb_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_no_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_no_Cascade", {}),
)(_CSb_FluorLine_Kissel_no_Cascade)
overload(
    _xraylib.CSb_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_no_Cascade", {}),
)(_CSb_FluorLine_Kissel_no_Cascade)


@overload(
    xraylib_np.CSb_FluorLine_Kissel_no_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorLine_Kissel_no_Cascade", {}),
)
def _CSb_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_no_Cascade", 2, 1)
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


def _CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Nonradiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Nonradiative_Cascade", {}),
)(_CS_FluorLine_Kissel_Nonradiative_Cascade)
overload(
    _xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Nonradiative_Cascade", {}),
)(_CS_FluorLine_Kissel_Nonradiative_Cascade)


@overload(
    xraylib_np.CS_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl_np.get("CS_FluorLine_Kissel_Nonradiative_Cascade", {}),
)
def _CS_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Nonradiative_Cascade", 2, 1)
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


def _CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Nonradiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Nonradiative_Cascade", {}),
)(_CSb_FluorLine_Kissel_Nonradiative_Cascade)
overload(
    _xraylib.CSb_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Nonradiative_Cascade", {}),
)(_CSb_FluorLine_Kissel_Nonradiative_Cascade)


@overload(
    xraylib_np.CSb_FluorLine_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorLine_Kissel_Nonradiative_Cascade", {}),
)
def _CSb_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Nonradiative_Cascade", 2, 1)
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


def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Radiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Radiative_Cascade", {}),
)(_CS_FluorLine_Kissel_Radiative_Cascade)
overload(
    _xraylib.CS_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CS_FluorLine_Kissel_Radiative_Cascade", {}),
)(_CS_FluorLine_Kissel_Radiative_Cascade)


@overload(
    xraylib_np.CS_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl_np.get("CS_FluorLine_Kissel_Radiative_Cascade", {}),
)
def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine_Kissel_Radiative_Cascade", 2, 1)
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


def _CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Radiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Radiative_Cascade", {}),
)(_CSb_FluorLine_Kissel_Radiative_Cascade)
overload(
    _xraylib.CSb_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorLine_Kissel_Radiative_Cascade", {}),
)(_CSb_FluorLine_Kissel_Radiative_Cascade)


@overload(
    xraylib_np.CSb_FluorLine_Kissel_Radiative_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorLine_Kissel_Radiative_Cascade", {}),
)
def _CSb_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine_Kissel_Radiative_Cascade", 2, 1)
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


def _CS_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorShell_Kissel,
    jit_options=config.xrl.get("CS_FluorShell_Kissel", {}),
)(_CS_FluorShell_Kissel)
overload(
    _xraylib.CS_FluorShell_Kissel,
    jit_options=config.xrl.get("CS_FluorShell_Kissel", {}),
)(_CS_FluorShell_Kissel)


@overload(
    xraylib_np.CS_FluorShell_Kissel,
    jit_options=config.xrl_np.get("CS_FluorShell_Kissel", {}),
)
def _CS_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel", 2, 1)
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


def _CSb_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorShell_Kissel,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel", {}),
)(_CSb_FluorShell_Kissel)
overload(
    _xraylib.CSb_FluorShell_Kissel,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel", {}),
)(_CSb_FluorShell_Kissel)


@overload(
    xraylib_np.CSb_FluorShell_Kissel,
    jit_options=config.xrl_np.get("CSb_FluorShell_Kissel", {}),
)
def _CSb_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel", 2, 1)
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


def _CS_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorShell_Kissel_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Cascade", {}),
)(_CS_FluorShell_Kissel_Cascade)
overload(
    _xraylib.CS_FluorShell_Kissel_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Cascade", {}),
)(_CS_FluorShell_Kissel_Cascade)


@overload(
    xraylib_np.CS_FluorShell_Kissel_Cascade,
    jit_options=config.xrl_np.get("CS_FluorShell_Kissel_Cascade", {}),
)
def _CS_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Cascade", 2, 1)
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


def _CSb_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorShell_Kissel_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Cascade", {}),
)(_CSb_FluorShell_Kissel_Cascade)
overload(
    _xraylib.CSb_FluorShell_Kissel_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Cascade", {}),
)(_CSb_FluorShell_Kissel_Cascade)


@overload(
    xraylib_np.CSb_FluorShell_Kissel_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorShell_Kissel_Cascade", {}),
)
def _CSb_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Cascade", 2, 1)
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


def _CS_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_no_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_no_Cascade", {}),
)(_CS_FluorShell_Kissel_no_Cascade)
overload(
    _xraylib.CS_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_no_Cascade", {}),
)(_CS_FluorShell_Kissel_no_Cascade)


@overload(
    xraylib_np.CS_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl_np.get("CS_FluorShell_Kissel_no_Cascade", {}),
)
def _CS_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_no_Cascade", 2, 1)
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


def _CSb_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_no_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_no_Cascade", {}),
)(_CSb_FluorShell_Kissel_no_Cascade)
overload(
    _xraylib.CSb_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_no_Cascade", {}),
)(_CSb_FluorShell_Kissel_no_Cascade)


@overload(
    xraylib_np.CSb_FluorShell_Kissel_no_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorShell_Kissel_no_Cascade", {}),
)
def _CSb_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_no_Cascade", 2, 1)
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


def _CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Nonradiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Nonradiative_Cascade", {}),
)(_CS_FluorShell_Kissel_Nonradiative_Cascade)
overload(
    _xraylib.CS_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Nonradiative_Cascade", {}),
)(_CS_FluorShell_Kissel_Nonradiative_Cascade)


@overload(
    xraylib_np.CS_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl_np.get("CS_FluorShell_Kissel_Nonradiative_Cascade", {}),
)
def _CS_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Nonradiative_Cascade", 2, 1)
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


def _CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Nonradiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Nonradiative_Cascade", {}),
)(_CSb_FluorShell_Kissel_Nonradiative_Cascade)
overload(
    _xraylib.CSb_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Nonradiative_Cascade", {}),
)(_CSb_FluorShell_Kissel_Nonradiative_Cascade)


@overload(
    xraylib_np.CSb_FluorShell_Kissel_Nonradiative_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorShell_Kissel_Nonradiative_Cascade", {}),
)
def _CSb_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Nonradiative_Cascade", 2, 1)
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


def _CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Radiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CS_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Radiative_Cascade", {}),
)(_CS_FluorShell_Kissel_Radiative_Cascade)
overload(
    _xraylib.CS_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CS_FluorShell_Kissel_Radiative_Cascade", {}),
)(_CS_FluorShell_Kissel_Radiative_Cascade)


@overload(
    xraylib_np.CS_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl_np.get("CS_FluorShell_Kissel_Radiative_Cascade", {}),
)
def _CS_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell_Kissel_Radiative_Cascade", 2, 1)
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


def _CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Radiative_Cascade", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Radiative_Cascade", {}),
)(_CSb_FluorShell_Kissel_Radiative_Cascade)
overload(
    _xraylib.CSb_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl.get("CSb_FluorShell_Kissel_Radiative_Cascade", {}),
)(_CSb_FluorShell_Kissel_Radiative_Cascade)


@overload(
    xraylib_np.CSb_FluorShell_Kissel_Radiative_Cascade,
    jit_options=config.xrl_np.get("CSb_FluorShell_Kissel_Radiative_Cascade", {}),
)
def _CSb_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell_Kissel_Radiative_Cascade", 2, 1)
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


def _CS_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorLine", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine, jit_options=config.xrl.get("CS_FluorLine", {}))(
    _CS_FluorLine,
)
overload(_xraylib.CS_FluorLine, jit_options=config.xrl.get("CS_FluorLine", {}))(
    _CS_FluorLine,
)


@overload(xraylib_np.CS_FluorLine, jit_options=config.xrl_np.get("CS_FluorLine", {}))
def _CS_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CS_FluorLine", 2, 1)
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


def _CSb_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorLine", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_LINE} | {INVALID_LINE}"

    def impl(Z, line, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, line, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine, jit_options=config.xrl.get("CSb_FluorLine", {}))(
    _CSb_FluorLine,
)
overload(_xraylib.CSb_FluorLine, jit_options=config.xrl.get("CSb_FluorLine", {}))(
    _CSb_FluorLine,
)


@overload(xraylib_np.CSb_FluorLine, jit_options=config.xrl_np.get("CSb_FluorLine", {}))
def _CSb_FluorLine_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    _check_ndim(Z, line, E)
    xrl_fcn = _nint_ndouble("CSb_FluorLine", 2, 1)
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


def _CS_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_FluorShell", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell, jit_options=config.xrl.get("CS_FluorShell", {}))(
    _CS_FluorShell,
)
overload(_xraylib.CS_FluorShell, jit_options=config.xrl.get("CS_FluorShell", {}))(
    _CS_FluorShell,
)


@overload(xraylib_np.CS_FluorShell, jit_options=config.xrl_np.get("CS_FluorShell", {}))
def _CS_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_FluorShell", 2, 1)
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


def _CSb_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_FluorShell", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell, jit_options=config.xrl.get("CSb_FluorShell", {}))(
    _CSb_FluorShell,
)
overload(_xraylib.CSb_FluorShell, jit_options=config.xrl.get("CSb_FluorShell", {}))(
    _CSb_FluorShell,
)


@overload(
    xraylib_np.CSb_FluorShell,
    jit_options=config.xrl_np.get("CSb_FluorShell", {}),
)
def _CSb_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_FluorShell", 2, 1)
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


def _CS_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CS_Photo_Partial", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Photo_Partial, jit_options=config.xrl.get("CS_Photo_Partial", {}))(
    _CS_Photo_Partial,
)
overload(_xraylib.CS_Photo_Partial, jit_options=config.xrl.get("CS_Photo_Partial", {}))(
    _CS_Photo_Partial,
)


@overload(
    xraylib_np.CS_Photo_Partial,
    jit_options=config.xrl_np.get("CS_Photo_Partial", {}),
)
def _CS_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CS_Photo_Partial", 2, 1)
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


def _CSb_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    xrl_fcn = _nint_ndouble("CSb_Photo_Partial", 2, 1)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY} | {UNKNOWN_SHELL} | {INVALID_SHELL}"

    def impl(Z, shell, E):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, shell, E, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.CSb_Photo_Partial,
    jit_options=config.xrl.get("CSb_Photo_Partial", {}),
)(_CSb_Photo_Partial)
overload(
    _xraylib.CSb_Photo_Partial,
    jit_options=config.xrl.get("CSb_Photo_Partial", {}),
)(_CSb_Photo_Partial)


@overload(
    xraylib_np.CSb_Photo_Partial,
    jit_options=config.xrl_np.get("CSb_Photo_Partial", {}),
)
def _CSb_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    _check_ndim(Z, shell, E)
    xrl_fcn = _nint_ndouble("CSb_Photo_Partial", 2, 1)
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


def _DCS_Compt(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    xrl_fcn = _nint_ndouble("DCS_Compt", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCS_Compt, jit_options=config.xrl.get("DCS_Compt", {}))(_DCS_Compt)
overload(_xraylib.DCS_Compt, jit_options=config.xrl.get("DCS_Compt", {}))(_DCS_Compt)


@overload(xraylib_np.DCS_Compt, jit_options=config.xrl_np.get("DCS_Compt", {}))
def _DCS_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _nint_ndouble("DCS_Compt", 1, 2)
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


def _DCS_Rayl(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    xrl_fcn = _nint_ndouble("DCS_Rayl", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCS_Rayl, jit_options=config.xrl.get("DCS_Rayl", {}))(_DCS_Rayl)
overload(_xraylib.DCS_Rayl, jit_options=config.xrl.get("DCS_Rayl", {}))(_DCS_Rayl)


@overload(xraylib_np.DCS_Rayl, jit_options=config.xrl_np.get("DCS_Rayl", {}))
def _DCS_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _nint_ndouble("DCS_Rayl", 1, 2)
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


def _DCSb_Compt(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    xrl_fcn = _nint_ndouble("DCSb_Compt", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSb_Compt, jit_options=config.xrl.get("DCSb_Compt", {}))(_DCSb_Compt)
overload(_xraylib.DCSb_Compt, jit_options=config.xrl.get("DCSb_Compt", {}))(_DCSb_Compt)


@overload(xraylib_np.DCSb_Compt, jit_options=config.xrl_np.get("DCSb_Compt", {}))
def _DCSb_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _nint_ndouble("DCSb_Compt", 1, 2)
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


def _DCSb_Rayl(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    xrl_fcn = _nint_ndouble("DCSb_Rayl", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSb_Rayl, jit_options=config.xrl.get("DCSb_Rayl", {}))(_DCSb_Rayl)
overload(_xraylib.DCSb_Rayl, jit_options=config.xrl.get("DCSb_Rayl", {}))(_DCSb_Rayl)


@overload(xraylib_np.DCSb_Rayl, jit_options=config.xrl_np.get("DCSb_Rayl", {}))
def _DCSb_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    _check_ndim(Z, E, theta)
    xrl_fcn = _nint_ndouble("DCSb_Rayl", 1, 2)
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
def _PL1_auger_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    xrl_fcn = _nint_ndouble("PL1_auger_cascade_kissel", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL1_auger_cascade_kissel,
    jit_options=config.xrl.get("PL1_auger_cascade_kissel", {}),
)(_PL1_auger_cascade_kissel)

overload(
    _xraylib.PL1_auger_cascade_kissel,
    jit_options=config.xrl.get("PL1_auger_cascade_kissel", {}),
)(_PL1_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL1_full_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    xrl_fcn = _nint_ndouble("PL1_full_cascade_kissel", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL1_full_cascade_kissel,
    jit_options=config.xrl.get("PL1_full_cascade_kissel", {}),
)(_PL1_full_cascade_kissel)
overload(
    _xraylib.PL1_full_cascade_kissel,
    jit_options=config.xrl.get("PL1_full_cascade_kissel", {}),
)(_PL1_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL1_rad_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    xrl_fcn = _nint_ndouble("PL1_rad_cascade_kissel", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL1_rad_cascade_kissel,
    jit_options=config.xrl.get("PL1_rad_cascade_kissel", {}),
)(_PL1_rad_cascade_kissel)
overload(
    _xraylib.PL1_rad_cascade_kissel,
    jit_options=config.xrl.get("PL1_rad_cascade_kissel", {}),
)(_PL1_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_pure_kissel(Z, E, PL1):
    _check_types((Z, E, PL1), 1, 2)
    xrl_fcn = _nint_ndouble("PL2_pure_kissel", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL2_pure_kissel, jit_options=config.xrl.get("PL2_pure_kissel", {}))(
    _PL2_pure_kissel,
)
overload(_xraylib.PL2_pure_kissel, jit_options=config.xrl.get("PL2_pure_kissel", {}))(
    _PL2_pure_kissel,
)


# !!! Not implemented in xraylib_np
def _PM2_pure_kissel(Z, E, PM1):
    _check_types((Z, E, PM1), 1, 2)
    xrl_fcn = _nint_ndouble("PM2_pure_kissel", 1, 2)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PM1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PM1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM2_pure_kissel, jit_options=config.xrl.get("PM2_pure_kissel", {}))(
    _PM2_pure_kissel,
)
overload(_xraylib.PM2_pure_kissel, jit_options=config.xrl.get("PM2_pure_kissel", {}))(
    _PM2_pure_kissel,
)


# ---------------------------------- 1 int, 3 double --------------------------------- #


def _DCSP_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("DCSP_Rayl", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_Rayl, jit_options=config.xrl.get("DCSP_Rayl", {}))(_DCSP_Rayl)
overload(_xraylib.DCSP_Rayl, jit_options=config.xrl.get("DCSP_Rayl", {}))(_DCSP_Rayl)


@overload(xraylib_np.DCSP_Rayl, jit_options=config.xrl_np.get("DCSP_Rayl", {}))
def _DCSP_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _nint_ndouble("DCSP_Rayl", 1, 3)
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


def _DCSP_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("DCSP_Compt", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_Compt, jit_options=config.xrl.get("DCSP_Compt", {}))(_DCSP_Compt)
overload(_xraylib.DCSP_Compt, jit_options=config.xrl.get("DCSP_Compt", {}))(_DCSP_Compt)


@overload(xraylib_np.DCSP_Compt, jit_options=config.xrl_np.get("DCSP_Compt", {}))
def _DCSP_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _nint_ndouble("DCSP_Compt", 1, 3)
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


def _DCSPb_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("DCSPb_Rayl", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSPb_Rayl, jit_options=config.xrl.get("DCSPb_Rayl", {}))(_DCSPb_Rayl)
overload(_xraylib.DCSPb_Rayl, jit_options=config.xrl.get("DCSPb_Rayl", {}))(_DCSPb_Rayl)


@overload(xraylib_np.DCSPb_Rayl, jit_options=config.xrl_np.get("DCSPb_Rayl", {}))
def _DCSPb_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _nint_ndouble("DCSPb_Rayl", 1, 3)
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


def _DCSPb_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("DCSPb_Compt", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSPb_Compt, jit_options=config.xrl.get("DCSPb_Compt", {}))(
    _DCSPb_Compt,
)
overload(_xraylib.DCSPb_Compt, jit_options=config.xrl.get("DCSPb_Compt", {}))(
    _DCSPb_Compt,
)


@overload(xraylib_np.DCSPb_Compt, jit_options=config.xrl_np.get("DCSPb_Compt", {}))
def _DCSPb_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    _check_ndim(Z, E, theta, phi)
    xrl_fcn = _nint_ndouble("DCSPb_Compt", 1, 3)
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
def _PL2_auger_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("PL2_auger_cascade_kissel", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL2_auger_cascade_kissel,
    jit_options=config.xrl.get("PL2_auger_cascade_kissel", {}),
)(_PL2_auger_cascade_kissel)
overload(
    _xraylib.PL2_auger_cascade_kissel,
    jit_options=config.xrl.get("PL2_auger_cascade_kissel", {}),
)(_PL2_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_full_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    xrl_fcn = _nint_ndouble("PL2_full_cascade_kissel", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL2_full_cascade_kissel,
    jit_options=config.xrl.get("PL2_full_cascade_kissel", {}),
)(_PL2_full_cascade_kissel)
overload(
    _xraylib.PL2_full_cascade_kissel,
    jit_options=config.xrl.get("PL2_full_cascade_kissel", {}),
)(_PL2_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_rad_cascade_kissel(Z, E, PK, PL1):
    _check_types((Z, E, PK, PL1), 1, 3)
    xrl_fcn = _nint_ndouble("PL2_rad_cascade_kissel", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL2_rad_cascade_kissel,
    jit_options=config.xrl.get("PL2_rad_cascade_kissel", {}),
)(_PL2_rad_cascade_kissel)
overload(
    _xraylib.PL2_rad_cascade_kissel,
    jit_options=config.xrl.get("PL2_rad_cascade_kissel", {}),
)(_PL2_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_pure_kissel(Z, E, PL1, PL2):
    _check_types((Z, E, PL1, PL2), 1, 3)
    xrl_fcn = _nint_ndouble("PL3_pure_kissel", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PL1, PL2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PL1, PL2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL3_pure_kissel, jit_options=config.xrl.get("PL3_pure_kissel", {}))(
    _PL3_pure_kissel,
)
overload(_xraylib.PL3_pure_kissel, jit_options=config.xrl.get("PL3_pure_kissel", {}))(
    _PL3_pure_kissel,
)


# !!! Not implemented in xraylib_np
def _PM3_pure_kissel(Z, E, PM1, PM2):
    _check_types((Z, E, PM1, PM2), 1, 3)
    xrl_fcn = _nint_ndouble("PM3_pure_kissel", 1, 3)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM3_pure_kissel, jit_options=config.xrl.get("PM3_pure_kissel", {}))(
    _PM3_pure_kissel,
)
overload(_xraylib.PM3_pure_kissel, jit_options=config.xrl.get("PM3_pure_kissel", {}))(
    _PM3_pure_kissel,
)


# ---------------------------------- 1 int, 4 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PL3_auger_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), 1, 4)
    xrl_fcn = _nint_ndouble("PL3_auger_cascade_kissel", 1, 4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL3_auger_cascade_kissel,
    jit_options=config.xrl.get("PL3_auger_cascade_kissel", {}),
)(_PL3_auger_cascade_kissel)
overload(
    _xraylib.PL3_auger_cascade_kissel,
    jit_options=config.xrl.get("PL3_auger_cascade_kissel", {}),
)(_PL3_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_full_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), 1, 4)
    xrl_fcn = _nint_ndouble("PL3_full_cascade_kissel", 1, 4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PL1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PL1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL3_full_cascade_kissel,
    jit_options=config.xrl.get("PL3_full_cascade_kissel", {}),
)(_PL3_full_cascade_kissel)
overload(
    _xraylib.PL3_full_cascade_kissel,
    jit_options=config.xrl.get("PL3_full_cascade_kissel", {}),
)(_PL3_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2):
    _check_types((Z, E, PK, PL1, PL2), 1, 4)
    xrl_fcn = _nint_ndouble("PL3_rad_cascade_kissel", 1, 4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PL3_rad_cascade_kissel,
    jit_options=config.xrl.get("PL3_rad_cascade_kissel", {}),
)(_PL3_rad_cascade_kissel)
overload(
    _xraylib.PL3_rad_cascade_kissel,
    jit_options=config.xrl.get("PL3_rad_cascade_kissel", {}),
)(_PL3_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_pure_kissel(Z, E, theta, phi, PM1):
    _check_types((Z, E, theta, phi, PM1), 1, 4)
    xrl_fcn = _nint_ndouble("PM4_pure_kissel", 1, 4)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM1):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM1, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM4_pure_kissel, jit_options=config.xrl.get("PM4_pure_kissel", {}))(
    _PM4_pure_kissel,
)
overload(_xraylib.PM4_pure_kissel, jit_options=config.xrl.get("PM4_pure_kissel", {}))(
    _PM4_pure_kissel,
)


# ---------------------------------- 1 int, 5 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM1_auger_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), 1, 5)
    xrl_fcn = _nint_ndouble("PM1_auger_cascade_kissel", 1, 5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM1_auger_cascade_kissel,
    jit_options=config.xrl.get("PM1_auger_cascade_kissel", {}),
)(_PM1_auger_cascade_kissel)
overload(
    _xraylib.PM1_auger_cascade_kissel,
    jit_options=config.xrl.get("PM1_auger_cascade_kissel", {}),
)(_PM1_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM1_full_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), 1, 5)
    xrl_fcn = _nint_ndouble("PM1_full_cascade_kissel", 1, 5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM1_full_cascade_kissel,
    jit_options=config.xrl.get("PM1_full_cascade_kissel", {}),
)(_PM1_full_cascade_kissel)
overload(
    _xraylib.PM1_full_cascade_kissel,
    jit_options=config.xrl.get("PM1_full_cascade_kissel", {}),
)(_PM1_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM1_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3):
    _check_types((Z, E, PK, PL, PL2, PL3), 1, 5)
    xrl_fcn = _nint_ndouble("PM1_rad_cascade_kissel", 1, 5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL, PL2, PL3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM1_rad_cascade_kissel,
    jit_options=config.xrl.get("PM1_rad_cascade_kissel", {}),
)(_PM1_rad_cascade_kissel)
overload(
    _xraylib.PM1_rad_cascade_kissel,
    jit_options=config.xrl.get("PM1_rad_cascade_kissel", {}),
)(_PM1_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_pure_kissel(Z, E, theta, phi, PM1, PM2):
    _check_types((Z, E, theta, phi, PM1, PM2), 1, 5)
    xrl_fcn = _nint_ndouble("PM5_pure_kissel", 1, 5)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM5_pure_kissel, jit_options=config.xrl.get("PM5_pure_kissel", {}))(
    _PM5_pure_kissel,
)
overload(_xraylib.PM5_pure_kissel, jit_options=config.xrl.get("PM5_pure_kissel", {}))(
    _PM5_pure_kissel,
)


# ---------------------------------- 1 int, 6 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM2_auger_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), 1, 6)
    xrl_fcn = _nint_ndouble("PM2_auger_cascade_kissel", 1, 6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM2_auger_cascade_kissel,
    jit_options=config.xrl.get("PM2_auger_cascade_kissel", {}),
)(_PM2_auger_cascade_kissel)
overload(
    _xraylib.PM2_auger_cascade_kissel,
    jit_options=config.xrl.get("PM2_auger_cascade_kissel", {}),
)(_PM2_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM2_full_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), 1, 6)
    xrl_fcn = _nint_ndouble("PM2_full_cascade_kissel", 1, 6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM2_full_cascade_kissel,
    jit_options=config.xrl.get("PM2_full_cascade_kissel", {}),
)(_PM2_full_cascade_kissel)
overload(
    _xraylib.PM2_full_cascade_kissel,
    jit_options=config.xrl.get("PM2_full_cascade_kissel", {}),
)(_PM2_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM2_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, PL4):
    _check_types((Z, E, PK, PL, PL2, PL3, PL4), 1, 6)
    xrl_fcn = _nint_ndouble("PM2_rad_cascade_kissel", 1, 6)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL, PL2, PL3, PL4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, PL4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM2_rad_cascade_kissel,
    jit_options=config.xrl.get("PM2_rad_cascade_kissel", {}),
)(_PM2_rad_cascade_kissel)
overload(
    _xraylib.PM2_rad_cascade_kissel,
    jit_options=config.xrl.get("PM2_rad_cascade_kissel", {}),
)(_PM2_rad_cascade_kissel)


# ---------------------------------- 1 int, 7 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    xrl_fcn = _nint_ndouble("PM3_auger_cascade_kissel", 1, 7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM3_auger_cascade_kissel,
    jit_options=config.xrl.get("PM3_auger_cascade_kissel", {}),
)(_PM3_auger_cascade_kissel)
overload(
    _xraylib.PM3_auger_cascade_kissel,
    jit_options=config.xrl.get("PM3_auger_cascade_kissel", {}),
)(_PM3_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    xrl_fcn = _nint_ndouble("PM3_full_cascade_kissel", 1, 7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM3_full_cascade_kissel,
    jit_options=config.xrl.get("PM3_full_cascade_kissel", {}),
)(_PM3_full_cascade_kissel)
overload(
    _xraylib.PM3_full_cascade_kissel,
    jit_options=config.xrl.get("PM3_full_cascade_kissel", {}),
)(_PM3_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    xrl_fcn = _nint_ndouble("PM3_rad_cascade_kissel", 1, 7)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM3_rad_cascade_kissel,
    jit_options=config.xrl.get("PM3_rad_cascade_kissel", {}),
)(_PM3_rad_cascade_kissel)
overload(
    _xraylib.PM3_rad_cascade_kissel,
    jit_options=config.xrl.get("PM3_rad_cascade_kissel", {}),
)(_PM3_rad_cascade_kissel)


# ---------------------------------- 1 int, 8 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    xrl_fcn = _nint_ndouble("PM4_auger_cascade_kissel", 1, 8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM4_auger_cascade_kissel,
    jit_options=config.xrl.get("PM4_auger_cascade_kissel", {}),
)(_PM4_auger_cascade_kissel)
overload(
    _xraylib.PM4_auger_cascade_kissel,
    jit_options=config.xrl.get("PM4_auger_cascade_kissel", {}),
)(_PM4_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    xrl_fcn = _nint_ndouble("PM4_full_cascade_kissel", 1, 8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM4_full_cascade_kissel,
    jit_options=config.xrl.get("PM4_full_cascade_kissel", {}),
)(_PM4_full_cascade_kissel)
overload(
    _xraylib.PM4_full_cascade_kissel,
    jit_options=config.xrl.get("PM4_full_cascade_kissel", {}),
)(_PM4_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    xrl_fcn = _nint_ndouble("PM4_rad_cascade_kissel", 1, 8)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM4_rad_cascade_kissel,
    jit_options=config.xrl.get("PM4_rad_cascade_kissel", {}),
)(_PM4_rad_cascade_kissel)
overload(
    _xraylib.PM4_rad_cascade_kissel,
    jit_options=config.xrl.get("PM4_rad_cascade_kissel", {}),
)(_PM4_rad_cascade_kissel)


# ---------------------------------- 1 int, 9 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    xrl_fcn = _nint_ndouble("PM5_auger_cascade_kissel", 1, 9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM5_auger_cascade_kissel,
    jit_options=config.xrl.get("PM5_auger_cascade_kissel", {}),
)(_PM5_auger_cascade_kissel)
overload(
    _xraylib.PM5_auger_cascade_kissel,
    jit_options=config.xrl.get("PM5_auger_cascade_kissel", {}),
)(_PM5_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    xrl_fcn = _nint_ndouble("PM5_full_cascade_kissel", 1, 9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM5_full_cascade_kissel,
    jit_options=config.xrl.get("PM5_full_cascade_kissel", {}),
)(_PM5_full_cascade_kissel)
overload(
    _xraylib.PM5_full_cascade_kissel,
    jit_options=config.xrl.get("PM5_full_cascade_kissel", {}),
)(_PM5_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    xrl_fcn = _nint_ndouble("PM5_rad_cascade_kissel", 1, 9)
    msg = f"{Z_OUT_OF_RANGE} | {NEGATIVE_ENERGY}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(
    xraylib.PM5_rad_cascade_kissel,
    jit_options=config.xrl.get("PM5_rad_cascade_kissel", {}),
)(_PM5_rad_cascade_kissel)
overload(
    _xraylib.PM5_rad_cascade_kissel,
    jit_options=config.xrl.get("PM5_rad_cascade_kissel", {}),
)(_PM5_rad_cascade_kissel)


# ------------------------------------- 3 double ------------------------------------- #


def _DCSP_KN(E, theta, phi):
    _check_types((E, theta, phi), 0, 3)
    xrl_fcn = _nint_ndouble("DCSP_KN", 0, 3)
    msg = f"{NEGATIVE_ENERGY}"

    def impl(E, theta, phi):
        error = array([0, 0], dtype=int32)
        result = xrl_fcn(E, theta, phi, error.ctypes)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_KN, jit_options=config.xrl.get("DCSP_KN", {}))(_DCSP_KN)
overload(_xraylib.DCSP_KN, jit_options=config.xrl.get("DCSP_KN", {}))(_DCSP_KN)


@overload(xraylib_np.DCSP_KN, jit_options=config.xrl_np.get("DCSP_KN", {}))
def _DCSP_KN_np(E, theta, phi):
    _check_types((E, theta, phi), 0, 3, _np=True)
    _check_ndim(E, theta, phi)
    xrl_fcn = _nint_ndouble("DCSP_KN", 0, 3)
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

# TODO(nin17): CS_Total_CP
# TODO(nin17): CS_Photo_CP
# TODO(nin17): CS_Rayl_CP
# TODO(nin17): CS_Compt_CP
# TODO(nin17): CS_Energy_CP
# TODO(nin17): CS_Photo_Total_CP
# TODO(nin17): CS_Total_Kissel_CP
# TODO(nin17): CSb_Total_CP
# TODO(nin17): CSb_Photo_CP
# TODO(nin17): CSb_Rayl_CP
# TODO(nin17): CSb_Compt_CP
# TODO(nin17): CSb_Energy_CP
# TODO(nin17): CSb_Photo_Total_CP
# TODO(nin17): CSb_Total_Kissel_CP


# -------------------------------- 1 string, 2 double -------------------------------- #

# TODO(nin17): DCS_Rayl_CP
# TODO(nin17): DCS_Compt_CP
# TODO(nin17): DCSb_Rayl_CP
# TODO(nin17): DCSb_Compt_CP
# TODO(nin17): Refractive_Index_Re
# TODO(nin17): Refractive_Index_Im
# TODO(nin17): Refractive_Index

# -------------------------------- 1 string, 3 double -------------------------------- #

# TODO(nin17): DCSP_Rayl_CP
# TODO(nin17): DCSP_Compt_CP
# TODO(nin17): DCSPb_Rayl_CP
# TODO(nin17): DCSPb_Compt_CP

# TODO(nin17): Other functions with string returns etc...
