"""
Overloads for xraylib functions to work with numba.
"""
# pylint: disable=c-extension-no-member, too-many-lines, not-an-iterable
# TODO check variable names are the same as in the xraylib functions
# TODO jit_options


from ctypes.util import find_library

import _xraylib
import numpy as np
import xraylib_np
from llvmlite import binding
from numba import errors, types, vectorize
from numba.extending import overload

import xraylib

from . import config

binding.load_library_permanently(find_library("xrl"))


def _nint_ndouble(name: str, nint: int = 0, ndouble: int = 0) -> types.ExternalFunction:
    """_summary_

    Parameters
    ----------
    name : str
        _description_
    nint : int, optional
        _description_, by default 0
    ndouble : int, optional
        _description_, by default 0

    Returns
    -------
    types.ExternalFunction
        _description_
    """
    argtypes = [types.int32 for _ in range(nint)] + [
        types.float64 for _ in range(ndouble)
    ]
    sig = types.float64(*argtypes, types.voidptr)
    return types.ExternalFunction(name, sig)


def _check_types(args, nint=0, ndouble=0, _np=False):
    if _np:
        for i in range(nint):
            if not isinstance(args[i].dtype, types.Integer):
                raise errors.NumbaTypeError(f"Expected got {args[i].dtype}")
        for i in range(nint, nint + ndouble):
            if not args[i].dtype is types.float64:
                raise errors.NumbaTypeError(f"Expected got {args[i].dtype}")
    else:
        for i in range(nint):
            if not isinstance(args[i], types.Integer):
                raise errors.NumbaTypeError(f"Expected {types.Integer} got {args[i]}")
        for i in range(nint, nint + ndouble):
            if not args[i] is types.float64:
                raise errors.NumbaTypeError(f"Expected {types.float64} got {args[i]}")


# @extending.register_jitable
# def _parser(compound):
#     with objmode(elements="int64[:]", mass_fractions="float64[:]", n_elements="int64"):
#         parsed_compound = xraylib.CompoundParser(compound)
#         elements = np.array(parsed_compound["Elements"])
#         mass_fractions = np.array(parsed_compound["massFractions"])
#         n_elements = parsed_compound["nElements"]

#     return elements, mass_fractions, n_elements


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

# --------------------------------------- 1 int -------------------------------------- #


def _AtomicWeight(Z):
    _check_types((Z,), 1)
    aw = _nint_ndouble("AtomicWeight", 1)

    def impl(Z):
        error = np.array([0, 0], dtype=np.int32)
        result = aw(Z, error.ctypes.data)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


overload(xraylib.AtomicWeight)(_AtomicWeight)
overload(_xraylib.AtomicWeight)(_AtomicWeight)


@overload(xraylib_np.AtomicWeight)
def _AtomicWeight_np(Z):
    _check_types((Z,), 1, _np=True)
    aw = _nint_ndouble("AtomicWeight", 1)

    @vectorize
    def _impl(Z):
        return aw(Z, 0)

    def impl(Z):
        if not config.allow_Nd:
            assert Z.ndim == 1
        return _impl(Z)

    return impl


def _ElementDensity(Z):
    _check_types((Z,), 1)
    ed = _nint_ndouble("ElementDensity", 1)

    def impl(Z):
        error = np.array([0, 0], dtype=np.int32)
        result = ed(Z, error.ctypes.data)
        if error.any():
            raise ValueError(Z_OUT_OF_RANGE)
        return result

    return impl


overload(xraylib.ElementDensity)(_ElementDensity)
overload(_xraylib.ElementDensity)(_ElementDensity)


@overload(xraylib_np.ElementDensity)
def _ElementDensity_np(Z):
    _check_types((Z,), 1, _np=True)
    ed = _nint_ndouble("ElementDensity", 1)

    @vectorize
    def _impl(Z):
        return ed(Z, 0)

    def impl(Z):
        if not config.allow_Nd:
            assert Z.ndim == 1
        return _impl(Z)

    return impl


# ------------------------------------- 1 double ------------------------------------- #


def _CS_KN(E):
    _check_types((E,), 0, 1)
    cs_kn = _nint_ndouble("CS_KN", 0, 1)

    def impl(E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_kn(E, error.ctypes.data)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.CS_KN)(_CS_KN)
overload(_xraylib.CS_KN)(_CS_KN)


@overload(xraylib_np.CS_KN)
def _CS_KN_np(E):
    _check_types((E,), 0, 1, _np=True)
    cs_kn = _nint_ndouble("CS_KN", 0, 1)

    @vectorize
    def _impl(E):
        return cs_kn(E, 0)

    def impl(E):
        if not config.allow_Nd:
            assert E.ndim == 1
        return _impl(E)

    return impl


def _DCS_Thoms(theta):
    _check_types((theta,), 0, 1)
    dcs_thoms = _nint_ndouble("DCS_Thoms", 0, 1)

    def impl(theta):
        return dcs_thoms(theta, 0)

    return impl


overload(xraylib.DCS_Thoms)(_DCS_Thoms)
overload(_xraylib.DCS_Thoms)(_DCS_Thoms)


@overload(xraylib_np.DCS_Thoms)
def _DCS_Thoms_np(theta):
    _check_types((theta,), 0, 1, _np=True)
    dcs_thoms = _nint_ndouble("DCS_Thoms", 0, 1)

    @vectorize
    def _impl(theta):
        return dcs_thoms(theta, 0)

    def impl(theta):
        if not config.allow_Nd:
            assert theta.ndim == 1
        return _impl(theta)

    return impl


# --------------------------------------- 2 int -------------------------------------- #


def _AtomicLevelWidth(Z, shell):
    _check_types((Z, shell), 2)
    alw = _nint_ndouble("AtomicLevelWidth", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = alw(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AtomicLevelWidth)(_AtomicLevelWidth)
overload(_xraylib.AtomicLevelWidth)(_AtomicLevelWidth)


@overload(xraylib_np.AtomicLevelWidth)
def _AtomicLevelWidth_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    alw = _nint_ndouble("AtomicLevelWidth", 2)

    @vectorize
    def _impl(Z, shell):
        return alw(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _AugerRate(Z, auger_trans):
    _check_types((Z, auger_trans), 2)
    ar = _nint_ndouble("AugerRate", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_AUGER, INVALID_AUGER))

    def impl(Z, auger_trans):
        error = np.array([0, 0], dtype=np.int32)
        result = ar(Z, auger_trans, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AugerRate)(_AugerRate)
overload(_xraylib.AugerRate)(_AugerRate)


@overload(xraylib_np.AugerRate)
def _AugerRate_np(Z, auger_trans):
    _check_types((Z, auger_trans), 2, _np=True)
    ar = _nint_ndouble("AugerRate", 2)

    @vectorize
    def _impl(Z, auger_trans):
        return ar(Z, auger_trans, 0)

    def impl(Z, auger_trans):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert auger_trans.ndim == 1
        return _impl(Z[..., None], auger_trans[None, ...])

    return impl


def _AugerYield(Z, shell):
    _check_types((Z, shell), 2)
    ay = _nint_ndouble("AugerYield", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = ay(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.AugerYield)(_AugerYield)
overload(_xraylib.AugerYield)(_AugerYield)


@overload(xraylib_np.AugerYield)
def _AugerYield_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    ay = _nint_ndouble("AugerYield", 2)

    @vectorize
    def _impl(Z, shell):
        return ay(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _CosKronTransProb(Z, trans):
    _check_types((Z, trans), 2)
    cktp = _nint_ndouble("CosKronTransProb", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_CK, INVALID_CK))

    def impl(Z, trans):
        error = np.array([0, 0], dtype=np.int32)
        result = cktp(Z, trans, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CosKronTransProb)(_CosKronTransProb)
overload(_xraylib.CosKronTransProb)(_CosKronTransProb)


@overload(xraylib_np.CosKronTransProb)
def _CosKronTransProb_np(Z, trans):
    _check_types((Z, trans), 2, _np=True)
    cktp = _nint_ndouble("CosKronTransProb", 2)

    @vectorize
    def _impl(Z, trans):
        return cktp(Z, trans, 0)

    def impl(Z, trans):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert trans.ndim == 1
        return _impl(Z[..., None], trans[None, ...])

    return impl


def _EdgeEnergy(Z, shell):
    _check_types((Z, shell), 2)
    ee = _nint_ndouble("EdgeEnergy", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = ee(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.EdgeEnergy)(_EdgeEnergy)
overload(_xraylib.EdgeEnergy)(_EdgeEnergy)


@overload(xraylib_np.EdgeEnergy)
def _EdgeEnergy_np(Z, shell):  # !!! change to shell
    _check_types((Z, shell), 2, _np=True)
    ee = _nint_ndouble("EdgeEnergy", 2)

    @vectorize
    def _impl(Z, shell):
        return ee(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _ElectronConfig(Z, shell):
    _check_types((Z, shell), 2)
    ec = _nint_ndouble("ElectronConfig", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = ec(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ElectronConfig)(_ElectronConfig)
overload(_xraylib.ElectronConfig)(_ElectronConfig)


@overload(xraylib_np.ElectronConfig)
def _ElectronConfig_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    ec = _nint_ndouble("ElectronConfig", 2)

    @vectorize
    def _impl(Z, shell):
        return ec(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _FluorYield(Z, shell):
    _check_types((Z, shell), 2)
    fy = _nint_ndouble("FluorYield", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = fy(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.FluorYield)(_FluorYield)
overload(_xraylib.FluorYield)(_FluorYield)


@overload(xraylib_np.FluorYield)
def _FluorYield_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    fy = _nint_ndouble("FluorYield", 2)

    @vectorize
    def _impl(Z, shell):
        return fy(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _JumpFactor(Z, shell):
    _check_types((Z, shell), 2)
    jf = _nint_ndouble("JumpFactor", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell):
        error = np.array([0, 0], dtype=np.int32)
        result = jf(Z, shell, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.JumpFactor)(_JumpFactor)
overload(_xraylib.JumpFactor)(_JumpFactor)


@overload(xraylib_np.JumpFactor)
def _JumpFactor_np(Z, shell):
    _check_types((Z, shell), 2, _np=True)
    jf = _nint_ndouble("JumpFactor", 2)

    @vectorize
    def _impl(Z, shell):
        return jf(Z, shell, 0)

    def impl(Z, shell):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
        return _impl(Z[..., None], shell[None, ...])

    return impl


def _LineEnergy(Z, line):
    _check_types((Z, line), 2)
    le = _nint_ndouble("LineEnergy", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line):
        error = np.array([0, 0], dtype=np.int32)
        result = le(Z, line, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.LineEnergy)(_LineEnergy)
overload(_xraylib.LineEnergy)(_LineEnergy)


@overload(xraylib_np.LineEnergy)
def _LineEnergy_np(Z, line):
    _check_types((Z, line), 2, _np=True)
    le = _nint_ndouble("LineEnergy", 2)

    @vectorize
    def _impl(Z, line):
        return le(Z, line, 0)

    def impl(Z, line):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
        return _impl(Z[..., None], line[None, ...])

    return impl


def _RadRate(Z, line):
    _check_types((Z, line), 2)
    rr = _nint_ndouble("RadRate", 2)
    msg = " | ".join((Z_OUT_OF_RANGE, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line):
        error = np.array([0, 0], dtype=np.int32)
        result = rr(Z, line, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.RadRate)(_RadRate)
overload(_xraylib.RadRate)(_RadRate)


@overload(xraylib_np.RadRate)
def _RadRate_np(Z, line):
    _check_types((Z, line), 2, _np=True)
    rr = _nint_ndouble("RadRate", 2)

    @vectorize
    def _impl(Z, line):
        return rr(Z, line, 0)

    def impl(Z, line):
        if config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
        return _impl(Z[..., None], line[None, ...])

    return impl


# ------------------------------------- 2 double ------------------------------------- #


def _ComptonEnergy(E0, theta):
    _check_types((E0, theta), 0, 2)
    ce = _nint_ndouble("ComptonEnergy", 0, 2)
    msg = NEGATIVE_ENERGY

    def impl(E0, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = ce(E0, theta, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ComptonEnergy)(_ComptonEnergy)
overload(_xraylib.ComptonEnergy)(_ComptonEnergy)


@overload(xraylib_np.ComptonEnergy)
def _ComptonEnergy_np(E0, theta):
    _check_types((E0, theta), 0, 2, _np=True)
    ce = _nint_ndouble("ComptonEnergy", 0, 2)

    @vectorize
    def _impl(E0, theta):
        return ce(E0, theta, 0)

    def impl(E0, theta):
        if not config.allow_Nd:
            assert E0.ndim == 1
            assert theta.ndim == 1
        return _impl(E0[..., None], theta[None, ...])

    return impl


def _DCS_KN(E, theta):
    _check_types((E, theta), 0, 2)
    dcs_kn = _nint_ndouble("DCS_KN", 0, 2)

    def impl(E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = dcs_kn(E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.DCS_KN)(_DCS_KN)
overload(_xraylib.DCS_KN)(_DCS_KN)


@overload(xraylib_np.DCS_KN)
def _DCS_KN_np(E, theta):
    _check_types((E, theta), 0, 2, _np=True)
    dcs_kn = _nint_ndouble("DCS_KN", 0, 2)

    @vectorize
    def _impl(E, theta):
        return dcs_kn(E, theta, 0)

    def impl(E, theta):
        if not config.allow_Nd:
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(E[..., None], theta[None, ...])

    return impl


def _DCSP_Thoms(theta, phi):
    _check_types((theta, phi), 0, 2)
    dcsp_thoms = _nint_ndouble("DCSP_Thoms", 0, 2)

    def impl(theta, phi):
        return dcsp_thoms(theta, phi, 0)

    return impl


overload(xraylib.DCSP_Thoms)(_DCSP_Thoms)
overload(_xraylib.DCSP_Thoms)(_DCSP_Thoms)


@overload(xraylib_np.DCSP_Thoms)
def _DCSP_Thoms_np(theta, phi):
    _check_types((theta, phi), 0, 2, _np=True)
    dcsp_thoms = _nint_ndouble("DCSP_Thoms", 0, 2)

    @vectorize
    def _impl(theta, phi):
        return dcsp_thoms(theta, phi, 0)

    def impl(theta, phi):
        if not config.allow_Nd:
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(theta[..., None], phi[None, ...])

    return impl


def _MomentTransf(E, theta):
    _check_types((E, theta), 0, 2)
    mt = _nint_ndouble("MomentTransf", 0, 2)

    def impl(E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = mt(E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(NEGATIVE_ENERGY)
        return result

    return impl


overload(xraylib.MomentTransf)(_MomentTransf)
overload(_xraylib.MomentTransf)(_MomentTransf)


@overload(xraylib_np.MomentTransf)
def _MomentTransf_np(E, theta):
    _check_types((E, theta), 0, 2, _np=True)
    mt = _nint_ndouble("MomentTransf", 0, 2)

    @vectorize
    def _impl(E, theta):
        return mt(E, theta, 0)

    def impl(E, theta):
        if not config.allow_Nd:
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(E[..., None], theta[None, ...])

    return impl


# ---------------------------------- 1 int, 1 double --------------------------------- #


def _ComptonProfile(Z, p):
    _check_types((Z, p), 1, 1)
    cp = _nint_ndouble("ComptonProfile", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_PZ, SPLINE_EXTRAPOLATION))

    def impl(Z, p):
        error = np.array([0, 0], dtype=np.int32)
        result = cp(Z, p, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ComptonProfile)(_ComptonProfile)
overload(_xraylib.ComptonProfile)(_ComptonProfile)


@overload(xraylib_np.ComptonProfile)
def _ComptonProfile_np(Z, p):
    _check_types((Z, p), 1, 1, _np=True)
    cp = _nint_ndouble("ComptonProfile", 1, 1)

    @vectorize
    def _impl(Z, p):
        return cp(Z, p, 0)

    def impl(Z, p):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert p.ndim == 1
        return _impl(Z[..., None], p[None, ...])

    return impl


def _CS_Compt(Z, E):
    _check_types((Z, E), 1, 1)
    cs_compt = _nint_ndouble("CS_Compt", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_compt(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Compt)(_CS_Compt)
overload(_xraylib.CS_Compt)(_CS_Compt)


@overload(xraylib_np.CS_Compt)
def _CS_Compt_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_compt = _nint_ndouble("CS_Compt", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_compt(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Energy(Z, E):
    _check_types((Z, E), 1, 1)
    cs_energy = _nint_ndouble("CS_Energy", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_energy(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Energy)(_CS_Energy)
overload(_xraylib.CS_Energy)(_CS_Energy)


@overload(xraylib_np.CS_Energy)
def _CS_Energy(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_energy = _nint_ndouble("CS_Energy", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_energy(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Photo(Z, E):
    _check_types((Z, E), 1, 1)
    cs_photo = _nint_ndouble("CS_Photo", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_photo(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Photo)(_CS_Photo)
overload(_xraylib.CS_Photo)(_CS_Photo)


@overload(xraylib_np.CS_Photo)
def _CS_Photo_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_photo = _nint_ndouble("CS_Photo", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_photo(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1)
    cs_photo_total = _nint_ndouble("CS_Photo_Total", 1, 1)
    msg = " | ".join(
        (Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION, UNAVALIABLE_PHOTO_CS)
    )

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_photo_total(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Photo_Total)(_CS_Photo_Total)
overload(_xraylib.CS_Photo_Total)(_CS_Photo_Total)


@overload(xraylib_np.CS_Photo_Total)
def _CS_Photo_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_photo_total = _nint_ndouble("CS_Photo_Total", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_photo_total(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Rayl(Z, E):
    _check_types((Z, E), 1, 1)
    cs_rayl = _nint_ndouble("CS_Rayl", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_rayl(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Rayl)(_CS_Rayl)
overload(_xraylib.CS_Rayl)(_CS_Rayl)


@overload(xraylib_np.CS_Rayl)
def _CS_Rayl_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_rayl = _nint_ndouble("CS_Rayl", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_rayl(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Total(Z, E):
    _check_types((Z, E), 1, 1)
    cs_total = _nint_ndouble("CS_Total", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_total(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Total)(_CS_Total)
overload(_xraylib.CS_Total)(_CS_Total)


@overload(xraylib_np.CS_Total)
def _CS_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_total = _nint_ndouble("CS_Total", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_total(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CS_Total_Kissel(Z, E):
    _check_types((Z, E), 1, 1)
    cs_total_kissel = _nint_ndouble("CS_Total_Kissel", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_total_kissel(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Total_Kissel)(_CS_Total_Kissel)
overload(_xraylib.CS_Total_Kissel)(_CS_Total_Kissel)


@overload(xraylib_np.CS_Total_Kissel)
def _CS_Total_Kissel_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    cs_total_kissel = _nint_ndouble("CS_Total_Kissel", 1, 1)

    @vectorize
    def _impl(Z, E):
        return cs_total_kissel(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Compt(Z, E):
    _check_types((Z, E), 1, 1)
    csb_compt = _nint_ndouble("CSb_Compt", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_compt(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Compt)(_CSb_Compt)
overload(_xraylib.CSb_Compt)(_CSb_Compt)


@overload(xraylib_np.CSb_Compt)
def _CSb_Compt_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_compt = _nint_ndouble("CSb_Compt", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_compt(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Photo(Z, E):
    _check_types((Z, E), 1, 1)
    csb_photo = _nint_ndouble("CSb_Photo", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_photo(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Photo)(_CSb_Photo)
overload(_xraylib.CSb_Photo)(_CSb_Photo)


@overload(xraylib_np.CSb_Photo)
def _CSb_Photo_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_photo = _nint_ndouble("CSb_Photo", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_photo(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1)
    csb_photo_total = _nint_ndouble("CSb_Photo_Total", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_photo_total(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Photo_Total)(_CSb_Photo_Total)
overload(_xraylib.CSb_Photo_Total)(_CSb_Photo_Total)


@overload(xraylib_np.CSb_Photo_Total)
def _CSb_Photo_Total(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_photo_total = _nint_ndouble("CSb_Photo_Total", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_photo_total(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Rayl(Z, E):
    _check_types((Z, E), 1, 1)
    csb_rayl = _nint_ndouble("CSb_Rayl", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_rayl(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Rayl)(_CSb_Rayl)
overload(_xraylib.CSb_Rayl)(_CSb_Rayl)


@overload(xraylib_np.CSb_Rayl)
def _CSb_Rayl_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_rayl = _nint_ndouble("CSb_Rayl", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_rayl(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Total(Z, E):
    _check_types((Z, E), 1, 1)
    csb_total = _nint_ndouble("CSb_Total", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_total(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Total)(_CSb_Total)
overload(_xraylib.CSb_Total)(_CSb_Total)


@overload(xraylib_np.CSb_Total)
def _CSb_Total_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_total = _nint_ndouble("CSb_Total", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_total(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _CSb_Total_Kissel(Z, E):
    _check_types((Z, E), 1, 1)
    csb_total_kissel = _nint_ndouble("CSb_Total_Kissel", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_total_kissel(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Total_Kissel)(_CSb_Total_Kissel)
overload(_xraylib.CSb_Total_Kissel)(_CSb_Total_Kissel)


@overload(xraylib_np.CSb_Total_Kissel)
def _CSb_Total_Kissel_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    csb_total_kissel = _nint_ndouble("CSb_Total_Kissel", 1, 1)

    @vectorize
    def _impl(Z, E):
        return csb_total_kissel(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _FF_Rayl(Z, q):
    _check_types((Z, q), 1, 1)
    ff_rayl = _nint_ndouble("FF_Rayl", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_Q, SPLINE_EXTRAPOLATION))

    def impl(Z, q):
        error = np.array([0, 0], dtype=np.int32)
        result = ff_rayl(Z, q, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.FF_Rayl)(_FF_Rayl)
overload(_xraylib.FF_Rayl)(_FF_Rayl)


@overload(xraylib_np.FF_Rayl)
def _FF_Rayl_np(Z, q):
    _check_types((Z, q), 1, 1, _np=True)
    ff_rayl = _nint_ndouble("FF_Rayl", 1, 1)

    @vectorize
    def _impl(Z, q):
        return ff_rayl(Z, q, 0)

    def impl(Z, q):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert q.ndim == 1
        return _impl(Z[..., None], q[None, ...])

    return impl


def _SF_Compt(Z, q):
    _check_types((Z, q), 1, 1)
    sf_compt = _nint_ndouble("SF_Compt", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_Q, SPLINE_EXTRAPOLATION))

    def impl(Z, q):
        error = np.array([0, 0], dtype=np.int32)
        result = sf_compt(Z, q, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.SF_Compt)(_SF_Compt)
overload(_xraylib.SF_Compt)(_SF_Compt)


@overload(xraylib_np.SF_Compt)
def _SF_Compt_np(Z, q):
    _check_types((Z, q), 1, 1, _np=True)
    sf_compt = _nint_ndouble("SF_Compt", 1, 1)

    @vectorize
    def _impl(Z, q):
        return sf_compt(Z, q, 0)

    def impl(Z, q):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert q.ndim == 1
        return _impl(Z[..., None], q[None, ...])

    return impl


def _Fi(Z, E):
    _check_types((Z, E), 1, 1)
    fi = _nint_ndouble("Fi", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = fi(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.Fi)(_Fi)
overload(_xraylib.Fi)(_Fi)


@overload(xraylib_np.Fi)
def _Fi_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    fi = _nint_ndouble("Fi", 1, 1)

    @vectorize
    def _impl(Z, E):
        return fi(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


def _Fii(Z, E):
    _check_types((Z, E), 1, 1)
    fii = _nint_ndouble("Fii", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, E):
        error = np.array([0, 0], dtype=np.int32)
        result = fii(Z, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.Fii)(_Fii)
overload(_xraylib.Fii)(_Fii)


@overload(xraylib_np.Fii)
def _Fii_np(Z, E):
    _check_types((Z, E), 1, 1, _np=True)
    fii = _nint_ndouble("Fii", 1, 1)

    @vectorize
    def _impl(Z, E):
        return fii(Z, E, 0)

    def impl(Z, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None], E[None, ...])

    return impl


# !!! Not implemented in xraylib_np
def _PL1_pure_kissel(Z, energy):
    _check_types((Z, energy), 1, 1)
    pl1_pure_kissel = _nint_ndouble("PL1_pure_kissel", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, energy):
        error = np.array([0, 0], dtype=np.int32)
        result = pl1_pure_kissel(Z, energy, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL1_pure_kissel)(_PL1_pure_kissel)
overload(_xraylib.PL1_pure_kissel)(_PL1_pure_kissel)


# !!! Not implemented in xraylib_np
def _PM1_pure_kissel(Z, energy):
    _check_types((Z, energy), 1, 1)
    pm1_pure_kissel = _nint_ndouble("PM1_pure_kissel", 1, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, SPLINE_EXTRAPOLATION))

    def impl(Z, energy):
        error = np.array([0, 0], dtype=np.int32)
        result = pm1_pure_kissel(Z, energy, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM1_pure_kissel)(_PM1_pure_kissel)
overload(_xraylib.PM1_pure_kissel)(_PM1_pure_kissel)

# ---------------------------------- 2 int, 1 double --------------------------------- #


def _ComptonProfile_Partial(Z, shell, pz):
    _check_types((Z, shell, pz), 2, 1)
    cpp = _nint_ndouble("ComptonProfile_Partial", 2, 1)
    msg = " | ".join(
        (
            Z_OUT_OF_RANGE,
            NEGATIVE_PZ,
            SPLINE_EXTRAPOLATION,
            UNKNOWN_SHELL,
            INVALID_SHELL,
        )
    )

    def impl(Z, shell, pz):
        error = np.array([0, 0], dtype=np.int32)
        result = cpp(Z, shell, pz, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.ComptonProfile_Partial)(_ComptonProfile_Partial)
overload(_xraylib.ComptonProfile_Partial)(_ComptonProfile_Partial)


@overload(xraylib_np.ComptonProfile_Partial)
def _ComptonProfile_Partial_np(Z, shell, pz):
    _check_types((Z, shell, pz), 2, 1, _np=True)
    cpp = _nint_ndouble("ComptonProfile_Partial", 2, 1)

    @vectorize
    def _impl(Z, shell, pz):
        return cpp(Z, shell, pz, 0)

    def impl(Z, shell, pz):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert pz.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], pz[None, None, ...])

    return impl


def _CS_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line_kissel = _nint_ndouble("CS_FluorLine_Kissel", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line_kissel(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine_Kissel)(_CS_FluorLine_Kissel)
overload(_xraylib.CS_FluorLine_Kissel)(_CS_FluorLine_Kissel)


@overload(xraylib_np.CS_FluorLine_Kissel)
def _CS_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line_kissel = _nint_ndouble("CS_FluorLine_Kissel", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line_kissel(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine_Kissel(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line_kissel = _nint_ndouble("CSb_FluorLine_Kissel", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line_kissel(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine_Kissel)(_CSb_FluorLine_Kissel)
overload(_xraylib.CSb_FluorLine_Kissel)(_CSb_FluorLine_Kissel)


@overload(xraylib_np.CSb_FluorLine_Kissel)
def _CSb_FluorLine_Kissel_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line_kissel = _nint_ndouble("CSb_FluorLine_Kissel", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line_kissel(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line_kissel_cascade = _nint_ndouble("CS_FluorLine_Kissel_Cascade", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line_kissel_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine_Kissel_Cascade)(_CS_FluorLine_Kissel_Cascade)
overload(_xraylib.CS_FluorLine_Kissel_Cascade)(_CS_FluorLine_Kissel_Cascade)


@overload(xraylib_np.CS_FluorLine_Kissel_Cascade)
def _CS_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line_kissel_cascade = _nint_ndouble("CS_FluorLine_Kissel_Cascade", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line_kissel_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine_Kissel_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line_kissel_cascade = _nint_ndouble("CSb_FluorLine_Kissel_Cascade", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line_kissel_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine_Kissel_Cascade)(_CSb_FluorLine_Kissel_Cascade)
overload(_xraylib.CSb_FluorLine_Kissel_Cascade)(_CSb_FluorLine_Kissel_Cascade)


@overload(xraylib_np.CSb_FluorLine_Kissel_Cascade)
def _CSb_FluorLine_Kissel_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line_kissel_cascade = _nint_ndouble("CSb_FluorLine_Kissel_Cascade", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line_kissel_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line_kissel_no_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_no_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line_kissel_no_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine_Kissel_no_Cascade)(_CS_FluorLine_Kissel_no_Cascade)
overload(_xraylib.CS_FluorLine_Kissel_no_Cascade)(_CS_FluorLine_Kissel_no_Cascade)


@overload(xraylib_np.CS_FluorLine_Kissel_no_Cascade)
def _CS_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line_kissel_no_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_no_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line_kissel_no_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine_Kissel_no_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line_kissel_no_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_no_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line_kissel_no_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine_Kissel_no_Cascade)(_CSb_FluorLine_Kissel_no_Cascade)
overload(_xraylib.CSb_FluorLine_Kissel_no_Cascade)(_CSb_FluorLine_Kissel_no_Cascade)


@overload(xraylib_np.CSb_FluorLine_Kissel_no_Cascade)
def _CSb_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line_kissel_no_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_no_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line_kissel_no_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line_kissel_nonradiative_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_Nonradiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line_kissel_nonradiative_cascade(
            Z, line, E, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade)(
    _CS_FluorLine_Kissel_Nonradiative_Cascade
)
overload(_xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade)(
    _CS_FluorLine_Kissel_Nonradiative_Cascade
)


@overload(xraylib_np.CS_FluorLine_Kissel_Nonradiative_Cascade)
def _CS_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line_kissel_nonradiative_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_Nonradiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line_kissel_nonradiative_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line_kissel_nonradiative_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_Nonradiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line_kissel_nonradiative_cascade(
            Z, line, E, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine_Kissel_Nonradiative_Cascade)(
    _CSb_FluorLine_Kissel_Nonradiative_Cascade
)
overload(_xraylib.CSb_FluorLine_Kissel_Nonradiative_Cascade)(
    _CSb_FluorLine_Kissel_Nonradiative_Cascade
)


@overload(xraylib_np.CSb_FluorLine_Kissel_Nonradiative_Cascade)
def _CSb_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line_kissel_nonradiative_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_Nonradiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line_kissel_nonradiative_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line_kissel_radiative_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_Radiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line_kissel_radiative_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine_Kissel_Radiative_Cascade)(
    _CS_FluorLine_Kissel_Radiative_Cascade
)
overload(_xraylib.CS_FluorLine_Kissel_Radiative_Cascade)(
    _CS_FluorLine_Kissel_Radiative_Cascade
)


@overload(xraylib_np.CS_FluorLine_Kissel_Radiative_Cascade)
def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line_kissel_radiative_cascade = _nint_ndouble(
        "CS_FluorLine_Kissel_Radiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line_kissel_radiative_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line_kissel_radiative_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_Radiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line_kissel_radiative_cascade(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine_Kissel_Radiative_Cascade)(
    _CSb_FluorLine_Kissel_Radiative_Cascade
)
overload(_xraylib.CSb_FluorLine_Kissel_Radiative_Cascade)(
    _CSb_FluorLine_Kissel_Radiative_Cascade
)


@overload(xraylib_np.CSb_FluorLine_Kissel_Radiative_Cascade)
def _CSb_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line_kissel_radiative_cascade = _nint_ndouble(
        "CSb_FluorLine_Kissel_Radiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line_kissel_radiative_cascade(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell_kissel = _nint_ndouble("CS_FluorShell_Kissel", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell_kissel(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell_Kissel)(_CS_FluorShell_Kissel)
overload(_xraylib.CS_FluorShell_Kissel)(_CS_FluorShell_Kissel)


@overload(xraylib_np.CS_FluorShell_Kissel)
def _CS_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell_kissel = _nint_ndouble("CS_FluorShell_Kissel", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell_kissel(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell_Kissel(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell_kissel = _nint_ndouble("CSb_FluorShell_Kissel", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell_kissel(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell_Kissel)(_CSb_FluorShell_Kissel)
overload(_xraylib.CSb_FluorShell_Kissel)(_CSb_FluorShell_Kissel)


@overload(xraylib_np.CSb_FluorShell_Kissel)
def _CSb_FluorShell_Kissel_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell_kissel = _nint_ndouble("CSb_FluorShell_Kissel", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell_kissel(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell_kissel_cascade = _nint_ndouble("CS_FluorShell_Kissel_Cascade", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell_kissel_cascade(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell_Kissel_Cascade)(_CS_FluorShell_Kissel_Cascade)
overload(_xraylib.CS_FluorShell_Kissel_Cascade)(_CS_FluorShell_Kissel_Cascade)


@overload(xraylib_np.CS_FluorShell_Kissel_Cascade)
def _CS_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell_kissel_cascade = _nint_ndouble("CS_FluorShell_Kissel_Cascade", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell_kissel_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell_Kissel_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell_kissel_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell_kissel_cascade(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell_Kissel_Cascade)(_CSb_FluorShell_Kissel_Cascade)
overload(_xraylib.CSb_FluorShell_Kissel_Cascade)(_CSb_FluorShell_Kissel_Cascade)


@overload(xraylib_np.CSb_FluorShell_Kissel_Cascade)
def _CSb_FluorShell_Kissel_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell_kissel_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell_kissel_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell_kissel_no_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_no_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell_kissel_no_cascade(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell_Kissel_no_Cascade)(_CS_FluorShell_Kissel_no_Cascade)
overload(_xraylib.CS_FluorShell_Kissel_no_Cascade)(_CS_FluorShell_Kissel_no_Cascade)


@overload(xraylib_np.CS_FluorShell_Kissel_no_Cascade)
def _CS_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell_kissel_no_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_no_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell_kissel_no_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell_Kissel_no_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell_kissel_no_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_no_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell_kissel_no_cascade(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell_Kissel_no_Cascade)(_CSb_FluorShell_Kissel_no_Cascade)
overload(_xraylib.CSb_FluorShell_Kissel_no_Cascade)(_CSb_FluorShell_Kissel_no_Cascade)


@overload(xraylib_np.CSb_FluorShell_Kissel_no_Cascade)
def _CSb_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell_kissel_no_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_no_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell_kissel_no_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell_kissel_nonradiative_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_Nonradiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell_kissel_nonradiative_cascade(
            Z, shell, E, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell_Kissel_Nonradiative_Cascade)(
    _CS_FluorShell_Kissel_Nonradiative_Cascade
)
overload(_xraylib.CS_FluorShell_Kissel_Nonradiative_Cascade)(
    _CS_FluorShell_Kissel_Nonradiative_Cascade
)


@overload(xraylib_np.CS_FluorShell_Kissel_Nonradiative_Cascade)
def _CS_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell_kissel_nonradiative_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_Nonradiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell_kissel_nonradiative_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell_kissel_nonradiative_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Nonradiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell_kissel_nonradiative_cascade(
            Z, shell, E, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell_Kissel_Nonradiative_Cascade)(
    _CSb_FluorShell_Kissel_Nonradiative_Cascade
)
overload(_xraylib.CSb_FluorShell_Kissel_Nonradiative_Cascade)(
    _CSb_FluorShell_Kissel_Nonradiative_Cascade
)


@overload(xraylib_np.CSb_FluorShell_Kissel_Nonradiative_Cascade)
def _CSb_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell_kissel_nonradiative_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Nonradiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell_kissel_nonradiative_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell_kissel_radiative_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_Radiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell_kissel_radiative_cascade(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell_Kissel_Radiative_Cascade)(
    _CS_FluorShell_Kissel_Radiative_Cascade
)
overload(_xraylib.CS_FluorShell_Kissel_Radiative_Cascade)(
    _CS_FluorShell_Kissel_Radiative_Cascade
)


@overload(xraylib_np.CS_FluorShell_Kissel_Radiative_Cascade)
def _CS_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell_kissel_radiative_cascade = _nint_ndouble(
        "CS_FluorShell_Kissel_Radiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell_kissel_radiative_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell_kissel_radiative_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Radiative_Cascade", 2, 1
    )
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell_kissel_radiative_cascade(
            Z, shell, E, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell_Kissel_Radiative_Cascade)(
    _CSb_FluorShell_Kissel_Radiative_Cascade
)
overload(_xraylib.CSb_FluorShell_Kissel_Radiative_Cascade)(
    _CSb_FluorShell_Kissel_Radiative_Cascade
)


@overload(xraylib_np.CSb_FluorShell_Kissel_Radiative_Cascade)
def _CSb_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell_kissel_radiative_cascade = _nint_ndouble(
        "CSb_FluorShell_Kissel_Radiative_Cascade", 2, 1
    )

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell_kissel_radiative_cascade(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    cs_fluor_line = _nint_ndouble("CS_FluorLine", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_line(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorLine)(_CS_FluorLine)
overload(_xraylib.CS_FluorLine)(_CS_FluorLine)


@overload(xraylib_np.CS_FluorLine)
def _CS_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    cs_fluor_line = _nint_ndouble("CS_FluorLine", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return cs_fluor_line(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorLine(Z, line, E):
    _check_types((Z, line, E), 2, 1)
    csb_fluor_line = _nint_ndouble("CSb_FluorLine", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_LINE, INVALID_LINE))

    def impl(Z, line, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_line(Z, line, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorLine)(_CSb_FluorLine)
overload(_xraylib.CSb_FluorLine)(_CSb_FluorLine)


@overload(xraylib_np.CSb_FluorLine)
def _CSb_FluorLine_np(Z, line, E):
    _check_types((Z, line, E), 2, 1, _np=True)
    csb_fluor_line = _nint_ndouble("CSb_FluorLine", 2, 1)

    @vectorize
    def _impl(Z, line, E):
        return csb_fluor_line(Z, line, E, 0)

    def impl(Z, line, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert line.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], line[None, ..., None], E[None, None, ...])

    return impl


def _CS_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_fluor_shell = _nint_ndouble("CS_FluorShell", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_fluor_shell(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_FluorShell)(_CS_FluorShell)
overload(_xraylib.CS_FluorShell)(_CS_FluorShell)


@overload(xraylib_np.CS_FluorShell)
def _CS_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_fluor_shell = _nint_ndouble("CS_FluorShell", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return cs_fluor_shell(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_FluorShell(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_fluor_shell = _nint_ndouble("CSb_FluorShell", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_fluor_shell(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_FluorShell)(_CSb_FluorShell)
overload(_xraylib.CSb_FluorShell)(_CSb_FluorShell)


@overload(xraylib_np.CSb_FluorShell)
def _CSb_FluorShell_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_fluor_shell = _nint_ndouble("CSb_FluorShell", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return csb_fluor_shell(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CS_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    cs_photo_partial = _nint_ndouble("CS_Photo_Partial", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = cs_photo_partial(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CS_Photo_Partial)(_CS_Photo_Partial)
overload(_xraylib.CS_Photo_Partial)(_CS_Photo_Partial)


@overload(xraylib_np.CS_Photo_Partial)
def _CS_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    cs_photo_partial = _nint_ndouble("CS_Photo_Partial", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return cs_photo_partial(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


def _CSb_Photo_Partial(Z, shell, E):
    _check_types((Z, shell, E), 2, 1)
    csb_photo_partial = _nint_ndouble("CSb_Photo_Partial", 2, 1)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY, UNKNOWN_SHELL, INVALID_SHELL))

    def impl(Z, shell, E):
        error = np.array([0, 0], dtype=np.int32)
        result = csb_photo_partial(Z, shell, E, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.CSb_Photo_Partial)(_CSb_Photo_Partial)
overload(_xraylib.CSb_Photo_Partial)(_CSb_Photo_Partial)


@overload(xraylib_np.CSb_Photo_Partial)
def _CSb_Photo_Partial_np(Z, shell, E):
    _check_types((Z, shell, E), 2, 1, _np=True)
    csb_photo_partial = _nint_ndouble("CSb_Photo_Partial", 2, 1)

    @vectorize
    def _impl(Z, shell, E):
        return csb_photo_partial(Z, shell, E, 0)

    def impl(Z, shell, E):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert shell.ndim == 1
            assert E.ndim == 1
        return _impl(Z[..., None, None], shell[None, ..., None], E[None, None, ...])

    return impl


# ---------------------------------- 1 int, 2 double --------------------------------- #


def _DCS_Compt(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    dcs_compt = _nint_ndouble("DCS_Compt", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = dcs_compt(Z, E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCS_Compt)(_DCS_Compt)
overload(_xraylib.DCS_Compt)(_DCS_Compt)


@overload(xraylib_np.DCS_Compt)
def _DCS_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    dcs_compt = _nint_ndouble("DCS_Compt", 1, 2)

    @vectorize
    def _impl(Z, E, theta):
        return dcs_compt(Z, E, theta, 0)

    def impl(Z, E, theta):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(Z[..., None, None], E[None, ..., None], theta[None, None, ...])

    return impl


def _DCS_Rayl(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    dcs_rayl = _nint_ndouble("DCS_Rayl", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = dcs_rayl(Z, E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCS_Rayl)(_DCS_Rayl)
overload(_xraylib.DCS_Rayl)(_DCS_Rayl)


@overload(xraylib_np.DCS_Rayl)
def _DCS_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    dcs_rayl = _nint_ndouble("DCS_Rayl", 1, 2)

    @vectorize
    def _impl(Z, E, theta):
        return dcs_rayl(Z, E, theta, 0)

    def impl(Z, E, theta):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(Z[..., None, None], E[None, ..., None], theta[None, None, ...])

    return impl


def _DCSb_Compt(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    dcsb_compt = _nint_ndouble("DCSb_Compt", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = dcsb_compt(Z, E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSb_Compt)(_DCSb_Compt)
overload(_xraylib.DCSb_Compt)(_DCSb_Compt)


@overload(xraylib_np.DCSb_Compt)
def _DCSb_Compt_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    dcsb_compt = _nint_ndouble("DCSb_Compt", 1, 2)

    @vectorize
    def _impl(Z, E, theta):
        return dcsb_compt(Z, E, theta, 0)

    def impl(Z, E, theta):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(Z[..., None, None], E[None, ..., None], theta[None, None, ...])

    return impl


def _DCSb_Rayl(Z, E, theta):
    _check_types((Z, E, theta), 1, 2)
    dcsb_rayl = _nint_ndouble("DCSb_Rayl", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta):
        error = np.array([0, 0], dtype=np.int32)
        result = dcsb_rayl(Z, E, theta, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSb_Rayl)(_DCSb_Rayl)
overload(_xraylib.DCSb_Rayl)(_DCSb_Rayl)


@overload(xraylib_np.DCSb_Rayl)
def _DCSb_Rayl_np(Z, E, theta):
    _check_types((Z, E, theta), 1, 2, _np=True)
    dcsb_rayl = _nint_ndouble("DCSb_Rayl", 1, 2)

    @vectorize
    def _impl(Z, E, theta):
        return dcsb_rayl(Z, E, theta, 0)

    def impl(Z, E, theta):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
        return _impl(Z[..., None, None], E[None, ..., None], theta[None, None, ...])

    return impl


# !!! Not implemented in xraylib_np
def _PL1_auger_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    pl1_auger_cascade_kissel = _nint_ndouble("PL1_auger_cascade_kissel", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK):
        error = np.array([0, 0], dtype=np.int32)
        result = pl1_auger_cascade_kissel(Z, E, PK, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL1_auger_cascade_kissel)(_PL1_auger_cascade_kissel)
overload(_xraylib.PL1_auger_cascade_kissel)(_PL1_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL1_full_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    pl1_full_cascade_kissel = _nint_ndouble("PL1_full_cascade_kissel", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK):
        error = np.array([0, 0], dtype=np.int32)
        result = pl1_full_cascade_kissel(Z, E, PK, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL1_full_cascade_kissel)(_PL1_full_cascade_kissel)
overload(_xraylib.PL1_full_cascade_kissel)(_PL1_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL1_rad_cascade_kissel(Z, E, PK):
    _check_types((Z, E, PK), 1, 2)
    pl1_rad_cascade_kissel = _nint_ndouble("PL1_rad_cascade_kissel", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK):
        error = np.array([0, 0], dtype=np.int32)
        result = pl1_rad_cascade_kissel(Z, E, PK, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL1_rad_cascade_kissel)(_PL1_rad_cascade_kissel)
overload(_xraylib.PL1_rad_cascade_kissel)(_PL1_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_pure_kissel(Z, E, PL1):
    _check_types((Z, E, PL1), 1, 2)
    pl2_pure_kissel = _nint_ndouble("PL2_pure_kissel", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PL1):
        error = np.array([0, 0], dtype=np.int32)
        result = pl2_pure_kissel(Z, E, PL1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL2_pure_kissel)(_PL2_pure_kissel)
overload(_xraylib.PL2_pure_kissel)(_PL2_pure_kissel)


# !!! Not implemented in xraylib_np
def _PM2_pure_kissel(Z, E, PM1):
    _check_types((Z, E, PM1), 1, 2)
    pm2_pure_kissel = _nint_ndouble("PM2_pure_kissel", 1, 2)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PM1):
        error = np.array([0, 0], dtype=np.int32)
        result = pm2_pure_kissel(Z, E, PM1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM2_pure_kissel)(_PM2_pure_kissel)
overload(_xraylib.PM2_pure_kissel)(_PM2_pure_kissel)


# ---------------------------------- 1 int, 3 double --------------------------------- #


def _DCSP_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    dcsp_rayl = _nint_ndouble("DCSP_Rayl", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = dcsp_rayl(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_Rayl)(_DCSP_Rayl)
overload(_xraylib.DCSP_Rayl)(_DCSP_Rayl)


@overload(xraylib_np.DCSP_Rayl)
def _DCSP_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    dcsp_rayl = _nint_ndouble("DCSP_Rayl", 1, 3)

    @vectorize
    def _impl(Z, E, theta, phi):
        return dcsp_rayl(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(
            Z[..., None, None, None],
            E[None, ..., None, None],
            theta[None, None, ..., None],
            phi[None, None, None, ...],
        )

    return impl


def _DCSP_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    dcsp_compt = _nint_ndouble("DCSP_Compt", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = dcsp_compt(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_Compt)(_DCSP_Compt)
overload(_xraylib.DCSP_Compt)(_DCSP_Compt)


@overload(xraylib_np.DCSP_Compt)
def _DCSP_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    dcsp_compt = _nint_ndouble("DCSP_Compt", 1, 3)

    @vectorize
    def _impl(Z, E, theta, phi):
        return dcsp_compt(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(
            Z[..., None, None, None],
            E[None, ..., None, None],
            theta[None, None, ..., None],
            phi[None, None, None, ...],
        )

    return impl


def _DCSPb_Rayl(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    dcspb_rayl = _nint_ndouble("DCSPb_Rayl", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = dcspb_rayl(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSPb_Rayl)(_DCSPb_Rayl)
overload(_xraylib.DCSPb_Rayl)(_DCSPb_Rayl)


@overload(xraylib_np.DCSPb_Rayl)
def _DCSPb_Rayl_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    dcspb_rayl = _nint_ndouble("DCSPb_Rayl", 1, 3)

    @vectorize
    def _impl(Z, E, theta, phi):
        return dcspb_rayl(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(
            Z[..., None, None, None],
            E[None, ..., None, None],
            theta[None, None, ..., None],
            phi[None, None, None, ...],
        )

    return impl


def _DCSPb_Compt(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    dcspb_compt = _nint_ndouble("DCSPb_Compt", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = dcspb_compt(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSPb_Compt)(_DCSPb_Compt)
overload(_xraylib.DCSPb_Compt)(_DCSPb_Compt)


@overload(xraylib_np.DCSPb_Compt)
def _DCSPb_Compt_np(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3, _np=True)
    dcspb_compt = _nint_ndouble("DCSPb_Compt", 1, 3)

    @vectorize
    def _impl(Z, E, theta, phi):
        return dcspb_compt(Z, E, theta, phi, 0)

    def impl(Z, E, theta, phi):
        if not config.allow_Nd:
            assert Z.ndim == 1
            assert E.ndim == 1
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(
            Z[..., None, None, None],
            E[None, ..., None, None],
            theta[None, None, ..., None],
            phi[None, None, None, ...],
        )

    return impl


# !!! Not implemented in xraylib_np
def _PL2_auger_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    pl2_auger_cascade_kissel = _nint_ndouble("PL2_auger_cascade_kissel", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = pl2_auger_cascade_kissel(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL2_auger_cascade_kissel)(_PL2_auger_cascade_kissel)
overload(_xraylib.PL2_auger_cascade_kissel)(_PL2_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_full_cascade_kissel(Z, E, theta, phi):
    _check_types((Z, E, theta, phi), 1, 3)
    pl2_full_cascade_kissel = _nint_ndouble("PL2_full_cascade_kissel", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = pl2_full_cascade_kissel(Z, E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL2_full_cascade_kissel)(_PL2_full_cascade_kissel)
overload(_xraylib.PL2_full_cascade_kissel)(_PL2_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL2_rad_cascade_kissel(Z, E, PK, PL1):
    _check_types((Z, E, PK, PL1), 1, 3)
    pl2_rad_cascade_kissel = _nint_ndouble("PL2_rad_cascade_kissel", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1):
        error = np.array([0, 0], dtype=np.int32)
        result = pl2_rad_cascade_kissel(Z, E, PK, PL1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL2_rad_cascade_kissel)(_PL2_rad_cascade_kissel)
overload(_xraylib.PL2_rad_cascade_kissel)(_PL2_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_pure_kissel(Z, E, PL1, PL2):
    _check_types((Z, E, PL1, PL2), 1, 3)
    pl3_pure_kissel = _nint_ndouble("PL3_pure_kissel", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PL1, PL2):
        error = np.array([0, 0], dtype=np.int32)
        result = pl3_pure_kissel(Z, E, PL1, PL2, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL3_pure_kissel)(_PL3_pure_kissel)
overload(_xraylib.PL3_pure_kissel)(_PL3_pure_kissel)


# !!! Not implemented in xraylib_np
def _PM3_pure_kissel(Z, E, PM1, PM2):
    _check_types((Z, E, PM1, PM2), 1, 3)
    pm3_pure_kissel = _nint_ndouble("PM3_pure_kissel", 1, 3)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PM1, PM2):
        error = np.array([0, 0], dtype=np.int32)
        result = pm3_pure_kissel(Z, E, PM1, PM2, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM3_pure_kissel)(_PM3_pure_kissel)
overload(_xraylib.PM3_pure_kissel)(_PM3_pure_kissel)


# ---------------------------------- 1 int, 4 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PL3_auger_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), 1, 4)
    pl3_auger_cascade_kissel = _nint_ndouble("PL3_auger_cascade_kissel", 1, 4)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PL1):
        error = np.array([0, 0], dtype=np.int32)
        result = pl3_auger_cascade_kissel(Z, E, theta, phi, PL1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL3_auger_cascade_kissel)(_PL3_auger_cascade_kissel)
overload(_xraylib.PL3_auger_cascade_kissel)(_PL3_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_full_cascade_kissel(Z, E, theta, phi, PL1):
    _check_types((Z, E, theta, phi, PL1), 1, 4)
    pl3_full_cascade_kissel = _nint_ndouble("PL3_full_cascade_kissel", 1, 4)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PL1):
        error = np.array([0, 0], dtype=np.int32)
        result = pl3_full_cascade_kissel(Z, E, theta, phi, PL1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL3_full_cascade_kissel)(_PL3_full_cascade_kissel)
overload(_xraylib.PL3_full_cascade_kissel)(_PL3_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2):
    _check_types((Z, E, PK, PL1, PL2), 1, 4)
    pl3_rad_cascade_kissel = _nint_ndouble("PL3_rad_cascade_kissel", 1, 4)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2):
        error = np.array([0, 0], dtype=np.int32)
        result = pl3_rad_cascade_kissel(Z, E, PK, PL1, PL2, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PL3_rad_cascade_kissel)(_PL3_rad_cascade_kissel)
overload(_xraylib.PL3_rad_cascade_kissel)(_PL3_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_pure_kissel(Z, E, theta, phi, PM1):
    _check_types((Z, E, theta, phi, PM1), 1, 4)
    pm4_pure_kissel = _nint_ndouble("PM4_pure_kissel", 1, 4)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM1):
        error = np.array([0, 0], dtype=np.int32)
        result = pm4_pure_kissel(Z, E, theta, phi, PM1, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM4_pure_kissel)(_PM4_pure_kissel)
overload(_xraylib.PM4_pure_kissel)(_PM4_pure_kissel)


# ---------------------------------- 1 int, 5 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM1_auger_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), 1, 5)
    pm1_auger_cascade_kissel = _nint_ndouble("PM1_auger_cascade_kissel", 1, 5)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM2, PM3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm1_auger_cascade_kissel(Z, E, theta, phi, PM2, PM3, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM1_auger_cascade_kissel)(_PM1_auger_cascade_kissel)
overload(_xraylib.PM1_auger_cascade_kissel)(_PM1_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM1_full_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    _check_types((Z, E, theta, phi, PM2, PM3), 1, 5)
    pm1_full_cascade_kissel = _nint_ndouble("PM1_full_cascade_kissel", 1, 5)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM2, PM3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm1_full_cascade_kissel(Z, E, theta, phi, PM2, PM3, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM1_full_cascade_kissel)(_PM1_full_cascade_kissel)
overload(_xraylib.PM1_full_cascade_kissel)(_PM1_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM1_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3):
    _check_types((Z, E, PK, PL, PL2, PL3), 1, 5)
    pm1_rad_cascade_kissel = _nint_ndouble("PM1_rad_cascade_kissel", 1, 5)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL, PL2, PL3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm1_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM1_rad_cascade_kissel)(_PM1_rad_cascade_kissel)
overload(_xraylib.PM1_rad_cascade_kissel)(_PM1_rad_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_pure_kissel(Z, E, theta, phi, PM1, PM2):
    _check_types((Z, E, theta, phi, PM1, PM2), 1, 5)
    pm5_pure_kissel = _nint_ndouble("PM5_pure_kissel", 1, 5)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM1, PM2):
        error = np.array([0, 0], dtype=np.int32)
        result = pm5_pure_kissel(Z, E, theta, phi, PM1, PM2, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM5_pure_kissel)(_PM5_pure_kissel)
overload(_xraylib.PM5_pure_kissel)(_PM5_pure_kissel)


# ---------------------------------- 1 int, 6 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM2_auger_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), 1, 6)
    pm2_auger_cascade_kissel = _nint_ndouble("PM2_auger_cascade_kissel", 1, 6)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = np.array([0, 0], dtype=np.int32)
        result = pm2_auger_cascade_kissel(
            Z, E, theta, phi, PM3, PM4, PM5, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM2_auger_cascade_kissel)(_PM2_auger_cascade_kissel)
overload(_xraylib.PM2_auger_cascade_kissel)(_PM2_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM2_full_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    _check_types((Z, E, theta, phi, PM3, PM4, PM5), 1, 6)
    pm2_full_cascade_kissel = _nint_ndouble("PM2_full_cascade_kissel", 1, 6)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        error = np.array([0, 0], dtype=np.int32)
        result = pm2_full_cascade_kissel(
            Z, E, theta, phi, PM3, PM4, PM5, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM2_full_cascade_kissel)(_PM2_full_cascade_kissel)
overload(_xraylib.PM2_full_cascade_kissel)(_PM2_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM2_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, PL4):
    _check_types((Z, E, PK, PL, PL2, PL3, PL4), 1, 6)
    pm2_rad_cascade_kissel = _nint_ndouble("PM2_rad_cascade_kissel", 1, 6)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL, PL2, PL3, PL4):
        error = np.array([0, 0], dtype=np.int32)
        result = pm2_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, PL4, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM2_rad_cascade_kissel)(_PM2_rad_cascade_kissel)
overload(_xraylib.PM2_rad_cascade_kissel)(_PM2_rad_cascade_kissel)


# ---------------------------------- 1 int, 7 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    pm3_auger_cascade_kissel = _nint_ndouble("PM3_auger_cascade_kissel", 1, 7)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = np.array([0, 0], dtype=np.int32)
        result = pm3_auger_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM3_auger_cascade_kissel)(_PM3_auger_cascade_kissel)
overload(_xraylib.PM3_auger_cascade_kissel)(_PM3_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    pm3_full_cascade_kissel = _nint_ndouble("PM3_full_cascade_kissel", 1, 7)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = np.array([0, 0], dtype=np.int32)
        result = pm3_full_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM3_full_cascade_kissel)(_PM3_full_cascade_kissel)
overload(_xraylib.PM3_full_cascade_kissel)(_PM3_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2), 1, 7)
    pm3_rad_cascade_kissel = _nint_ndouble("PM3_rad_cascade_kissel", 1, 7)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        error = np.array([0, 0], dtype=np.int32)
        result = pm3_rad_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM3_rad_cascade_kissel)(_PM3_rad_cascade_kissel)
overload(_xraylib.PM3_rad_cascade_kissel)(_PM3_rad_cascade_kissel)


# ---------------------------------- 1 int, 8 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    pm4_auger_cascade_kissel = _nint_ndouble("PM4_auger_cascade_kissel", 1, 8)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm4_auger_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM4_auger_cascade_kissel)(_PM4_auger_cascade_kissel)
overload(_xraylib.PM4_auger_cascade_kissel)(_PM4_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    pm4_full_cascade_kissel = _nint_ndouble("PM4_full_cascade_kissel", 1, 8)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm4_full_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM4_full_cascade_kissel)(_PM4_full_cascade_kissel)
overload(_xraylib.PM4_full_cascade_kissel)(_PM4_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), 1, 8)
    pm4_rad_cascade_kissel = _nint_ndouble("PM4_rad_cascade_kissel", 1, 8)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        error = np.array([0, 0], dtype=np.int32)
        result = pm4_rad_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM4_rad_cascade_kissel)(_PM4_rad_cascade_kissel)
overload(_xraylib.PM4_rad_cascade_kissel)(_PM4_rad_cascade_kissel)


# ---------------------------------- 1 int, 9 double --------------------------------- #


# !!! Not implemented in xraylib_np
def _PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    pm5_auger_cascade_kissel = _nint_ndouble("PM5_auger_cascade_kissel", 1, 9)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = np.array([0, 0], dtype=np.int32)
        result = pm5_auger_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM5_auger_cascade_kissel)(_PM5_auger_cascade_kissel)
overload(_xraylib.PM5_auger_cascade_kissel)(_PM5_auger_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    pm5_full_cascade_kissel = _nint_ndouble("PM5_full_cascade_kissel", 1, 9)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = np.array([0, 0], dtype=np.int32)
        result = pm5_full_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM5_full_cascade_kissel)(_PM5_full_cascade_kissel)
overload(_xraylib.PM5_full_cascade_kissel)(_PM5_full_cascade_kissel)


# !!! Not implemented in xraylib_np
def _PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    _check_types((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), 1, 9)
    pm5_rad_cascade_kissel = _nint_ndouble("PM5_rad_cascade_kissel", 1, 9)
    msg = " | ".join((Z_OUT_OF_RANGE, NEGATIVE_ENERGY))

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        error = np.array([0, 0], dtype=np.int32)
        result = pm5_rad_cascade_kissel(
            Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, error.ctypes.data
        )
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.PM5_rad_cascade_kissel)(_PM5_rad_cascade_kissel)
overload(_xraylib.PM5_rad_cascade_kissel)(_PM5_rad_cascade_kissel)


# ------------------------------------- 3 double ------------------------------------- #


def _DCSP_KN(E, theta, phi):
    _check_types((E, theta, phi), 0, 3)
    dcsp_kn = _nint_ndouble("DCSP_KN", 0, 3)
    msg = " | ".join((NEGATIVE_ENERGY,))

    def impl(E, theta, phi):
        error = np.array([0, 0], dtype=np.int32)
        result = dcsp_kn(E, theta, phi, error.ctypes.data)
        if error.any():
            raise ValueError(msg)
        return result

    return impl


overload(xraylib.DCSP_KN)(_DCSP_KN)
overload(_xraylib.DCSP_KN)(_DCSP_KN)


@overload(xraylib_np.DCSP_KN, {"parallel": False, "nogil": True})
def _DCSP_KN_np(E, theta, phi):
    _check_types((E, theta, phi), 0, 3, _np=True)
    dcsp_kn = _nint_ndouble("DCSP_KN", 0, 3)

    @vectorize
    def _impl(E, theta, phi):
        return dcsp_kn(E, theta, phi, 0)

    def impl(E, theta, phi):
        if not config.allow_Nd:
            assert E.ndim == 1
            assert theta.ndim == 1
            assert phi.ndim == 1
        return _impl(E[..., None, None], theta[None, ..., None], phi[None, None, ...])

    return impl


# -------------------------------- 1 string, 1 double -------------------------------- #

# ??? How to pass a python string to an external function

# TODO: CS_Total_CP
# TODO: CS_Photo_CP
# TODO: CS_Rayl_CP
# TODO: CS_Compt_CP
# TODO: CS_Energy_CP
# TODO: CS_Photo_Total_CP
# TODO: CS_Total_Kissel_CP
# TODO: CSb_Total_CP
# TODO: CSb_Photo_CP
# TODO: CSb_Rayl_CP
# TODO: CSb_Compt_CP
# TODO: CSb_Energy_CP
# TODO: CSb_Photo_Total_CP
# TODO: CSb_Total_Kissel_CP


# -------------------------------- 1 string, 2 double -------------------------------- #

# TODO: DCS_Rayl_CP
# TODO: DCS_Compt_CP
# TODO: DCSb_Rayl_CP
# TODO: DCSb_Compt_CP
# TODO: Refractive_Index_Re
# TODO: Refractive_Index_Im
# TODO: Refractive_Index

# -------------------------------- 1 string, 3 double -------------------------------- #

# TODO: DCSP_Rayl_CP
# TODO: DCSP_Compt_CP
# TODO: DCSPb_Rayl_CP
# TODO: DCSPb_Compt_CP

# TODO: Other functions with string returns etc...
