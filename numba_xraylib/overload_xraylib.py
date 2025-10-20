"""Overloads for functions in the xraylib namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending
from numpy import array, uint64

from .utils import check_types_xrl, convert_str, external_fcn, overload_xrl

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Error messages
ZOOR = "Z out of range"  # Z_OUT_OF_RANGE
NE = "Energy must be strictly positive"  # NEGATIVE_ENERGY
ND = "Density must be strictly positive"  # NEGATIVE_DENSITY
NQ = "q must be positive"  # NEGATIVE_Q
NPZ = "pz must be positive"  # NEGATIVE_PZ
IS = "Invalid shell for this atomic number"  # INVALID_SHELL
IL = "Invalid line for this atomic number"  # INVALID_LINE
ICK = "Invalid Coster-Kronig transition for this atomic number"  # INVALID_CK
IA = "Invalid Auger transition macro for this atomic number"  # INVALID_AUGER
US = "Unknown shell macro provided"  # UNKNOWN_SHELL
UL = "Unknown line macro provided"  # UNKNOWN_LINE
UCK = "Unknown Coster-Kronig transition macro provided"  # UNKNOWN_CK
UA = "Unknown Auger transition macro provided"  # UNKNOWN_AUGER
SE = "Spline extrapolation is not allowed"  # SPLINE_EXTRAPOLATION
UPCS = "Photoionization cross section unavailable for atomic number and energy"
# UNAVALIABLE_PHOTO_CS


@extending.register_jitable
def _error() -> NDArray[uint64]:
    return array([0], dtype=uint64)


@extending.register_jitable
def _check_error(e: NDArray[uint64]) -> uint64:
    return e.item()  # type: ignore  # noqa: PGH003


# --------------------------------------- 1 int -------------------------------------- #


@overload_xrl
def _AtomicWeight(Z):
    check_types_xrl((Z,), ni=1)
    xrl_fcn = external_fcn(ni=1)

    def impl(Z):
        e = _error()
        result = xrl_fcn(Z, e.ctypes)
        if _check_error(e):
            raise ValueError(ZOOR)
        return result

    return impl


@overload_xrl
def _ElementDensity(Z):
    check_types_xrl((Z,), ni=1)
    xrl_fcn = external_fcn(ni=1)

    def impl(Z):
        e = _error()
        result = xrl_fcn(Z, e.ctypes)
        if _check_error(e):
            raise ValueError(ZOOR)
        return result

    return impl


# ------------------------------------- 1 double ------------------------------------- #


@overload_xrl
def _CS_KN(E):
    check_types_xrl((E,), nf=1)
    xrl_fcn = external_fcn(nf=1)

    def impl(E):
        e = _error()
        result = xrl_fcn(E, e.ctypes)
        if _check_error(e):
            raise ValueError(NE)
        return result

    return impl


@overload_xrl
def _DCS_Thoms(theta):
    check_types_xrl((theta,), nf=1)
    xrl_fcn = external_fcn(nf=1)

    def impl(theta):
        return xrl_fcn(theta, 0)  # !!! no need for error arg

    return impl


# --------------------------------------- 2 int -------------------------------------- #


@overload_xrl
def _AtomicLevelWidth(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _AugerRate(Z, auger_trans):
    check_types_xrl((Z, auger_trans), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {UA} | {IA}"

    def impl(Z, auger_trans):
        e = _error()
        result = xrl_fcn(Z, auger_trans, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _AugerYield(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CosKronTransProb(Z, trans):
    check_types_xrl((Z, trans), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {UCK} | {ICK}"

    def impl(Z, trans):
        e = _error()
        result = xrl_fcn(Z, trans, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _EdgeEnergy(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _ElectronConfig(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _FluorYield(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _JumpFactor(Z, shell):
    check_types_xrl((Z, shell), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {US} | {IS}"

    def impl(Z, shell):
        e = _error()
        result = xrl_fcn(Z, shell, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _LineEnergy(Z, line):
    check_types_xrl((Z, line), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {UL} | {IL}"

    def impl(Z, line):
        e = _error()
        result = xrl_fcn(Z, line, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _RadRate(Z, line):
    check_types_xrl((Z, line), ni=2)
    xrl_fcn = external_fcn(ni=2)
    msg = f"{ZOOR} | {UL} | {IL}"

    def impl(Z, line):
        e = _error()
        result = xrl_fcn(Z, line, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ------------------------------------- 2 double ------------------------------------- #


@overload_xrl
def _ComptonEnergy(E0, theta):
    check_types_xrl((E0, theta), nf=2)
    xrl_fcn = external_fcn(nf=2)
    msg = NE

    def impl(E0, theta):
        e = _error()
        result = xrl_fcn(E0, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCS_KN(E, theta):
    check_types_xrl((E, theta), nf=2)
    xrl_fcn = external_fcn(nf=2)

    def impl(E, theta):
        e = _error()
        result = xrl_fcn(E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(NE)
        return result

    return impl


@overload_xrl
def _DCSP_Thoms(theta, phi):
    check_types_xrl((theta, phi), nf=2)
    xrl_fcn = external_fcn(nf=2)

    def impl(theta, phi):
        return xrl_fcn(theta, phi, 0)  # !!! no need for error arg

    return impl


@overload_xrl
def _MomentTransf(E, theta):
    check_types_xrl((E, theta), nf=2)
    xrl_fcn = external_fcn(nf=2)

    def impl(E, theta):
        e = _error()
        result = xrl_fcn(E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(NE)
        return result

    return impl


# ---------------------------------- 1 int, 1 double --------------------------------- #


@overload_xrl
def _ComptonProfile(Z, p):
    check_types_xrl((Z, p), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NPZ} | {SE}"

    def impl(Z, p):
        e = _error()
        result = xrl_fcn(Z, p, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Compt(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Energy(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Photo(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Photo_Total(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE} | {UPCS}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Rayl(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Total(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Total_Kissel(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Compt(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Photo(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Photo_Total(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Rayl(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Total(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Total_Kissel(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _FF_Rayl(Z, q):
    check_types_xrl((Z, q), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NQ} | {SE}"

    def impl(Z, q):
        e = _error()
        result = xrl_fcn(Z, q, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _SF_Compt(Z, q):
    check_types_xrl((Z, q), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NQ} | {SE}"

    def impl(Z, q):
        e = _error()
        result = xrl_fcn(Z, q, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _Fi(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _Fii(Z, E):
    check_types_xrl((Z, E), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, E):
        e = _error()
        result = xrl_fcn(Z, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL1_pure_kissel(Z, energy):
    check_types_xrl((Z, energy), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, energy):
        e = _error()
        result = xrl_fcn(Z, energy, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM1_pure_kissel(Z, energy):
    check_types_xrl((Z, energy), ni=1, nf=1)
    xrl_fcn = external_fcn(ni=1, nf=1)
    msg = f"{ZOOR} | {NE} | {SE}"

    def impl(Z, energy):
        e = _error()
        result = xrl_fcn(Z, energy, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 2 int, 1 double --------------------------------- #


@overload_xrl
def _ComptonProfile_Partial(Z, shell, pz):
    check_types_xrl((Z, shell, pz), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NPZ} | {SE} | {US} | {IS}"

    def impl(Z, shell, pz):
        e = _error()
        result = xrl_fcn(Z, shell, pz, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine_Kissel(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine_Kissel(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine_Kissel_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine_Kissel_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine_Kissel_no_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine_Kissel_no_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell_Kissel(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell_Kissel(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell_Kissel_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell_Kissel_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell_Kissel_no_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell_Kissel_no_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorLine(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorLine(Z, line, E):
    check_types_xrl((Z, line, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {UL} | {IL}"

    def impl(Z, line, E):
        e = _error()
        result = xrl_fcn(Z, line, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_FluorShell(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_FluorShell(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Photo_Partial(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Photo_Partial(Z, shell, E):
    check_types_xrl((Z, shell, E), ni=2, nf=1)
    xrl_fcn = external_fcn(ni=2, nf=1)
    msg = f"{ZOOR} | {NE} | {US} | {IS}"

    def impl(Z, shell, E):
        e = _error()
        result = xrl_fcn(Z, shell, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 2 double --------------------------------- #


@overload_xrl
def _DCS_Compt(Z, E, theta):
    check_types_xrl((Z, E, theta), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta):
        e = _error()
        result = xrl_fcn(Z, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCS_Rayl(Z, E, theta):
    check_types_xrl((Z, E, theta), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta):
        e = _error()
        result = xrl_fcn(Z, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSb_Compt(Z, E, theta):
    check_types_xrl((Z, E, theta), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta):
        e = _error()
        result = xrl_fcn(Z, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSb_Rayl(Z, E, theta):
    check_types_xrl((Z, E, theta), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta):
        e = _error()
        result = xrl_fcn(Z, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL1_auger_cascade_kissel(Z, E, PK):
    check_types_xrl((Z, E, PK), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK):
        e = _error()
        result = xrl_fcn(Z, E, PK, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL1_full_cascade_kissel(Z, E, PK):
    check_types_xrl((Z, E, PK), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK):
        e = _error()
        result = xrl_fcn(Z, E, PK, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL1_rad_cascade_kissel(Z, E, PK):
    check_types_xrl((Z, E, PK), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK):
        e = _error()
        result = xrl_fcn(Z, E, PK, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL2_pure_kissel(Z, E, PL1):
    check_types_xrl((Z, E, PL1), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PL1):
        e = _error()
        result = xrl_fcn(Z, E, PL1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM2_pure_kissel(Z, E, PM1):
    check_types_xrl((Z, E, PM1), ni=1, nf=2)
    xrl_fcn = external_fcn(ni=1, nf=2)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PM1):
        e = _error()
        result = xrl_fcn(Z, E, PM1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 3 double --------------------------------- #


@overload_xrl
def _DCSP_Rayl(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSP_Compt(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSPb_Rayl(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSPb_Compt(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL2_auger_cascade_kissel(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL2_full_cascade_kissel(Z, E, theta, phi):
    check_types_xrl((Z, E, theta, phi), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL2_rad_cascade_kissel(Z, E, PK, PL1):
    check_types_xrl((Z, E, PK, PL1), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL3_pure_kissel(Z, E, PL1, PL2):
    check_types_xrl((Z, E, PL1, PL2), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PL1, PL2):
        e = _error()
        result = xrl_fcn(Z, E, PL1, PL2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM3_pure_kissel(Z, E, PM1, PM2):
    check_types_xrl((Z, E, PM1, PM2), ni=1, nf=3)
    xrl_fcn = external_fcn(ni=1, nf=3)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PM1, PM2):
        e = _error()
        result = xrl_fcn(Z, E, PM1, PM2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 4 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL3_auger_cascade_kissel(Z, E, theta, phi, PL1):
    check_types_xrl((Z, E, theta, phi, PL1), ni=1, nf=4)
    xrl_fcn = external_fcn(ni=1, nf=4)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PL1):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PL1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL3_full_cascade_kissel(Z, E, theta, phi, PL1):
    check_types_xrl((Z, E, theta, phi, PL1), ni=1, nf=4)
    xrl_fcn = external_fcn(ni=1, nf=4)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PL1):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PL1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2):
    check_types_xrl((Z, E, PK, PL1, PL2), ni=1, nf=4)
    xrl_fcn = external_fcn(ni=1, nf=4)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM4_pure_kissel(Z, E, theta, phi, PM1):
    check_types_xrl((Z, E, theta, phi, PM1), ni=1, nf=4)
    xrl_fcn = external_fcn(ni=1, nf=4)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM1):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM1, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 5 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM1_auger_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    check_types_xrl((Z, E, theta, phi, PM2, PM3), ni=1, nf=5)
    xrl_fcn = external_fcn(ni=1, nf=5)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM2, PM3):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM1_full_cascade_kissel(Z, E, theta, phi, PM2, PM3):
    check_types_xrl((Z, E, theta, phi, PM2, PM3), ni=1, nf=5)
    xrl_fcn = external_fcn(ni=1, nf=5)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM2, PM3):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM2, PM3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM1_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3):
    check_types_xrl((Z, E, PK, PL, PL2, PL3), ni=1, nf=5)
    xrl_fcn = external_fcn(ni=1, nf=5)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL, PL2, PL3):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM5_pure_kissel(Z, E, theta, phi, PM1, PM2):
    check_types_xrl((Z, E, theta, phi, PM1, PM2), ni=1, nf=5)
    xrl_fcn = external_fcn(ni=1, nf=5)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM1, PM2):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM1, PM2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 6 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM2_auger_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    check_types_xrl((Z, E, theta, phi, PM3, PM4, PM5), ni=1, nf=6)
    xrl_fcn = external_fcn(ni=1, nf=6)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM2_full_cascade_kissel(Z, E, theta, phi, PM3, PM4, PM5):
    check_types_xrl((Z, E, theta, phi, PM3, PM4, PM5), ni=1, nf=6)
    xrl_fcn = external_fcn(ni=1, nf=6)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, theta, phi, PM3, PM4, PM5):
        e = _error()
        result = xrl_fcn(Z, E, theta, phi, PM3, PM4, PM5, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM2_rad_cascade_kissel(Z, E, PK, PL, PL2, PL3, PL4):
    check_types_xrl((Z, E, PK, PL, PL2, PL3, PL4), ni=1, nf=6)
    xrl_fcn = external_fcn(ni=1, nf=6)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL, PL2, PL3, PL4):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL, PL2, PL3, PL4, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 7 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni=1, nf=7)
    xrl_fcn = external_fcn(ni=1, nf=7)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni=1, nf=7)
    xrl_fcn = external_fcn(ni=1, nf=7)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2), ni=1, nf=7)
    xrl_fcn = external_fcn(ni=1, nf=7)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 8 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni=1, nf=8)
    xrl_fcn = external_fcn(ni=1, nf=8)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni=1, nf=8)
    xrl_fcn = external_fcn(ni=1, nf=8)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3), ni=1, nf=8)
    xrl_fcn = external_fcn(ni=1, nf=8)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ---------------------------------- 1 int, 9 double --------------------------------- #


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni=1, nf=9)
    xrl_fcn = external_fcn(ni=1, nf=9)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni=1, nf=9)
    xrl_fcn = external_fcn(ni=1, nf=9)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# !!! Not implemented in xraylib_np
@overload_xrl
def _PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
    check_types_xrl((Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4), ni=1, nf=9)
    xrl_fcn = external_fcn(ni=1, nf=9)
    msg = f"{ZOOR} | {NE}"

    def impl(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4):
        e = _error()
        result = xrl_fcn(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# ------------------------------------- 3 double ------------------------------------- #


@overload_xrl
def _DCSP_KN(E, theta, phi):
    check_types_xrl((E, theta, phi), nf=3)
    xrl_fcn = external_fcn(nf=3)
    msg = f"{NE}"

    def impl(E, theta, phi):
        e = _error()
        result = xrl_fcn(E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# -------------------------------- 1 string, 1 double -------------------------------- #

# ??? How to pass a python string to an external function


@overload_xrl
def _CS_Total_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""  # TODO(nin17): Error message

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Photo_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""  # TODO(nin17): Error message

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Rayl_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Compt_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Energy_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Photo_Total_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CS_Total_Kissel_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Total_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Photo_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Rayl_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Compt_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Photo_Total_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _CSb_Total_Kissel_CP(compound, E):
    check_types_xrl((compound, E), ns=1, nf=1)
    xrl_fcn = external_fcn(ns=1, nf=1)

    msg = ""

    def impl(compound, E):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# -------------------------------- 1 string, 2 double -------------------------------- #


@overload_xrl
def _DCS_Rayl_CP(compound, E, theta):
    check_types_xrl((compound, E, theta), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, theta):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCS_Compt_CP(compound, E, theta):
    check_types_xrl((compound, E, theta), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, theta):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSb_Rayl_CP(compound, E, theta):
    check_types_xrl((compound, E, theta), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, theta):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSb_Compt_CP(compound, E, theta):
    check_types_xrl((compound, E, theta), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, theta):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _Refractive_Index_Re(compound, E, density):
    check_types_xrl((compound, E, density), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, density):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, density, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _Refractive_Index_Im(compound, E, density):
    check_types_xrl((compound, E, density), ns=1, nf=2)
    xrl_fcn = external_fcn(ns=1, nf=2)

    msg = ""

    def impl(compound, E, density):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, density, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# TODO(nin17): Refractive_Index
# TODO(nin17): complex return values

# -------------------------------- 1 string, 3 double -------------------------------- #


@overload_xrl
def _DCSP_Rayl_CP(compound, E, theta, phi):
    check_types_xrl((compound, E, theta, phi), ns=1, nf=3)
    xrl_fcn = external_fcn(ns=1, nf=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSP_Compt_CP(compound, E, theta, phi):
    check_types_xrl((compound, E, theta, phi), ns=1, nf=3)
    xrl_fcn = external_fcn(ns=1, nf=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSPb_Rayl_CP(compound, E, theta, phi):
    check_types_xrl((compound, E, theta, phi), ns=1, nf=3)
    xrl_fcn = external_fcn(ns=1, nf=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


@overload_xrl
def _DCSPb_Compt_CP(compound, E, theta, phi):
    check_types_xrl((compound, E, theta, phi), ns=1, nf=3)
    xrl_fcn = external_fcn(ns=1, nf=3)

    msg = ""

    def impl(compound, E, theta, phi):
        c = convert_str(compound)
        e = _error()
        result = xrl_fcn(c.ctypes, E, theta, phi, e.ctypes)
        if _check_error(e):
            raise ValueError(msg)
        return result

    return impl


# TODO(nin17): Other functions with string returns etc...
