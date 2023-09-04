"""_summary_
"""

# pylint: disable=invalid-name, too-many-arguments, not-an-iterable

import ctypes as ct
from ctypes import util


import numba as nb
import numpy as np
import xraylib as xrl

from .config import config


# # Load the xraylib shared library
_xrl = ct.CDLL(util.find_library("libxrl"))

# TODO: the other functions
# TODO: AtomicNumberToSymbol
# TODO: Atomic_Factors
# TODO: Bragg_angle
# TODO: CompoundParser
# TODO: SymbolToAtomicNumber
# TODO: docstrings
# ??? is the ct.c_void_p necessary?

# The functions below are sorted by number and types of input arguments

# 1 int
_sig_1_int = ct.c_int, ct.c_void_p

_AtomicWeight = _xrl.AtomicWeight
_AtomicWeight.argtypes = _sig_1_int
_AtomicWeight.restype = ct.c_double


@nb.njit(**config["xrl"].get("AtomicWeight", {}))
def AtomicWeight(Z: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _AtomicWeight(Z, 0)
    if result:
        return result
    raise ValueError("Invalid Z")


_ElementDensity = _xrl.ElementDensity
_ElementDensity.argtypes = ct.c_int, ct.c_void_p
_ElementDensity.restype = ct.c_double


@nb.njit(**config["xrl"].get("ElementDensity", {}))
def ElementDensity(Z: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _ElementDensity(Z, 0)
    if result:
        return result
    raise ValueError("Invalid Z")


# 1 double
_sig_1_double = ct.c_double, ct.c_void_p

_CS_KN = _xrl.CS_KN
_CS_KN.argtypes = _sig_1_double
_CS_KN.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_KN", {}))
def CS_KN(E: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_KN(E, 0)
    if result:
        return result
    raise ValueError("Invalid E")


_DCS_Thoms = _xrl.DCS_Thoms
_DCS_Thoms.argtypes = _sig_1_double
_DCS_Thoms.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCS_Thoms", {}))
def DCS_Thoms(theta: float) -> float:
    """_summary_

    Parameters
    ----------
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCS_Thoms(theta, 0)
    if result:
        return result
    raise ValueError("Invalid theta")


# 2 ints
_sig_2_int = ct.c_int, ct.c_int, ct.c_void_p

_AtomicLevelWidth = _xrl.AtomicLevelWidth
_AtomicLevelWidth.argtypes = _sig_2_int
_AtomicLevelWidth.restype = ct.c_double


@nb.njit(**config["xrl"].get("AtomicLevelWidth", {}))
def AtomicLevelWidth(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _AtomicLevelWidth(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_AugerRate = _xrl.AugerRate
_AugerRate.argtypes = _sig_2_int
_AugerRate.restype = ct.c_double


@nb.njit(**config["xrl"].get("AugerRate", {}))
def AugerRate(Z: int, auger_trans: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    auger_trans : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _AugerRate(Z, auger_trans, 0)
    if result:
        return result
    raise ValueError("Invalid Z | auger_trans")


_AugerYield = _xrl.AugerYield
_AugerYield.argtypes = _sig_2_int
_AugerYield.restype = ct.c_double


@nb.njit(**config["xrl"].get("AugerYield", {}))
def AugerYield(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _AugerYield(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_CosKronTransProb = _xrl.CosKronTransProb
_CosKronTransProb.argtypes = _sig_2_int
_CosKronTransProb.restype = ct.c_double


@nb.njit(**config["xrl"].get("CosKronTransProb", {}))
def CosKronTransProb(Z: int, trans: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    trans : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CosKronTransProb(Z, trans, 0)
    if result:
        return result
    raise ValueError("Invalid Z | trans")


_EdgeEnergy = _xrl.EdgeEnergy
_EdgeEnergy.argtypes = _sig_2_int
_EdgeEnergy.restype = ct.c_double


@nb.njit(**config["xrl"].get("EdgeEnergy", {}))
def EdgeEnergy(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _EdgeEnergy(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_ElectronConfig = _xrl.ElectronConfig
_ElectronConfig.argtypes = _sig_2_int
_ElectronConfig.restype = ct.c_double


@nb.njit(**config["xrl"].get("ElectronConfig", {}))
def ElectronConfig(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _ElectronConfig(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_FluorYield = _xrl.FluorYield
_FluorYield.argtypes = _sig_2_int
_FluorYield.restype = ct.c_double


@nb.njit(**config["xrl"].get("FluorYield", {}))
def FluorYield(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _FluorYield(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_JumpFactor = _xrl.JumpFactor
_JumpFactor.argtypes = _sig_2_int
_JumpFactor.restype = ct.c_double


@nb.njit(**config["xrl"].get("JumpFactor", {}))
def JumpFactor(Z: int, shell: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _JumpFactor(Z, shell, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell")


_LineEnergy = _xrl.LineEnergy
_LineEnergy.argtypes = _sig_2_int
_LineEnergy.restype = ct.c_double


@nb.njit(**config["xrl"].get("LineEnergy", {}))
def LineEnergy(Z: int, line: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _LineEnergy(Z, line, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line")


_RadRate = _xrl.RadRate
_RadRate.argtypes = _sig_2_int
_RadRate.restype = ct.c_double


@nb.njit(**config["xrl"].get("RadRate", {}))
def RadRate(Z: int, line: int) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _RadRate(Z, line, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line")


# 2 double
_sig_2_double = ct.c_double, ct.c_double, ct.c_void_p

_ComptonEnergy = _xrl.ComptonEnergy
_ComptonEnergy.argtypes = _sig_2_double
_ComptonEnergy.restype = ct.c_double


@nb.njit(**config["xrl"].get("ComptonEnergy", {}))
def ComptonEnergy(E0: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    E0 : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _ComptonEnergy(E0, theta, 0)
    if result:
        return result
    raise ValueError("Invalid E0 | theta")


_DCS_KN = _xrl.DCS_KN
_DCS_KN.argtypes = _sig_2_double
_DCS_KN.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCS_KN", {}))
def DCS_KN(E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCS_KN(E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid E | theta")


_DCSP_Thoms = _xrl.DCSP_Thoms
_DCSP_Thoms.argtypes = _sig_2_double
_DCSP_Thoms.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSP_Thoms", {}))
def DCSP_Thoms(theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSP_Thoms(theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid theta | phi")


_MomentTransf = _xrl.MomentTransf
_MomentTransf.argtypes = _sig_2_double
_MomentTransf.restype = ct.c_double


@nb.njit(**config["xrl"].get("MomentTransf", {}))
def MomentTransf(E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _MomentTransf(E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid E | theta")


# 1 int, 1 double
_sig_1_int_1_double = ct.c_int, ct.c_double, ct.c_void_p


_ComptonProfile = _xrl.ComptonProfile
_ComptonProfile.argtypes = _sig_1_int_1_double
_ComptonProfile.restype = ct.c_double


@nb.njit(**config["xrl"].get("ComptonProfile", {}))
def ComptonProfile(Z: int, pz: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    pz : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _ComptonProfile(Z, pz, 0)
    if result:
        return result
    raise ValueError("Invalid Z | pz")


_CS_Compt = _xrl.CS_Compt
_CS_Compt.argtypes = _sig_1_int_1_double
_CS_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Compt", {}))
def CS_Compt(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Compt(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Energy = _xrl.CS_Energy
_CS_Energy.argtypes = _sig_1_int_1_double
_CS_Energy.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Energy", {}))
def CS_Energy(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Energy(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Photo = _xrl.CS_Photo
_CS_Photo.argtypes = _sig_1_int_1_double
_CS_Photo.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Photo", {}))
def CS_Photo(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Photo(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Photo_Total = _xrl.CS_Photo_Total
_CS_Photo_Total.argtypes = _sig_1_int_1_double
_CS_Photo_Total.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Photo_Total", {}))
def CS_Photo_Total(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Photo_Total(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Rayl = _xrl.CS_Rayl
_CS_Rayl.argtypes = _sig_1_int_1_double
_CS_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Rayl", {}))
def CS_Rayl(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Rayl(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Total = _xrl.CS_Total
_CS_Total.argtypes = _sig_1_int_1_double
_CS_Total.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Total", {}))
def CS_Total(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Total(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CS_Total_Kissel = _xrl.CS_Total_Kissel
_CS_Total_Kissel.argtypes = _sig_1_int_1_double
_CS_Total_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Total_Kissel", {}))
def CS_Total_Kissel(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Total_Kissel(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Compt = _xrl.CSb_Compt
_CSb_Compt.argtypes = _sig_1_int_1_double
_CSb_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Compt", {}))
def CSb_Compt(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Compt(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Photo = _xrl.CSb_Photo
_CSb_Photo.argtypes = _sig_1_int_1_double
_CSb_Photo.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Photo", {}))
def CSb_Photo(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Photo(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Photo_Total = _xrl.CSb_Photo_Total
_CSb_Photo_Total.argtypes = _sig_1_int_1_double
_CSb_Photo_Total.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Photo_Total", {}))
def CSb_Photo_Total(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Photo_Total(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Rayl = _xrl.CSb_Rayl
_CSb_Rayl.argtypes = _sig_1_int_1_double
_CSb_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Rayl", {}))
def CSb_Rayl(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Rayl(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Total = _xrl.CSb_Total
_CSb_Total.argtypes = _sig_1_int_1_double
_CSb_Total.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Total", {}))
def CSb_Total(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Total(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_CSb_Total_Kissel = _xrl.CSb_Total_Kissel
_CSb_Total_Kissel.argtypes = _sig_1_int_1_double
_CSb_Total_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Total_Kissel", {}))
def CSb_Total_Kissel(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Total_Kissel(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_FF_Rayl = _xrl.FF_Rayl
_FF_Rayl.argtypes = _sig_1_int_1_double
_FF_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("FF_Rayl", {}))
def FF_Rayl(Z: int, q: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    q : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _FF_Rayl(Z, q, 0)
    if result:
        return result
    raise ValueError("Invalid Z | q")


_SF_Compt = _xrl.SF_Compt
_SF_Compt.argtypes = _sig_1_int_1_double
_SF_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("SF_Compt", {}))
def SF_Compt(Z: int, q: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    q : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _SF_Compt(Z, q, 0)
    if result:
        return result
    raise ValueError("Invalid Z | q")


_Fi = _xrl.Fi
_Fi.argtypes = _sig_1_int_1_double
_Fi.restype = ct.c_double


@nb.njit(**config["xrl"].get("Fi", {}))
def Fi(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _Fi(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_Fii = _xrl.Fii
_Fii.argtypes = _sig_1_int_1_double
_Fii.restype = ct.c_double


@nb.njit(**config["xrl"].get("Fii", {}))
def Fii(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _Fii(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_PL1_pure_kissel = _xrl.PL1_pure_kissel
_PL1_pure_kissel.argtypes = _sig_1_int_1_double
_PL1_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL1_pure_kissel", {}))
def PL1_pure_kissel(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL1_pure_kissel(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


_PM1_pure_kissel = _xrl.PM1_pure_kissel
_PM1_pure_kissel.argtypes = _sig_1_int_1_double
_PM1_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM1_pure_kissel", {}))
def PM1_pure_kissel(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM1_pure_kissel(Z, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E")


# 2 int, 1 double
_sig_2_int_1_double = ct.c_int, ct.c_int, ct.c_double, ct.c_void_p


_ComptonProfile_Partial = _xrl.ComptonProfile_Partial
_ComptonProfile_Partial.argtypes = _sig_2_int_1_double
_ComptonProfile_Partial.restype = ct.c_double


@nb.njit(**config["xrl"].get("ComptonProfile_Partial", {}))
def ComptonProfile_Partial(Z: int, shell: int, pz: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    pz : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _ComptonProfile_Partial(Z, shell, pz, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | pz")


_CS_FluorLine_Kissel = _xrl.CS_FluorLine_Kissel
_CS_FluorLine_Kissel.argtypes = _sig_2_int_1_double
_CS_FluorLine_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine_Kissel", {}))
def CS_FluorLine_Kissel(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine_Kissel(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine_Kissel = _xrl.CSb_FluorLine_Kissel
_CSb_FluorLine_Kissel.argtypes = _sig_2_int_1_double
_CSb_FluorLine_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine_Kissel", {}))
def CSb_FluorLine_Kissel(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine_Kissel(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorLine_Kissel_Cascade = _xrl.CS_FluorLine_Kissel_Cascade
_CS_FluorLine_Kissel_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorLine_Kissel_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine_Kissel_Cascade", {}))
def CS_FluorLine_Kissel_Cascade(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine_Kissel_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine_Kissel_Cascade = _xrl.CSb_FluorLine_Kissel_Cascade
_CSb_FluorLine_Kissel_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorLine_Kissel_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine_Kissel_Cascade", {}))
def CSb_FluorLine_Kissel_Cascade(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine_Kissel_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorLine_Kissel_no_Cascade = _xrl.CS_FluorLine_Kissel_no_Cascade
_CS_FluorLine_Kissel_no_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorLine_Kissel_no_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine_Kissel_no_Cascade", {}))
def CS_FluorLine_Kissel_no_Cascade(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine_Kissel_no_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine_Kissel_no_Cascade = _xrl.CSb_FluorLine_Kissel_no_Cascade
_CSb_FluorLine_Kissel_no_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorLine_Kissel_no_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine_Kissel_no_Cascade", {}))
def CSb_FluorLine_Kissel_no_Cascade(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine_Kissel_no_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorLine_Kissel_Nonradiative_Cascade = (
    _xrl.CS_FluorLine_Kissel_Nonradiative_Cascade
)
_CS_FluorLine_Kissel_Nonradiative_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorLine_Kissel_Nonradiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine_Kissel_Nonradiative_Cascade", {}))
def CS_FluorLine_Kissel_Nonradiative_Cascade(
    Z: int, line: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine_Kissel_Nonradiative_Cascade = (
    _xrl.CSb_FluorLine_Kissel_Nonradiative_Cascade
)
_CSb_FluorLine_Kissel_Nonradiative_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorLine_Kissel_Nonradiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine_Kissel_Nonradiative_Cascade", {}))
def CSb_FluorLine_Kissel_Nonradiative_Cascade(
    Z: int, line: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorLine_Kissel_Radiative_Cascade = (
    _xrl.CS_FluorLine_Kissel_Radiative_Cascade
)
_CS_FluorLine_Kissel_Radiative_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorLine_Kissel_Radiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine_Kissel_Radiative_Cascade", {}))
def CS_FluorLine_Kissel_Radiative_Cascade(
    Z: int, line: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine_Kissel_Radiative_Cascade = (
    _xrl.CSb_FluorLine_Kissel_Radiative_Cascade
)
_CSb_FluorLine_Kissel_Radiative_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorLine_Kissel_Radiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine_Kissel_Radiative_Cascade", {}))
def CSb_FluorLine_Kissel_Radiative_Cascade(
    Z: int, line: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorShell_Kissel = _xrl.CS_FluorShell_Kissel
_CS_FluorShell_Kissel.argtypes = _sig_2_int_1_double
_CS_FluorShell_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell_Kissel", {}))
def CS_FluorShell_Kissel(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell_Kissel(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell_Kissel = _xrl.CSb_FluorShell_Kissel
_CSb_FluorShell_Kissel.argtypes = _sig_2_int_1_double
_CSb_FluorShell_Kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell_Kissel", {}))
def CSb_FluorShell_Kissel(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell_Kissel(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_FluorShell_Kissel_Cascade = _xrl.CS_FluorShell_Kissel_Cascade
_CS_FluorShell_Kissel_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorShell_Kissel_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell_Kissel_Cascade", {}))
def CS_FluorShell_Kissel_Cascade(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell_Kissel_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell_Kissel_Cascade = _xrl.CSb_FluorShell_Kissel_Cascade
_CSb_FluorShell_Kissel_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorShell_Kissel_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell_Kissel_Cascade", {}))
def CSb_FluorShell_Kissel_Cascade(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell_Kissel_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_FluorShell_Kissel_no_Cascade = _xrl.CS_FluorShell_Kissel_no_Cascade
_CS_FluorShell_Kissel_no_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorShell_Kissel_no_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell_Kissel_no_Cascade", {}))
def CS_FluorShell_Kissel_no_Cascade(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell_Kissel_no_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell_Kissel_no_Cascade = _xrl.CSb_FluorShell_Kissel_no_Cascade
_CSb_FluorShell_Kissel_no_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorShell_Kissel_no_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell_Kissel_no_Cascade", {}))
def CSb_FluorShell_Kissel_no_Cascade(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell_Kissel_no_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_FluorShell_Kissel_Nonradiative_Cascade = (
    _xrl.CS_FluorShell_Kissel_Nonradiative_Cascade
)
_CS_FluorShell_Kissel_Nonradiative_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorShell_Kissel_Nonradiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell_Kissel_Nonradiative_Cascade", {}))
def CS_FluorShell_Kissel_Nonradiative_Cascade(
    Z: int, shell: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell_Kissel_Nonradiative_Cascade = (
    _xrl.CSb_FluorShell_Kissel_Nonradiative_Cascade
)
_CSb_FluorShell_Kissel_Nonradiative_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorShell_Kissel_Nonradiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell_Kissel_Nonradiative_Cascade", {}))
def CSb_FluorShell_Kissel_Nonradiative_Cascade(
    Z: int, shell: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_FluorShell_Kissel_Radiative_Cascade = (
    _xrl.CS_FluorShell_Kissel_Radiative_Cascade
)
_CS_FluorShell_Kissel_Radiative_Cascade.argtypes = _sig_2_int_1_double
_CS_FluorShell_Kissel_Radiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell_Kissel_Radiative_Cascade", {}))
def CS_FluorShell_Kissel_Radiative_Cascade(
    Z: int, shell: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell_Kissel_Radiative_Cascade = (
    _xrl.CSb_FluorShell_Kissel_Radiative_Cascade
)
_CSb_FluorShell_Kissel_Radiative_Cascade.argtypes = _sig_2_int_1_double
_CSb_FluorShell_Kissel_Radiative_Cascade.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell_Kissel_Radiative_Cascade", {}))
def CSb_FluorShell_Kissel_Radiative_Cascade(
    Z: int, shell: int, E: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_FluorLine = _xrl.CS_FluorLine
_CS_FluorLine.argtypes = _sig_2_int_1_double
_CS_FluorLine.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorLine", {}))
def CS_FluorLine(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorLine(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CSb_FluorLine = _xrl.CSb_FluorLine
_CSb_FluorLine.argtypes = _sig_2_int_1_double
_CSb_FluorLine.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorLine", {}))
def CSb_FluorLine(Z: int, line: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    line : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorLine(Z, line, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | line | E")


_CS_FluorShell = _xrl.CS_FluorShell
_CS_FluorShell.argtypes = _sig_2_int_1_double
_CS_FluorShell.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_FluorShell", {}))
def CS_FluorShell(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_FluorShell(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_FluorShell = _xrl.CSb_FluorShell
_CSb_FluorShell.argtypes = _sig_2_int_1_double
_CSb_FluorShell.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_FluorShell", {}))
def CSb_FluorShell(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_FluorShell(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CS_Photo_Partial = _xrl.CS_Photo_Partial
_CS_Photo_Partial.argtypes = _sig_2_int_1_double
_CS_Photo_Partial.restype = ct.c_double


@nb.njit(**config["xrl"].get("CS_Photo_Partial", {}))
def CS_Photo_Partial(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CS_Photo_Partial(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


_CSb_Photo_Partial = _xrl.CSb_Photo_Partial
_CSb_Photo_Partial.argtypes = _sig_2_int_1_double
_CSb_Photo_Partial.restype = ct.c_double


@nb.njit(**config["xrl"].get("CSb_Photo_Partial", {}))
def CSb_Photo_Partial(Z: int, shell: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    shell : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _CSb_Photo_Partial(Z, shell, E, 0)
    if result:
        return result
    raise ValueError("Invalid Z | shell | E")


# 1 int, 2 double
_sig_1_int_2_double = ct.c_int, ct.c_double, ct.c_double, ct.c_void_p


_DCS_Compt = _xrl.DCS_Compt
_DCS_Compt.argtypes = _sig_1_int_2_double
_DCS_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCS_Compt", {}))
def DCS_Compt(Z: int, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCS_Compt(Z, E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta")


_DCS_Rayl = _xrl.DCS_Rayl
_DCS_Rayl.argtypes = _sig_1_int_2_double
_DCS_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCS_Rayl", {}))
def DCS_Rayl(Z: int, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCS_Rayl(Z, E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta")


_DCSb_Compt = _xrl.DCSb_Compt
_DCSb_Compt.argtypes = _sig_1_int_2_double
_DCSb_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSb_Compt", {}))
def DCSb_Compt(Z: int, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSb_Compt(Z, E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta")


_DCSb_Rayl = _xrl.DCSb_Rayl
_DCSb_Rayl.argtypes = _sig_1_int_2_double
_DCSb_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSb_Rayl", {}))
def DCSb_Rayl(Z: int, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSb_Rayl(Z, E, theta, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta")


_PL1_auger_cascade_kissel = _xrl.PL1_auger_cascade_kissel
_PL1_auger_cascade_kissel.argtypes = _sig_1_int_2_double
_PL1_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL1_auger_cascade_kissel", {}))
def PL1_auger_cascade_kissel(Z: int, E: float, PK: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL1_auger_cascade_kissel(Z, E, PK, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK")


_PL1_full_cascade_kissel = _xrl.PL1_full_cascade_kissel
_PL1_full_cascade_kissel.argtypes = _sig_1_int_2_double
_PL1_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL1_full_cascade_kissel", {}))
def PL1_full_cascade_kissel(Z: int, E: float, PK: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL1_full_cascade_kissel(Z, E, PK, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK")


_PL1_rad_cascade_kissel = _xrl.PL1_rad_cascade_kissel
_PL1_rad_cascade_kissel.argtypes = _sig_1_int_2_double
_PL1_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL1_rad_cascade_kissel", {}))
def PL1_rad_cascade_kissel(Z: int, E: float, PK: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL1_rad_cascade_kissel(Z, E, PK, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK")


_PL2_pure_kissel = _xrl.PL2_pure_kissel
_PL2_pure_kissel.argtypes = _sig_1_int_2_double
_PL2_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL2_pure_kissel", {}))
def PL2_pure_kissel(Z: int, E: float, PL1: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PL1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL2_pure_kissel(Z, E, PL1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PL1")


_PM2_pure_kissel = _xrl.PM2_pure_kissel
_PM2_pure_kissel.argtypes = _sig_1_int_2_double
_PM2_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM2_pure_kissel", {}))
def PM2_pure_kissel(Z: int, E: float, PM1: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PM1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM2_pure_kissel(Z, E, PM1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PM1")


# 1 int, 3 double
_sig_1_int_3_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)


_DCSP_Rayl = _xrl.DCSP_Rayl
_DCSP_Rayl.argtypes = _sig_1_int_3_double
_DCSP_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSP_Rayl", {}))
def DCSP_Rayl(Z: int, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSP_Rayl(Z, E, theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta | phi")


_DCSP_Compt = _xrl.DCSP_Compt
_DCSP_Compt.argtypes = _sig_1_int_3_double
_DCSP_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSP_Compt", {}))
def DCSP_Compt(Z: int, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSP_Compt(Z, E, theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta | phi")


_DCSPb_Rayl = _xrl.DCSPb_Rayl
_DCSPb_Rayl.argtypes = _sig_1_int_3_double
_DCSPb_Rayl.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSPb_Rayl", {}))
def DCSPb_Rayl(Z: int, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSPb_Rayl(Z, E, theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta | phi")


_DCSPb_Compt = _xrl.DCSPb_Compt
_DCSPb_Compt.argtypes = _sig_1_int_3_double
_DCSPb_Compt.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSPb_Compt", {}))
def DCSPb_Compt(Z: int, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSPb_Compt(Z, E, theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | theta | phi")


_PL2_auger_cascade_kissel = _xrl.PL2_auger_cascade_kissel
_PL2_auger_cascade_kissel.argtypes = _sig_1_int_3_double
_PL2_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL2_auger_cascade_kissel", {}))
def PL2_auger_cascade_kissel(Z: int, E: float, PK: float, PL1: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL2_auger_cascade_kissel(Z, E, PK, PL1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1")


_PL2_full_cascade_kissel = _xrl.PL2_full_cascade_kissel
_PL2_full_cascade_kissel.argtypes = _sig_1_int_3_double
_PL2_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL2_full_cascade_kissel", {}))
def PL2_full_cascade_kissel(Z: int, E: float, PK: float, PL1: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL2_full_cascade_kissel(Z, E, PK, PL1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1")


_PL2_rad_cascade_kissel = _xrl.PL2_rad_cascade_kissel
_PL2_rad_cascade_kissel.argtypes = _sig_1_int_3_double
_PL2_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL2_rad_cascade_kissel", {}))
def PL2_rad_cascade_kissel(Z: int, E: float, PK: float, PL1: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL2_rad_cascade_kissel(Z, E, PK, PL1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1")


_PL3_pure_kissel = _xrl.PL3_pure_kissel
_PL3_pure_kissel.argtypes = _sig_1_int_3_double
_PL3_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL3_pure_kissel", {}))
def PL3_pure_kissel(Z: int, E: float, PL1: float, PL2: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL3_pure_kissel(Z, E, PL1, PL2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PL1 | PL2")


_PM3_pure_kissel = _xrl.PM3_pure_kissel
_PM3_pure_kissel.argtypes = _sig_1_int_3_double
_PM3_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM3_pure_kissel", {}))
def PM3_pure_kissel(Z: int, E: float, PM1: float, PM2: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM3_pure_kissel(Z, E, PM1, PM2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PM1 | PM2")


# 1 int, 4 double
_sig_1_int_4_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)


_PL3_auger_cascade_kissel = _xrl.PL3_auger_cascade_kissel
_PL3_auger_cascade_kissel.argtypes = _sig_1_int_4_double
_PL3_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL3_auger_cascade_kissel", {}))
def PL3_auger_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2")


_PL3_full_cascade_kissel = _xrl.PL3_full_cascade_kissel
_PL3_full_cascade_kissel.argtypes = _sig_1_int_4_double
_PL3_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL3_full_cascade_kissel", {}))
def PL3_full_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL3_full_cascade_kissel(Z, E, PK, PL1, PL2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2")


_PL3_rad_cascade_kissel = _xrl.PL3_rad_cascade_kissel
_PL3_rad_cascade_kissel.argtypes = _sig_1_int_4_double
_PL3_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PL3_rad_cascade_kissel", {}))
def PL3_rad_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2")


_PM4_pure_kissel = _xrl.PM4_pure_kissel
_PM4_pure_kissel.argtypes = _sig_1_int_4_double
_PM4_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM4_pure_kissel", {}))
def PM4_pure_kissel(
    Z: int, E: float, PM1: float, PM2: float, PM3: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM4_pure_kissel(Z, E, PM1, PM2, PM3, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PM1 | PM2 | PM3")


# 1 int, 5 double

_sig_1_int_5_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)


_PM1_auger_cascade_kissel = _xrl.PM1_auger_cascade_kissel
_PM1_auger_cascade_kissel.argtypes = _sig_1_int_5_double
_PM1_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM1_auger_cascade_kissel", {}))
def PM1_auger_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3")


_PM1_full_cascade_kissel = _xrl.PM1_full_cascade_kissel
_PM1_full_cascade_kissel.argtypes = _sig_1_int_5_double
_PM1_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM1_full_cascade_kissel", {}))
def PM1_full_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3")


_PM1_rad_cascade_kissel = _xrl.PM1_rad_cascade_kissel
_PM1_rad_cascade_kissel.argtypes = _sig_1_int_5_double
_PM1_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM1_rad_cascade_kissel", {}))
def PM1_rad_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3")


_PM5_pure_kissel = _xrl.PM5_pure_kissel
_PM5_pure_kissel.argtypes = _sig_1_int_5_double
_PM5_pure_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM5_pure_kissel", {}))
def PM5_pure_kissel(
    Z: int, E: float, PM1: float, PM2: float, PM3: float, PM4: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_
    PM4 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM5_pure_kissel(Z, E, PM1, PM2, PM3, PM4, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PM1 | PM2 | PM3 | PM4")


# 1 int, 6 double

_sig_1_int_6_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)

_PM2_auger_cascade_kissel = _xrl.PM2_auger_cascade_kissel
_PM2_auger_cascade_kissel.argtypes = _sig_1_int_6_double
_PM2_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM2_auger_cascade_kissel", {}))
def PM2_auger_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float, PM1: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1")


_PM2_full_cascade_kissel = _xrl.PM2_full_cascade_kissel
_PM2_full_cascade_kissel.argtypes = _sig_1_int_6_double
_PM2_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM2_full_cascade_kissel", {}))
def PM2_full_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float, PM1: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1")


_PM2_rad_cascade_kissel = _xrl.PM2_rad_cascade_kissel
_PM2_rad_cascade_kissel.argtypes = _sig_1_int_6_double
_PM2_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM2_rad_cascade_kissel", {}))
def PM2_rad_cascade_kissel(
    Z: int, E: float, PK: float, PL1: float, PL2: float, PL3: float, PM1: float
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1")


# 1 int, 7 double

_sig_1_int_7_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)


_PM3_auger_cascade_kissel = _xrl.PM3_auger_cascade_kissel
_PM3_auger_cascade_kissel.argtypes = _sig_1_int_7_double
_PM3_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM3_auger_cascade_kissel", {}))
def PM3_auger_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2")


_PM3_full_cascade_kissel = _xrl.PM3_full_cascade_kissel
_PM3_full_cascade_kissel.argtypes = _sig_1_int_7_double
_PM3_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM3_full_cascade_kissel", {}))
def PM3_full_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2")


_PM3_rad_cascade_kissel = _xrl.PM3_rad_cascade_kissel
_PM3_rad_cascade_kissel.argtypes = _sig_1_int_7_double
_PM3_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM3_rad_cascade_kissel", {}))
def PM3_rad_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2")


# 1 int, 8 double

_sig_1_int_8_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)

_PM4_auger_cascade_kissel = _xrl.PM4_auger_cascade_kissel
_PM4_auger_cascade_kissel.argtypes = _sig_1_int_8_double
_PM4_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM4_auger_cascade_kissel", {}))
def PM4_auger_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM4_auger_cascade_kissel(
        Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, 0
    )
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3")


_PM4_full_cascade_kissel = _xrl.PM4_full_cascade_kissel
_PM4_full_cascade_kissel.argtypes = _sig_1_int_8_double
_PM4_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM4_full_cascade_kissel", {}))
def PM4_full_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM4_full_cascade_kissel(
        Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, 0
    )
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3")


_PM4_rad_cascade_kissel = _xrl.PM4_rad_cascade_kissel
_PM4_rad_cascade_kissel.argtypes = _sig_1_int_8_double
_PM4_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM4_rad_cascade_kissel", {}))
def PM4_rad_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_
    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_
    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_

    PM3 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, 0)
    if result:
        return result
    raise ValueError("Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3")


# 1 int, 9 double

_sig_1_int_9_double = (
    ct.c_int,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_double,
    ct.c_void_p,
)


_PM5_auger_cascade_kissel = _xrl.PM5_auger_cascade_kissel
_PM5_auger_cascade_kissel.argtypes = _sig_1_int_9_double
_PM5_auger_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM5_auger_cascade_kissel", {}))
def PM5_auger_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
    PM4: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_
    PM4 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM5_auger_cascade_kissel(
        Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, 0
    )
    if result:
        return result
    raise ValueError(
        "Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3 | PM4"
    )


_PM5_full_cascade_kissel = _xrl.PM5_full_cascade_kissel
_PM5_full_cascade_kissel.argtypes = _sig_1_int_9_double
_PM5_full_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM5_full_cascade_kissel", {}))
def PM5_full_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
    PM4: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_
    PM4 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM5_full_cascade_kissel(
        Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, 0
    )
    if result:
        return result
    raise ValueError(
        "Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3 | PM4"
    )


_PM5_rad_cascade_kissel = _xrl.PM5_rad_cascade_kissel
_PM5_rad_cascade_kissel.argtypes = _sig_1_int_9_double
_PM5_rad_cascade_kissel.restype = ct.c_double


@nb.njit(**config["xrl"].get("PM5_rad_cascade_kissel", {}))
def PM5_rad_cascade_kissel(
    Z: int,
    E: float,
    PK: float,
    PL1: float,
    PL2: float,
    PL3: float,
    PM1: float,
    PM2: float,
    PM3: float,
    PM4: float,
) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    PK : float
        _description_
    PL1 : float
        _description_
    PL2 : float
        _description_

    PL3 : float
        _description_
    PM1 : float
        _description_
    PM2 : float
        _description_
    PM3 : float
        _description_
    PM4 : float
        _description_

    Returns
    -------
    float
        _description_
    """
    result = _PM5_rad_cascade_kissel(
        Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, 0
    )
    if result:
        return result
    raise ValueError(
        "Invalid Z | E | PK | PL1 | PL2 | PL3 | PM1 | PM2 | PM3 | PM4"
    )


# 3 double
_sig_3_double = ct.c_double, ct.c_double, ct.c_double, ct.c_void_p


_DCSP_KN = _xrl.DCSP_KN
_DCSP_KN.argtypes = _sig_3_double
_DCSP_KN.restype = ct.c_double


@nb.njit(**config["xrl"].get("DCSP_KN", {}))
def DCSP_KN(E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    result = _DCSP_KN(E, theta, phi, 0)
    if result:
        return result
    raise ValueError("Invalid E | theta | phi")


# TODO figure out how to pass a string from numba to c function


# TODO wrap CompoundParser
@nb.njit()
def _parser(compound):
    with nb.objmode(
        elements="int64[:]", mass_fractions="float64[:]", n_elements="int64"
    ):
        parsed_compound = xrl.CompoundParser(compound)
        elements = np.array(parsed_compound["Elements"])
        mass_fractions = np.array(parsed_compound["massFractions"])
        n_elements = parsed_compound["nElements"]

    return elements, mass_fractions, n_elements


# 1 string, 1 double


@nb.njit(**config["xrl"].get("CS_Total_CP", {}))
def CS_Total_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Total(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Photo_CP", {}))
def CS_Photo_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Photo(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Rayl_CP", {}))
def CS_Rayl_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Rayl(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Compt_CP", {}))
def CS_Compt_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Compt(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Energy_CP", {}))
def CS_Energy_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Energy(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Photo_Total_CP", {}))
def CS_Photo_Total_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Photo_Total(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CS_Total_Kissel_CP", {}))
def CS_Total_Kissel_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CS_Total_Kissel(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Total_CP", {}))
def CSb_Total_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Total(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Photo_CP", {}))
def CSb_Photo_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Photo(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Rayl_CP", {}))
def CSb_Rayl_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Rayl(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Compt_CP", {}))
def CSb_Compt_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Compt(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Photo_Total_CP", {}))
def CSb_Photo_Total_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Photo_Total(e[i], E, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("CSb_Total_Kissel_CP", {}))
def CSb_Total_Kissel_CP(compound: str, E: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _CSb_Total_Kissel(e[i], E, 0) * m[i]
    return result


# 1 string, 2 double


@nb.njit(**config["xrl"].get("DCS_Rayl_CP", {}))
def DCS_Rayl_CP(compound: str, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCS_Rayl(e[i], E, theta, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCS_Compt_CP", {}))
def DCS_Compt_CP(compound: str, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCS_Compt(e[i], E, theta, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCSb_Rayl_CP", {}))
def DCSb_Rayl_CP(compound: str, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSb_Rayl(e[i], E, theta, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCSb_Compt_CP", {}))
def DCSb_Compt_CP(compound: str, E: float, theta: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSb_Compt(e[i], E, theta, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("Refractive_Index_Im", {}))
def Refractive_Index_Im(compound: str, E: float, density: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    density : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    e, m, n = _parser(compound)

    if E <= 0:
        raise ValueError("Invalid E")
    if density <= 0:
        raise ValueError("Invalid density")
    result = 0.0
    for i in nb.prange(n):
        cs = _CS_Total(e[i], E, 0)
        if cs == 0:
            raise ValueError("Invalid Z | E | density")
        result += cs * m[i]
    return result * density * 9.8663479e-9 / E


@nb.njit(**config["xrl"].get("Refractive_Index_Re", {}))
def Refractive_Index_Re(compound: str, E: float, density: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    density : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    KD = 4.15179082788e-4
    e, m, n = _parser(compound)
    if E <= 0:
        raise ValueError("Invalid E")
    if density <= 0:
        raise ValueError("Invalid density")
    result = 0.0
    for i in nb.prange(n):
        fi = _Fi(e[i], E, 0)
        if fi == 0:
            raise ValueError("Invalid Z | E | density")
        atomic_weight = _AtomicWeight(e[i], 0)
        if atomic_weight == 0:
            raise ValueError("Invalid Z | E | density")
        result += m[i] * KD * (e[i] + fi) / atomic_weight / E / E
    return 1.0 - (result * density)


@nb.njit(**config["xrl"].get("Refractive_Index", {}))
def Refractive_Index(compound: str, E: float, density: float) -> float:
    KD = 4.15179082788e-4
    e, m, n = _parser(compound)
    if E <= 0:
        raise ValueError("Invalid E")
    if density <= 0:
        raise ValueError("Invalid density")
    result_re = 0.0
    result_im = 0.0
    for i in nb.prange(n):
        fi = _Fi(e[i], E, 0)
        if fi == 0:
            raise ValueError("Invalid Z | E | density")
        atomic_weight = _AtomicWeight(e[i], 0)
        if atomic_weight == 0:
            raise ValueError("Invalid Z | E | density")
        cs = _CS_Total(e[i], E, 0)
        if cs == 0:
            raise ValueError("Invalid Z | E | density")

        result_im += cs * m[i]
        result_re += m[i] * KD * (e[i] + fi) / atomic_weight / E / E

    return (
        1.0
        - (result_re * density)
        + 1j * (result_im * density * 9.8663479e-9 / E)
    )


# 1 string, 3 double


@nb.njit(**config["xrl"].get("DCSP_Rayl_CP", {}))
def DCSP_Rayl_CP(compound: str, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSP_Rayl(e[i], E, theta, phi, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCSP_Compt_CP", {}))
def DCSP_Compt_CP(compound: str, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSP_Compt(e[i], E, theta, phi, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCSPb_Rayl_CP", {}))
def DCSPb_Rayl_CP(compound: str, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSPb_Rayl(e[i], E, theta, phi, 0) * m[i]
    return result


@nb.njit(**config["xrl"].get("DCSPb_Compt_CP", {}))
def DCSPb_Compt_CP(compound: str, E: float, theta: float, phi: float) -> float:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    theta : float
        _description_
    phi : float
        _description_

    Returns
    -------
    float
        _description_
    """
    e, m, n = _parser(compound)
    result = 0.0
    for i in nb.prange(n):
        result += _DCSPb_Compt(e[i], E, theta, phi, 0) * m[i]
    return result
