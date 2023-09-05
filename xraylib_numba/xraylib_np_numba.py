"""
A numba compatible version of xraylib_np.
"""

# pylint: disable=not-an-iterable, protected-access, invalid-name

import numba as nb
import numpy as np
from numpy.typing import NDArray

import xraylib_numba.xraylib_numba as _xrl_nb
from .config import config

# TODO: docstrings


@nb.njit(**config["xrl_np"].get("AtomicWeight", {}))
def AtomicWeight(Z: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Atomic weight of an element Z in g/mol.

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number

    Returns
    -------
    NDArray[np.float64]
        Atomic weight in g/mol
    """
    assert Z.ndim == 1
    output = np.empty(Z.size, dtype=np.float64)
    for i in nb.prange(Z.size):
        output[i] = _xrl_nb._AtomicWeight(Z[i], 0)
    return output


@nb.njit(**config["xrl_np"].get("ElementDensity", {}))
def ElementDensity(Z: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Density of an element Z at room temperature and pressure in g/cm³.

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number

    Returns
    -------
    NDArray[np.float64]
        Density in g/cm³
    """
    assert Z.ndim == 1
    output = np.empty(Z.size, dtype=np.float64)
    for i in nb.prange(Z.size):
        output[i] = _xrl_nb._ElementDensity(Z[i], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_KN", {}))
def CS_KN(E: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert E.ndim == 1
    output = np.empty(E.size, dtype=np.float64)
    for i in nb.prange(E.size):
        output[i] = _xrl_nb._CS_KN(E[i], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCS_Thoms", {}))
def DCS_Thoms(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert theta.ndim == 1
    output = np.empty(theta.size, dtype=np.float64)
    for i in nb.prange(theta.size):  # pylint:   disable=not-an-iterable
        output[i] = _xrl_nb._DCS_Thoms(theta[i], 0)
    return output


@nb.njit(**config["xrl_np"].get("AtomicLevelWidth", {}))
def AtomicLevelWidth(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._AtomicLevelWidth(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("AugerRate", {}))
def AugerRate(
    Z: NDArray[np.int64], auger_trans: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    auger_trans : NDArray[np.int64]
        Macro identifying initial ionized shell and two resulting ejected
        elctrons

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert auger_trans.ndim == 1
    output = np.empty((Z.size, auger_trans.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(auger_trans.size):
            output[i, j] = _xrl_nb._AugerRate(Z[i], auger_trans[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("AugerYield", {}))
def AugerYield(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._AugerYield(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CosKronTransProb", {}))
def CosKronTransProb(
    Z: NDArray[np.int64], trans: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    trans : NDArray[np.int64]
        Transition type macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert trans.ndim == 1
    output = np.empty((Z.size, trans.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(trans.size):
            output[i, j] = _xrl_nb._CosKronTransProb(Z[i], trans[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("EdgeEnergy", {}))
def EdgeEnergy(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._EdgeEnergy(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("ElectronConfig", {}))
def ElectronConfig(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._ElectronConfig(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("FluorYield", {}))
def FluorYield(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._FluorYield(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("JumpFactor", {}))
def JumpFactor(
    Z: NDArray[np.int64], shell: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    output = np.empty((Z.size, shell.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            output[i, j] = _xrl_nb._JumpFactor(Z[i], shell[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("LineEnergy", {}))
def LineEnergy(
    Z: NDArray[np.int64], line: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    output = np.empty((Z.size, line.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            output[i, j] = _xrl_nb._LineEnergy(Z[i], line[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("RadRate", {}))
def RadRate(
    Z: NDArray[np.int64], line: NDArray[np.int64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    output = np.empty((Z.size, line.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            output[i, j] = _xrl_nb._RadRate(Z[i], line[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("ComptonEnergy", {}))
def ComptonEnergy(
    E0: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    E0 : NDArray[np.float64]
        Photon energy before scattering in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert E0.ndim == 1
    assert theta.ndim == 1
    output = np.empty((E0.size, theta.size), dtype=np.float64)
    for i in nb.prange(E0.size):
        for j in nb.prange(theta.size):
            output[i, j] = _xrl_nb._ComptonEnergy(E0[i], theta[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCS_KN", {}))
def DCS_KN(
    E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((E.size, theta.size), dtype=np.float64)
    for i in nb.prange(E.size):
        for j in nb.prange(theta.size):
            output[i, j] = _xrl_nb._DCS_KN(E[i], theta[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCSP_Thoms", {}))
def DCSP_Thoms(
    theta: NDArray[np.float64], phi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(theta.size):
        for j in nb.prange(phi.size):
            output[i, j] = _xrl_nb._DCSP_Thoms(theta[i], phi[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("MomentTransf", {}))
def MomentTransf(
    E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((E.size, theta.size), dtype=np.float64)
    for i in nb.prange(E.size):
        for j in nb.prange(theta.size):
            output[i, j] = _xrl_nb._MomentTransf(E[i], theta[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("ComptonProfile", {}))
def ComptonProfile(
    Z: NDArray[np.int64], pz: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    pz : NDArray[np.float64]
        Momentum in atomic units

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert pz.ndim == 1
    output = np.empty((Z.size, pz.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(pz.size):
            output[i, j] = _xrl_nb._ComptonProfile(Z[i], pz[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Compt", {}))
def CS_Compt(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Compt(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Energy", {}))
def CS_Energy(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Energy(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Photo", {}))
def CS_Photo(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Photo(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Photo_Total", {}))
def CS_Photo_Total(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Photo_Total(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Rayl", {}))
def CS_Rayl(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Rayl(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Total", {}))
def CS_Total(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Total(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CS_Total_Kissel", {}))
def CS_Total_Kissel(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CS_Total_Kissel(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Compt", {}))
def CSb_Compt(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Compt(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Photo", {}))
def CSb_Photo(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Photo(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Photo_Total", {}))
def CSb_Photo_Total(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Photo_Total(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Rayl", {}))
def CSb_Rayl(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Rayl(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Total", {}))
def CSb_Total(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Total(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_Total_Kissel", {}))
def CSb_Total_Kissel(
    Z: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._CSb_Total_Kissel(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("FF_Rayl", {}))
def FF_Rayl(
    Z: NDArray[np.int64], q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    q : NDArray[np.float64]
        Momentum transfer in Å⁻¹

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert q.ndim == 1
    output = np.empty((Z.size, q.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(q.size):
            output[i, j] = _xrl_nb._FF_Rayl(Z[i], q[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("SF_Compt", {}))
def SF_Compt(
    Z: NDArray[np.int64], q: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    q : NDArray[np.float64]
        Momentum transfer in Å⁻¹

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert q.ndim == 1
    output = np.empty((Z.size, q.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(q.size):
            output[i, j] = _xrl_nb._SF_Compt(Z[i], q[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("Fi", {}))
def Fi(Z: NDArray[np.int64], E: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._Fi(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("Fii", {}))
def Fii(Z: NDArray[np.int64], E: NDArray[np.float64]) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            output[i, j] = _xrl_nb._Fii(Z[i], E[j], 0)
    return output


@nb.njit(**config["xrl_np"].get("ComptonProfile_Partial", {}))
def ComptonProfile_Partial(
    Z: NDArray[np.int64], shell: NDArray[np.int64], pz: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    pz : NDArray[np.float64]
        Momentum in atomic units

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert pz.ndim == 1
    output = np.empty((Z.size, shell.size, pz.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(pz.size):
                output[i, j, k] = _xrl_nb._ComptonProfile_Partial(
                    Z[i], shell[j], pz[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorLine_Kissel", {}))
def CS_FluorLine_Kissel(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorLine_Kissel(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorLine_Kissel", {}))
def CSb_FluorLine_Kissel(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorLine_Kissel(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorLine_Kissel_Cascade", {}))
def CS_FluorLine_Kissel_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorLine_Kissel_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorLine_Kissel_Cascade", {}))
def CSb_FluorLine_Kissel_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorLine_Kissel_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorLine_Kissel_no_Cascade", {}))
def CS_FluorLine_Kissel_no_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorLine_Kissel_no_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorLine_Kissel_no_Cascade", {}))
def CSb_FluorLine_Kissel_no_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorLine_Kissel_no_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(
    **config["xrl_np"].get("CS_FluorLine_Kissel_Nonradiative_Cascade", {})
)
def CS_FluorLine_Kissel_Nonradiative_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CS_FluorLine_Kissel_Nonradiative_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(
    **config["xrl_np"].get("CSb_FluorLine_Kissel_Nonradiative_Cascade", {})
)
def CSb_FluorLine_Kissel_Nonradiative_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CSb_FluorLine_Kissel_Nonradiative_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorLine_Kissel_Radiative_Cascade", {}))
def CS_FluorLine_Kissel_Radiative_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CS_FluorLine_Kissel_Radiative_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorLine_Kissel_Radiative_Cascade", {}))
def CSb_FluorLine_Kissel_Radiative_Cascade(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CSb_FluorLine_Kissel_Radiative_Cascade(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorShell_Kissel", {}))
def CS_FluorShell_Kissel(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorShell_Kissel(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorShell_Kissel", {}))
def CSb_FluorShell_Kissel(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorShell_Kissel(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorShell_Kissel_Cascade", {}))
def CS_FluorShell_Kissel_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorShell_Kissel_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorShell_Kissel_Cascade", {}))
def CSb_FluorShell_Kissel_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorShell_Kissel_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorShell_Kissel_no_Cascade", {}))
def CS_FluorShell_Kissel_no_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorShell_Kissel_no_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorShell_Kissel_no_Cascade", {}))
def CSb_FluorShell_Kissel_no_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorShell_Kissel_no_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(
    **config["xrl_np"].get("CS_FluorShell_Kissel_Nonradiative_Cascade", {})
)
def CS_FluorShell_Kissel_Nonradiative_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CS_FluorShell_Kissel_Nonradiative_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(
    **config["xrl_np"].get("CSb_FluorShell_Kissel_Nonradiative_Cascade", {})
)
def CSb_FluorShell_Kissel_Nonradiative_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CSb_FluorShell_Kissel_Nonradiative_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorShell_Kissel_Radiative_Cascade", {}))
def CS_FluorShell_Kissel_Radiative_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CS_FluorShell_Kissel_Radiative_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorShell_Kissel_Radiative_Cascade", {}))
def CSb_FluorShell_Kissel_Radiative_Cascade(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[
                    i, j, k
                ] = _xrl_nb._CSb_FluorShell_Kissel_Radiative_Cascade(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorLine", {}))
def CS_FluorLine(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorLine(Z[i], line[j], E[k], 0)
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorLine", {}))
def CSb_FluorLine(
    Z: NDArray[np.int64], line: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    line : NDArray[np.int64]
        Line macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert line.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, line.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(line.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorLine(
                    Z[i], line[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_FluorShell", {}))
def CS_FluorShell(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_FluorShell(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_FluorShell", {}))
def CSb_FluorShell(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_FluorShell(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CS_Photo_Partial", {}))
def CS_Photo_Partial(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CS_Photo_Partial(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("CSb_Photo_Partial", {}))
def CSb_Photo_Partial(
    Z: NDArray[np.int64], shell: NDArray[np.int64], E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    shell : NDArray[np.int64]
        Shell macro
    E : NDArray[np.float64]
        Energy in keV

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert shell.ndim == 1
    assert E.ndim == 1
    output = np.empty((Z.size, shell.size, E.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(shell.size):
            for k in nb.prange(E.size):
                output[i, j, k] = _xrl_nb._CSb_Photo_Partial(
                    Z[i], shell[j], E[k], 0
                )
    return output


@nb.njit(**config["xrl_np"].get("DCS_Compt", {}))
def DCS_Compt(
    Z: NDArray[np.int64], E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((Z.size, E.size, theta.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                output[i, j, k] = _xrl_nb._DCS_Compt(Z[i], E[j], theta[k], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCS_Rayl", {}))
def DCS_Rayl(
    Z: NDArray[np.int64], E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((Z.size, E.size, theta.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                output[i, j, k] = _xrl_nb._DCS_Rayl(Z[i], E[j], theta[k], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCSb_Compt", {}))
def DCSb_Compt(
    Z: NDArray[np.int64], E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((Z.size, E.size, theta.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                output[i, j, k] = _xrl_nb._DCSb_Compt(Z[i], E[j], theta[k], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCSb_Rayl", {}))
def DCSb_Rayl(
    Z: NDArray[np.int64], E: NDArray[np.float64], theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    output = np.empty((Z.size, E.size, theta.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                output[i, j, k] = _xrl_nb._DCSb_Rayl(Z[i], E[j], theta[k], 0)
    return output


@nb.njit(**config["xrl_np"].get("DCSP_Compt", {}))
def DCSP_Compt(
    Z: NDArray[np.int64],
    E: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((Z.size, E.size, theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                for m in nb.prange(phi.size):
                    output[i, j, k, m] = _xrl_nb._DCSP_Compt(
                        Z[i], E[j], theta[k], phi[m], 0
                    )
    return output


@nb.njit(**config["xrl_np"].get("DCSP_Rayl", {}))
def DCSP_Rayl(
    Z: NDArray[np.int64],
    E: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((Z.size, E.size, theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                for m in nb.prange(phi.size):
                    output[i, j, k, m] = _xrl_nb._DCSP_Rayl(
                        Z[i], E[j], theta[k], phi[m], 0
                    )
    return output


@nb.njit(**config["xrl_np"].get("DCSPb_Compt", {}))
def DCSPb_Compt(
    Z: NDArray[np.int64],
    E: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((Z.size, E.size, theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                for m in nb.prange(phi.size):
                    output[i, j, k, m] = _xrl_nb._DCSPb_Compt(
                        Z[i], E[j], theta[k], phi[m], 0
                    )
    return output


@nb.njit(**config["xrl_np"].get("DCSPb_Rayl", {}))
def DCSPb_Rayl(
    Z: NDArray[np.int64],
    E: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    Z : NDArray[np.int64]
        Atomic number
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert Z.ndim == 1
    assert E.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((Z.size, E.size, theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(Z.size):
        for j in nb.prange(E.size):
            for k in nb.prange(theta.size):
                for m in nb.prange(phi.size):
                    output[i, j, k, m] = _xrl_nb._DCSPb_Rayl(
                        Z[i], E[j], theta[k], phi[m], 0
                    )
    return output


@nb.njit(**config["xrl_np"].get("DCSP_KN", {}))
def DCSP_KN(
    E: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """_summary_

    Parameters
    ----------
    E : NDArray[np.float64]
        Energy in keV
    theta : NDArray[np.float64]
        Scattering polar angle in radians
    phi : NDArray[np.float64]
        Scattering azimuthal angle in radians

    Returns
    -------
    NDArray[np.float64]
        _description_
    """
    assert E.ndim == 1
    assert theta.ndim == 1
    assert phi.ndim == 1
    output = np.empty((E.size, theta.size, phi.size), dtype=np.float64)
    for i in nb.prange(E.size):
        for j in nb.prange(theta.size):
            for k in nb.prange(phi.size):
                output[i, j, k] = _xrl_nb._DCSP_KN(E[i], theta[j], phi[k], 0)
    return output


# TODO add tests for these functions


# ??? should I implement these functions even though they aren't in the
# original
#     /* 1 string, 1 double */
#     _XRL_FUNCTION(CS_Total_CP)
#     _XRL_FUNCTION(CS_Photo_CP)
#     _XRL_FUNCTION(CS_Rayl_CP)
#     _XRL_FUNCTION(CS_Compt_CP)
#     _XRL_FUNCTION(CS_Energy_CP)
#     _XRL_FUNCTION(CS_Photo_Total_CP)
#     _XRL_FUNCTION(CS_Total_Kissel_CP)
#     _XRL_FUNCTION(CSb_Total_CP)
#     _XRL_FUNCTION(CSb_Photo_CP)
#     _XRL_FUNCTION(CSb_Rayl_CP)
#     _XRL_FUNCTION(CSb_Compt_CP)
#     _XRL_FUNCTION(CSb_Photo_Total_CP)
#     _XRL_FUNCTION(CSb_Total_Kissel_CP)
#     /* 1 string, 2 double */
#     _XRL_FUNCTION(DCS_Rayl_CP)
#     _XRL_FUNCTION(DCS_Compt_CP)
#     _XRL_FUNCTION(DCSb_Rayl_CP)
#     _XRL_FUNCTION(DCSb_Compt_CP)
#     _XRL_FUNCTION(Refractive_Index_Re)
#     _XRL_FUNCTION(Refractive_Index_Im)
#      /* 1 string, 3 double */
#     _XRL_FUNCTION(DCSP_Rayl_CP)
#     _XRL_FUNCTION(DCSP_Compt_CP)
#     _XRL_FUNCTION(DCSPb_Rayl_CP)
#     _XRL_FUNCTION(DCSPb_Compt_CP)
# }
