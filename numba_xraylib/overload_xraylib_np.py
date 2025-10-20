"""Overloads for xraylib functions to work with numba."""
# TODO(nin17): check variable names are the same as in the xraylib functions

from __future__ import annotations

from numba import vectorize
from numpy import broadcast_to

from .utils import (
    check_ndim,
    check_types_xrl_np,
    external_fcn,
    indices,
    overload_xrl_np,
)

# --------------------------------------- 1 int -------------------------------------- #


@overload_xrl_np
def _AtomicWeight_np(Z):
    check_types_xrl_np((Z,), ni=1)
    check_ndim(Z)
    xrl_fcn = external_fcn(ni=1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


@overload_xrl_np
def _ElementDensity_np(Z):
    check_types_xrl_np((Z,), ni=1)
    check_ndim(Z)
    xrl_fcn = external_fcn(ni=1)

    @vectorize
    def _impl(Z):
        return xrl_fcn(Z, 0)

    return lambda Z: _impl(Z)


# ------------------------------------- 1 double ------------------------------------- #


@overload_xrl_np
def _CS_KN_np(E):
    check_types_xrl_np((E,), nf=1)
    check_ndim(E)
    xrl_fcn = external_fcn(nf=1)

    @vectorize
    def _impl(E):
        return xrl_fcn(E, 0)

    return lambda E: _impl(E)


@overload_xrl_np
def _DCS_Thoms_np(theta):
    check_types_xrl_np((theta,), nf=1)
    check_ndim(theta)
    xrl_fcn = external_fcn(nf=1)

    @vectorize
    def _impl(theta):
        return xrl_fcn(theta, 0)

    return lambda theta: _impl(theta)


# --------------------------------------- 2 int -------------------------------------- #


@overload_xrl_np
def _AtomicLevelWidth_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)
        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _AugerRate_np(Z, auger_trans):
    check_types_xrl_np((Z, auger_trans), ni=2)
    check_ndim(Z, auger_trans)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, auger_trans)

    @vectorize
    def _impl(Z, auger_trans):
        return xrl_fcn(Z, auger_trans, 0)

    def impl(Z, auger_trans):
        shape = Z.shape + auger_trans.shape

        _Z = broadcast_to(Z[i0], shape)
        _auger_trans = broadcast_to(auger_trans[i1], shape)

        return _impl(_Z, _auger_trans)

    return impl


@overload_xrl_np
def _AugerYield_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _CosKronTransProb_np(Z, trans):
    check_types_xrl_np((Z, trans), ni=2)
    check_ndim(Z, trans)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, trans)

    @vectorize
    def _impl(Z, trans):
        return xrl_fcn(Z, trans, 0)

    def impl(Z, trans):
        shape = Z.shape + trans.shape

        _Z = broadcast_to(Z[i0], shape)
        _trans = broadcast_to(trans[i1], shape)

        return _impl(_Z, _trans)

    return impl


@overload_xrl_np
def _EdgeEnergy_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _ElectronConfig_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _FluorYield_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _JumpFactor_np(Z, shell):
    check_types_xrl_np((Z, shell), ni=2)
    check_ndim(Z, shell)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, shell)

    @vectorize
    def _impl(Z, shell):
        return xrl_fcn(Z, shell, 0)

    def impl(Z, shell):
        shape = Z.shape + shell.shape

        _Z = broadcast_to(Z[i0], shape)
        _shell = broadcast_to(shell[i1], shape)

        return _impl(_Z, _shell)

    return impl


@overload_xrl_np
def _LineEnergy_np(Z, line):
    check_types_xrl_np((Z, line), ni=2)
    check_ndim(Z, line)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, line)

    @vectorize
    def _impl(Z, line):
        return xrl_fcn(Z, line, 0)

    def impl(Z, line):
        shape = Z.shape + line.shape

        _Z = broadcast_to(Z[i0], shape)
        _line = broadcast_to(line[i1], shape)

        return _impl(_Z, _line)

    return impl


@overload_xrl_np
def _RadRate_np(Z, line):
    check_types_xrl_np((Z, line), ni=2)
    check_ndim(Z, line)
    xrl_fcn = external_fcn(ni=2)
    i0, i1 = indices(Z, line)

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


@overload_xrl_np
def _ComptonEnergy_np(E0, theta):
    check_types_xrl_np((E0, theta), nf=2)
    check_ndim(E0, theta)
    xrl_fcn = external_fcn(nf=2)
    i0, i1 = indices(E0, theta)

    @vectorize
    def _impl(E0, theta):
        return xrl_fcn(E0, theta, 0)

    def impl(E0, theta):
        shape = E0.shape + theta.shape

        _E0 = broadcast_to(E0[i0], shape)
        _theta = broadcast_to(theta[i1], shape)

        return _impl(_E0, _theta)

    return impl


@overload_xrl_np
def _DCS_KN_np(E, theta):
    check_types_xrl_np((E, theta), nf=2)
    check_ndim(E, theta)
    xrl_fcn = external_fcn(nf=2)
    i0, i1 = indices(E, theta)

    @vectorize
    def _impl(E, theta):
        return xrl_fcn(E, theta, 0)

    def impl(E, theta):
        shape = E.shape + theta.shape

        _E = broadcast_to(E[i0], shape)
        _theta = broadcast_to(theta[i1], shape)

        return _impl(_E, _theta)

    return impl


@overload_xrl_np
def _DCSP_Thoms_np(theta, phi):
    check_types_xrl_np((theta, phi), nf=2)
    check_ndim(theta, phi)
    xrl_fcn = external_fcn(nf=2)
    i0, i1 = indices(theta, phi)

    @vectorize
    def _impl(theta, phi):
        return xrl_fcn(theta, phi, 0)

    def impl(theta, phi):
        shape = theta.shape + phi.shape

        _theta = broadcast_to(theta[i0], shape)
        _phi = broadcast_to(phi[i1], shape)

        return _impl(_theta, _phi)

    return impl


@overload_xrl_np
def _MomentTransf_np(E, theta):
    check_types_xrl_np((E, theta), nf=2)
    check_ndim(E, theta)
    xrl_fcn = external_fcn(nf=2)
    i0, i1 = indices(E, theta)

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


@overload_xrl_np
def _ComptonProfile_np(Z, p):
    check_types_xrl_np((Z, p), ni=1, nf=1)
    check_ndim(Z, p)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, p)

    @vectorize
    def _impl(Z, p):
        return xrl_fcn(Z, p, 0)

    def impl(Z, p):
        shape = Z.shape + p.shape

        _Z = broadcast_to(Z[i0], shape)
        _p = broadcast_to(p[i1], shape)

        return _impl(_Z, _p)

    return impl


@overload_xrl_np
def _CS_Compt_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Energy_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Photo_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Photo_Total_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Rayl_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Total_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CS_Total_Kissel_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Compt_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Photo_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Photo_Total_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Rayl_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Total_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _CSb_Total_Kissel_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _FF_Rayl_np(Z, q):
    check_types_xrl_np((Z, q), ni=1, nf=1)
    check_ndim(Z, q)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, q)

    @vectorize
    def _impl(Z, q):
        return xrl_fcn(Z, q, 0)

    def impl(Z, q):
        shape = Z.shape + q.shape

        _Z = broadcast_to(Z[i0], shape)
        _q = broadcast_to(q[i1], shape)

        return _impl(_Z, _q)

    return impl


@overload_xrl_np
def _SF_Compt_np(Z, q):
    check_types_xrl_np((Z, q), ni=1, nf=1)
    check_ndim(Z, q)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, q)

    @vectorize
    def _impl(Z, q):
        return xrl_fcn(Z, q, 0)

    def impl(Z, q):
        shape = Z.shape + q.shape

        _Z = broadcast_to(Z[i0], shape)
        _q = broadcast_to(q[i1], shape)

        return _impl(_Z, _q)

    return impl


@overload_xrl_np
def _Fi_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


@overload_xrl_np
def _Fii_np(Z, E):
    check_types_xrl_np((Z, E), ni=1, nf=1)
    check_ndim(Z, E)
    xrl_fcn = external_fcn(ni=1, nf=1)
    i0, i1 = indices(Z, E)

    @vectorize
    def _impl(Z, E):
        return xrl_fcn(Z, E, 0)

    def impl(Z, E):
        shape = Z.shape + E.shape

        _Z = broadcast_to(Z[i0], shape)
        _E = broadcast_to(E[i1], shape)

        return _impl(_Z, _E)

    return impl


# ---------------------------------- 2 int, 1 double --------------------------------- #


@overload_xrl_np
def _ComptonProfile_Partial_np(Z, shell, pz):
    check_types_xrl_np((Z, shell, pz), ni=2, nf=1)
    check_ndim(Z, shell, pz)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, pz)

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


@overload_xrl_np
def _CS_FluorLine_Kissel_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_Kissel_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorLine_Kissel_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_Kissel_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_Kissel_no_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_Kissel_Nonradiative_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_Kissel_Radiative_Cascade_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorShell_Kissel_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_Kissel_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_FluorShell_Kissel_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_Kissel_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_Kissel_no_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_Kissel_Nonradiative_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_Kissel_Radiative_Cascade_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_FluorLine_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CSb_FluorLine_np(Z, line, E):
    check_types_xrl_np((Z, line, E), ni=2, nf=1)
    check_ndim(Z, line, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, line, E)

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


@overload_xrl_np
def _CS_FluorShell_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_FluorShell_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CS_Photo_Partial_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _CSb_Photo_Partial_np(Z, shell, E):
    check_types_xrl_np((Z, shell, E), ni=2, nf=1)
    check_ndim(Z, shell, E)
    xrl_fcn = external_fcn(ni=2, nf=1)
    i0, i1, i2 = indices(Z, shell, E)

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


@overload_xrl_np
def _DCS_Compt_np(Z, E, theta):
    check_types_xrl_np((Z, E, theta), ni=1, nf=2)
    check_ndim(Z, E, theta)
    xrl_fcn = external_fcn(ni=1, nf=2)
    i0, i1, i2 = indices(Z, E, theta)

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


@overload_xrl_np
def _DCS_Rayl_np(Z, E, theta):
    check_types_xrl_np((Z, E, theta), ni=1, nf=2)
    check_ndim(Z, E, theta)
    xrl_fcn = external_fcn(ni=1, nf=2)
    i0, i1, i2 = indices(Z, E, theta)

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


@overload_xrl_np
def _DCSb_Compt_np(Z, E, theta):
    check_types_xrl_np((Z, E, theta), ni=1, nf=2)
    check_ndim(Z, E, theta)
    xrl_fcn = external_fcn(ni=1, nf=2)
    i0, i1, i2 = indices(Z, E, theta)

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


@overload_xrl_np
def _DCSb_Rayl_np(Z, E, theta):
    check_types_xrl_np((Z, E, theta), ni=1, nf=2)
    check_ndim(Z, E, theta)
    xrl_fcn = external_fcn(ni=1, nf=2)
    i0, i1, i2 = indices(Z, E, theta)

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


# ---------------------------------- 1 int, 3 double --------------------------------- #


@overload_xrl_np
def _DCSP_Rayl_np(Z, E, theta, phi):
    check_types_xrl_np((Z, E, theta, phi), ni=1, nf=3)
    check_ndim(Z, E, theta, phi)
    xrl_fcn = external_fcn(ni=1, nf=3)
    i0, i1, i2, i3 = indices(Z, E, theta, phi)

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


@overload_xrl_np
def _DCSP_Compt_np(Z, E, theta, phi):
    check_types_xrl_np((Z, E, theta, phi), ni=1, nf=3)
    check_ndim(Z, E, theta, phi)
    xrl_fcn = external_fcn(ni=1, nf=3)
    i0, i1, i2, i3 = indices(Z, E, theta, phi)

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


@overload_xrl_np
def _DCSPb_Rayl_np(Z, E, theta, phi):
    check_types_xrl_np((Z, E, theta, phi), ni=1, nf=3)
    check_ndim(Z, E, theta, phi)
    xrl_fcn = external_fcn(ni=1, nf=3)
    i0, i1, i2, i3 = indices(Z, E, theta, phi)

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


@overload_xrl_np
def _DCSPb_Compt_np(Z, E, theta, phi):
    check_types_xrl_np((Z, E, theta, phi), ni=1, nf=3)
    check_ndim(Z, E, theta, phi)
    xrl_fcn = external_fcn(ni=1, nf=3)
    i0, i1, i2, i3 = indices(Z, E, theta, phi)

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


# ------------------------------------- 3 double ------------------------------------- #


@overload_xrl_np
def _DCSP_KN_np(E, theta, phi):
    check_types_xrl_np((E, theta, phi), nf=3)
    check_ndim(E, theta, phi)
    xrl_fcn = external_fcn(nf=3)
    i0, i1, i2 = indices(E, theta, phi)

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
