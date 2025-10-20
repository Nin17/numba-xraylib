"""Tests for numba-xraylib against xraylib and xraylib_np."""

from __future__ import annotations

import functools
import inspect
import random
from typing import TYPE_CHECKING

import numba as nb
import pytest
import xraylib
import xraylib_np
from numpy import asarray, broadcast_to, float64, int_, pi
from numpy.random import default_rng
from numpy.testing import assert_equal

from numba_xraylib import config

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

N = 10


class BaseTest:
    """Base class to obtain the xraylib functions from the test classes."""

    @functools.cached_property
    def func(self: BaseTest) -> str:
        """Xraylib function name.

        Returns
        -------
        str
            The name of the xraylib function.

        """
        return self.__class__.__name__.removeprefix("Test")

    @functools.cached_property
    def xrl_func(self: BaseTest) -> Callable:
        """Xraylib function.

        Returns
        -------
        callable
            The xraylib function.

        """
        return getattr(xraylib, self.func)

    @functools.cached_property
    def xrl_numba_func(self: BaseTest) -> Callable:
        """Xraylib function wrapped in numba.njit.

        Returns
        -------
        callable
            The xraylib function wrapped in numba.njit.

        """
        _func = getattr(xraylib, self.func)

        def func(*args: float) -> float:
            return _func(*args)

        return nb.njit(func)

    @functools.cached_property
    def xrl_sig(self: BaseTest) -> inspect.Signature:
        """Xraylib function signature.

        Returns
        -------
        _type_
            _description_

        """
        return inspect.signature(self.xrl_func)


class XraylibTest(BaseTest):
    """Base class for testing xraylib functions."""

    @staticmethod
    def _args(arg: str) -> int | float | str:
        i_args = {
            "Z": (0, 119),
            "shell": (1, 30),
            "auger_trans": (1, 30),
            "trans": (1, 30),
            "line": (1, 30),
        }
        f_args = {
            "E": (0.0, 1000.0),
            "theta": (0.5, 2 * pi),
            "E0": (0.0, 1000.0),
            "phi": (0.5, 2 * pi),
            "pz": (0.0, 1.0),
            "q": (0.0, 1.0),
            "PK": (0.0, 1.0),
            "PL1": (0.0, 1.0),
            "PL2": (0.0, 1.0),
            "PL3": (0.0, 1.0),
            "PM1": (0.0, 1.0),
            "PM2": (0.0, 1.0),
            "PM3": (0.0, 1.0),
            "PM4": (0.0, 1.0),
            "density": (0.0, 10000.0),
        }
        s_args = {
            "symbol": "H",
            "compound": "H2S(O4)",
        }
        if arg in i_args:
            return random.randint(*i_args[arg])
        if arg in f_args:
            return (random.random() - f_args[arg][0]) * f_args[arg][1]
        if arg in s_args:
            return s_args[arg]

        msg = f"Invalid argument: {arg}"
        raise KeyError(msg)

    @functools.cached_property
    def args(self: XraylibTest) -> tuple[int | float | str, ...]:
        """Arguments for the xraylib function.

        Returns
        -------
        tuple[int | float, ...]
            The arguments for the xraylib function.

        """
        return tuple([self._args(arg) for arg in self.xrl_sig.parameters])

    def test_xrl(self: XraylibTest) -> None:
        """Test njit function against xraylib return and possible error."""
        xrl_numba_result = None
        xrl_result = None
        try:
            xrl_result = self.xrl_func(*self.args)
            xrl_numba_result = self.xrl_numba_func(*self.args)
        except ValueError:
            with pytest.raises(ValueError):  # noqa: PT011
                self.xrl_numba_func(*self.args)

        assert_equal(xrl_result, xrl_numba_result)

    def test_bare_compile(self: XraylibTest) -> None:
        """Test function with numba.njit without wrapper function."""
        _func = getattr(xraylib, self.func)
        try:
            self.xrl_func(*self.args)
        except ValueError:
            with pytest.raises(ValueError):  # noqa: PT011
                nb.njit(_func)(*self.args)
        else:
            nb.njit(_func)(*self.args)


class XraylibNpTest(BaseTest):
    """Base class for testing xraylib_np functions."""

    @staticmethod
    def _args_np(arg: str) -> NDArray[int_] | NDArray[float64]:
        rng = default_rng(seed=random.randint(0, 2**32 - 1))
        i_args = {
            "Z": (0, 119, N),
            "shell": (1, 30, N + 3),
            "auger_trans": (1, 30, N + 4),
            "trans": (1, 30, N + 5),
            "line": (1, 30, N + 6),
        }
        f_args = {
            "E": (N + 1, 0.0, 1000.0),
            "theta": (N + 2, 0.5, 2 * pi),
            "E0": (N + 7, 0.0, 1000.0),
            "phi": (N + 8, 0.5, 2 * pi),
            "pz": (N + 9, 0.0, 1.0),
            "q": (N + 10, 0.0, 1.0),
        }
        if arg in i_args:
            return rng.integers(*i_args[arg]).astype(int_)
        if arg in f_args:
            return (rng.random(f_args[arg][0]) - f_args[arg][1]) * f_args[arg][2]

        msg = f"Invalid argument: {arg}"
        raise ValueError(msg)

    @functools.cached_property
    def args_np(
        self: XraylibNpTest,
    ) -> tuple[NDArray[int_] | NDArray[float64], ...]:
        """Arguments for the xraylib_np function.

        Returns
        -------
        tuple[NDArray[int_] | NDArray[float64], ...]
            The arguments for the xraylib_np function.

        """
        return tuple(self._args_np(arg) for arg in self.xrl_sig.parameters)

    @functools.cached_property
    def xrl_np_func(self: XraylibNpTest) -> Callable:
        """Xraylib_np function.

        Returns
        -------
        callable
            The xraylib_np function.

        """
        return getattr(xraylib_np, self.func)

    @functools.cached_property
    def xrl_np_numba_func(self: XraylibNpTest) -> Callable:
        """Xraylib_np function wrapped in numba.njit.

        Returns
        -------
        callable
            The xraylib_np function wrapped in numba.njit.

        """
        _xrlnp_func = getattr(xraylib_np, self.func)

        def func(*args: NDArray[int_] | NDArray[float64]) -> NDArray[float64]:
            return _xrlnp_func(*args)

        return nb.njit(func)

    @functools.cached_property
    def xrl_np_result(self: XraylibNpTest) -> NDArray[float64]:
        """Xraylib_np result."""
        return self.xrl_np_func(*self.args_np)

    @functools.cached_property
    def xrl_np_numba_result(self: XraylibNpTest) -> NDArray[float64]:
        """Xraylib_np result wrapped in numba.njit result."""
        return self.xrl_np_numba_func(*self.args_np)

    def test_dtype(self: XraylibNpTest) -> None:
        """Test dtypes of njit function and xraylib_np match."""
        assert self.xrl_np_result.dtype == self.xrl_np_numba_result.dtype  # noqa: S101

    def test_xrl_np(self: XraylibNpTest) -> None:
        """Test njit function against xraylib_np return."""
        assert_equal(self.xrl_np_result, self.xrl_np_numba_result)

    @pytest.mark.skip(reason="doesn't support directly jitting cython function")
    def test_bare_compile_np(self: XraylibNpTest) -> None:
        """Apply numba.njit to the cython function directly."""
        _func = getattr(xraylib_np, self.func)
        nb.njit(_func)(*self.args_np)

    def test_nd(self: XraylibNpTest) -> None:
        """Test with N-dimensional arrays."""
        rng = default_rng(seed=random.randint(0, 2**32 - 1))

        config.allow_nd = True
        _func_np = getattr(xraylib_np, self.func)

        xrl_result = _func_np(*[i[:1] for i in self.args_np])

        ndims = [random.randint(1, 3) for _ in self.args_np]
        shapes = [tuple([rng.integers(1, 4) for _ in range(ndim)]) for ndim in ndims]

        args_np = [
            broadcast_to(asarray(i[:1]).reshape(*[1] * ndim), shape)
            for i, ndim, shape in zip(self.args_np, ndims, shapes, strict=False)
        ]

        xrl_np_result = self.xrl_np_numba_func(*args_np)

        assert xrl_np_result.shape == sum(shapes, ())  # noqa: S101

        assert_equal(xrl_result.item(), xrl_np_result)

        config.allow_nd = False


class TestAtomicWeight(XraylibTest, XraylibNpTest): ...


class TestElementDensity(XraylibTest, XraylibNpTest): ...


class TestCS_KN(XraylibTest, XraylibNpTest): ...


class TestDCS_Thoms(XraylibTest, XraylibNpTest): ...


class TestAtomicLevelWidth(XraylibTest, XraylibNpTest): ...


class TestAugerRate(XraylibTest, XraylibNpTest): ...


class TestAugerYield(XraylibTest, XraylibNpTest): ...


class TestCosKronTransProb(XraylibTest, XraylibNpTest): ...


class TestEdgeEnergy(XraylibTest, XraylibNpTest): ...


class TestElectronConfig(XraylibTest, XraylibNpTest): ...


class TestFluorYield(XraylibTest, XraylibNpTest): ...


class TestJumpFactor(XraylibTest, XraylibNpTest): ...


class TestLineEnergy(XraylibTest, XraylibNpTest): ...


class TestRadRate(XraylibTest, XraylibNpTest): ...


class TestComptonEnergy(XraylibTest, XraylibNpTest): ...


class TestDCS_KN(XraylibTest, XraylibNpTest): ...


class TestDCSP_Thoms(XraylibTest, XraylibNpTest): ...


class TestMomentTransf(XraylibTest, XraylibNpTest): ...


class TestComptonProfile(XraylibTest, XraylibNpTest): ...


class TestCS_Compt(XraylibTest, XraylibNpTest): ...


class TestCS_Energy(XraylibTest, XraylibNpTest): ...


class TestCS_Photo(XraylibTest, XraylibNpTest): ...


class TestCS_Photo_Total(XraylibTest, XraylibNpTest): ...


class TestCS_Rayl(XraylibTest, XraylibNpTest): ...


class TestCS_Total(XraylibTest, XraylibNpTest): ...


class TestCS_Total_Kissel(XraylibTest, XraylibNpTest): ...


class TestCSb_Compt(XraylibTest, XraylibNpTest): ...


class TestCSb_Photo(XraylibTest, XraylibNpTest): ...


class TestCSb_Photo_Total(XraylibTest, XraylibNpTest): ...


class TestCSb_Rayl(XraylibTest, XraylibNpTest): ...


class TestCSb_Total(XraylibTest, XraylibNpTest): ...


class TestCSb_Total_Kissel(XraylibTest, XraylibNpTest): ...


class TestFF_Rayl(XraylibTest, XraylibNpTest): ...


class TestSF_Compt(XraylibTest, XraylibNpTest): ...


class TestFi(XraylibTest, XraylibNpTest): ...


class TestFii(XraylibTest, XraylibNpTest): ...


class TestPL1_pure_kissel(XraylibTest): ...


class TestPM1_pure_kissel(XraylibTest): ...


class TestComptonProfile_Partial(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine_Kissel(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine_Kissel(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine_Kissel_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine_Kissel_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine_Kissel_no_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine_Kissel_no_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine_Kissel_Nonradiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine_Kissel_Nonradiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine_Kissel_Radiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine_Kissel_Radiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell_Kissel(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell_Kissel(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell_Kissel_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell_Kissel_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell_Kissel_no_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell_Kissel_no_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell_Kissel_Nonradiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell_Kissel_Nonradiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell_Kissel_Radiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell_Kissel_Radiative_Cascade(XraylibTest, XraylibNpTest): ...


class TestCS_FluorLine(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorLine(XraylibTest, XraylibNpTest): ...


class TestCS_FluorShell(XraylibTest, XraylibNpTest): ...


class TestCSb_FluorShell(XraylibTest, XraylibNpTest): ...


class TestCS_Photo_Partial(XraylibTest, XraylibNpTest): ...


class TestCSb_Photo_Partial(XraylibTest, XraylibNpTest): ...


class TestDCS_Compt(XraylibTest, XraylibNpTest): ...


class TestDCS_Rayl(XraylibTest, XraylibNpTest): ...


class TestDCSb_Compt(XraylibTest, XraylibNpTest): ...


class TestDCSb_Rayl(XraylibTest, XraylibNpTest): ...


class TestPL1_auger_cascade_kissel(XraylibTest): ...


class PL1_full_cascade_kissel(XraylibTest): ...


class PL1_rad_cascade_kissel(XraylibTest): ...


class TestPL2_pure_kissel(XraylibTest): ...


class TestPM2_pure_kissel(XraylibTest): ...


class TestDCSP_Rayl(XraylibTest, XraylibNpTest): ...


class TestDCSP_Compt(XraylibTest, XraylibNpTest): ...


class TestDCSPb_Rayl(XraylibTest, XraylibNpTest): ...


class TestDCSPb_Compt(XraylibTest, XraylibNpTest): ...


class TestPL2_auger_cascade_kissel(XraylibTest): ...


class TestPL2_full_cascade_kissel(XraylibTest): ...


class TestPL2_rad_cascade_kissel(XraylibTest): ...


class TestPL3_pure_kissel(XraylibTest): ...


class TestPM3_pure_kissel(XraylibTest): ...


class TestPL3_auger_cascade_kissel(XraylibTest): ...


class TestPL3_full_cascade_kissel(XraylibTest): ...


class TestPL3_rad_cascade_kissel(XraylibTest): ...


class TestPM4_pure_kissel(XraylibTest): ...


class TestPM1_auger_cascade_kissel(XraylibTest): ...


class TestPM1_full_cascade_kissel(XraylibTest): ...


class TestPM1_rad_cascade_kissel(XraylibTest): ...


class TestPM5_pure_kissel(XraylibTest): ...


class TestPM2_auger_cascade_kissel(XraylibTest): ...


class TestPM2_full_cascade_kissel(XraylibTest): ...


class TestPM2_rad_cascade_kissel(XraylibTest): ...


class TestPM3_auger_cascade_kissel(XraylibTest): ...


class TestPM3_full_cascade_kissel(XraylibTest): ...


class TestPM3_rad_cascade_kissel(XraylibTest): ...


class TestPM4_auger_cascade_kissel(XraylibTest): ...


class TestPM4_full_cascade_kissel(XraylibTest): ...


class TestPM4_rad_cascade_kissel(XraylibTest): ...


class TestPM5_auger_cascade_kissel(XraylibTest): ...


class TestPM5_full_cascade_kissel(XraylibTest): ...


class TestPM5_rad_cascade_kissel(XraylibTest): ...


class TestDCSP_KN(XraylibTest, XraylibNpTest): ...


class TestCS_Total_CP(XraylibTest): ...


class TestCS_Photo_CP(XraylibTest): ...


class TestCS_Rayl_CP(XraylibTest): ...


class TestCS_Compt_CP(XraylibTest): ...


class TestCS_Energy_CP(XraylibTest): ...


class TestCS_Photo_Total_CP(XraylibTest): ...


class TestCS_Total_Kissel_CP(XraylibTest): ...


class TestCSb_Total_CP(XraylibTest): ...


class TestCSb_Photo_CP(XraylibTest): ...


class TestCSb_Rayl_CP(XraylibTest): ...


class TestCSb_Compt_CP(XraylibTest): ...


class TestCSb_Photo_Total_CP(XraylibTest): ...


class TestCSb_Total_Kissel_CP(XraylibTest): ...


class TestDCS_Compt_CP(XraylibTest): ...


class TestDCS_Rayl_CP(XraylibTest): ...


class TestDCSb_Compt_CP(XraylibTest): ...


class TestDCSb_Rayl_CP(XraylibTest): ...


class TestRefractive_Index_Re(XraylibTest): ...


class TestRefractive_Index_Im(XraylibTest): ...


@pytest.mark.xfail(reason="xrlComplex not implemented")
class TestRefractive_Index(XraylibTest): ...


class TestDCSP_Rayl_CP(XraylibTest): ...


class TestDCSP_Compt_CP(XraylibTest): ...


class TestDCSPb_Rayl_CP(XraylibTest): ...


class TestDCSPb_Compt_CP(XraylibTest): ...
