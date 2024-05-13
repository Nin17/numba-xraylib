"""Tests for xraylib_numba against xraylib and xraylib_np."""

from __future__ import annotations

import functools
import inspect
import random
from types import MappingProxyType
from typing import TYPE_CHECKING

import numba as nb
import pytest
import xraylib_np
from numpy import asarray, broadcast_to, float64, int_, pi
from numpy.random import default_rng
from numpy.testing import assert_equal

import xraylib

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ruff: noqa: D101, N801, S311


rng = default_rng()
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
    def xrl_func(self: BaseTest) -> callable:
        """Xraylib function.

        Returns
        -------
        callable
            The xraylib function.

        """
        return getattr(xraylib, self.func)

    @functools.cached_property
    def xrl_numba_func(self: BaseTest) -> callable:
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

    args_dict = MappingProxyType(
        {
            "Z": (random.randint, (0, 119)),
            "E": (lambda: random.random() * 1000, ()),
            "theta": (lambda: (random.random() - 0.5) * 2 * pi, ()),
            "shell": (random.randint, (1, 30)),  # TODO(nin17): sensible values
            "auger_trans": (random.randint, (1, 30)),  # TODO(nin17): sensible values
            "trans": (random.randint, (1, 30)),  # TODO(nin17): sensible values
            "line": (random.randint, (1, 30)),  # TODO(nin17): sensible values
            "E0": (lambda: random.random() * 1000, ()),  # TODO(nin17): sensible values
            "phi": (lambda: (random.random() - 0.5) * 2 * pi, ()),
            "pz": (random.random, ()),  # TODO(nin17): sensible values
            "q": (random.random, ()),  # TODO(nin17): sensible values
            "PK": (random.random, ()),  # TODO(nin17): sensible values
            "PL1": (random.random, ()),  # TODO(nin17): sensible values
            "PL2": (random.random, ()),  # TODO(nin17): sensible values
            "PL3": (random.random, ()),  # TODO(nin17): sensible values
            "PM1": (random.random, ()),  # TODO(nin17): sensible values
            "PM2": (random.random, ()),  # TODO(nin17): sensible values
            "PM3": (random.random, ()),  # TODO(nin17): sensible values
            "PM4": (random.random, ()),  # TODO(nin17): sensible values
        },
    )

    @functools.cached_property
    def args(self: XraylibTest) -> tuple[int | float, ...]:
        """Arguments for the xraylib function.

        Returns
        -------
        tuple[int | float, ...]
            The arguments for the xraylib function.

        """
        return tuple(
            [
                self.args_dict[arg][0](*self.args_dict[arg][1])
                for arg in self.xrl_sig.parameters
            ],
        )

    def test_xrl(self: XraylibTest) -> None:
        """Test njit function against xraylib return and possible error."""
        try:
            xrl_result = self.xrl_func(*self.args)
        except ValueError:
            xrl_result = 0.0
            with pytest.raises(ValueError):  # noqa: PT011
                self.xrl_numba_func(*self.args)
            return
        assert_equal(xrl_result, self.xrl_numba_func(*self.args))

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

    args_np_dict = MappingProxyType(
        {
            "Z": (lambda *args: rng.integers(*args).astype(int_), (0, 119, N)),
            "E": (lambda: rng.random(N + 1) * 1000, ()),
            "theta": (lambda: (rng.random(N + 2) - 0.5) * 2 * pi, ()),
            "shell": (
                lambda *args: rng.integers(*args).astype(int_),
                (1, 30, N + 3),
            ),  # TODO(nin17): sensible values
            "auger_trans": (
                lambda *args: rng.integers(*args).astype(int_),
                (1, 30, N + 4),
            ),  # TODO(nin17): sensible values
            "trans": (
                lambda *args: rng.integers(*args).astype(int_),
                (1, 30, N + 5),
            ),  # TODO(nin17): sensible values
            "line": (
                lambda *args: rng.integers(*args).astype(int_),
                (1, 30, N + 6),
            ),  # TODO(nin17): sensible values
            "E0": (
                lambda: rng.random(N + 7) * 1000,
                (),
            ),  # TODO(nin17): sensible values
            "phi": (lambda: (rng.random(N + 8) - 0.5) * 2 * pi, ()),
            "pz": (lambda: rng.random(N + 9), ()),  # TODO(nin17): sensible values
            "q": (lambda: rng.random(N + 10), ()),  # TODO(nin17): sensible values
        },
    )

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
        return tuple(
            self.args_np_dict[arg][0](*self.args_np_dict[arg][1])
            for arg in self.xrl_sig.parameters
        )

    @functools.cached_property
    def xrl_np_func(self: XraylibNpTest) -> callable:
        """Xraylib_np function.

        Returns
        -------
        callable
            The xraylib_np function.

        """
        return getattr(xraylib_np, self.func)

    @functools.cached_property
    def xrl_np_numba_func(self: XraylibNpTest) -> callable:
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

    # @pytest.mark.skipif(not config.allow_Nd, reason="N-dimensional arrays not allowed")
    def test_nd(self: XraylibNpTest) -> None:
        """Test with N-dimensional arrays."""
        from xraylib_numba import config

        config.allow_nd = True
        # TODO(nin17): set allow_nd to True
        _func_np = getattr(xraylib_np, self.func)

        xrl_result = _func_np(*[i[:1] for i in self.args_np])

        ndims = [random.randint(1, 3) for _ in self.args_np]
        shapes = [tuple([rng.integers(1, 4) for _ in range(ndim)]) for ndim in ndims]

        args_np = [
            broadcast_to(asarray(i[:1]).reshape(*[1] * ndim), shape)
            for i, ndim, shape in zip(self.args_np, ndims, shapes)
        ]

        xrl_np_result = self.xrl_np_numba_func(*args_np)

        assert xrl_np_result.shape == sum(shapes, ())  # noqa: S101

        assert_equal(xrl_result.item(), xrl_np_result)


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


# TODO(nin17): tests for functions with string arguments when implemented
