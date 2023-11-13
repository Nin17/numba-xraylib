"""_summary_
"""

import functools
import inspect
import random
import typing
import warnings

import numba as nb
import numpy as np
import pytest
import xraylib_np
from numpy.testing import assert_allclose

import xraylib
import xraylib_numba
from xraylib_numba import XraylibNumbaWarning, config

config.CHECK_RESULT = True


rng = np.random.default_rng()
N = 10


# def test_function_coverage_xrl():
#     """_summary_"""
#     xrl_funcs = {
#         i for i in dir(xrl) if callable(getattr(xrl, i)) and not i.startswith("_")
#     }
#     xrl_numba_funcs = {
#         i
#         for i in dir(xrl_numba)
#         if callable(getattr(xrl_numba, i)) and not i.startswith("_")
#     }
#     missing_funcs = sorted(xrl_funcs - xrl_numba_funcs)
#     try:
#         assert len(missing_funcs) == 0
#     except AssertionError as error:
#         print(missing_funcs)
#         raise error

# def test_function_coverage_xrlnp():
#     """_summary_"""
#     xrl_funcs = {
#         i for i in dir(xrl_np) if callable(getattr(xrl_np, i)) and not i.startswith("_")
#     }
#     xrl_numba_funcs = {
#         i
#         for i in dir(xrl_np_numba)
#         if callable(getattr(xrl_np_numba, i)) and not i.startswith("_")
#     }
#     missing_funcs = sorted(xrl_funcs - xrl_numba_funcs)
#     try:
#         assert len(missing_funcs) == 0
#     except AssertionError as error:
#         print(missing_funcs)
#         raise error


class BaseTest:
    """_summary_"""

    @functools.cached_property
    def func(self) -> str:
        """_summary_

        Returns
        -------
        str
            _description_
        """
        return self.__class__.__name__.removeprefix("Test")

    @functools.cached_property
    def xrl_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        return getattr(xraylib, self.func)

    @functools.cached_property
    def xrl_numba_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        _func = getattr(xraylib, self.func)

        def func(*args):
            return _func(*args)

        return nb.njit(func)

    @functools.cached_property
    def xrl_sig(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return inspect.signature(self.xrl_func)

    # @functools.cached_property
    # def xrl_numba_sig(self):
    #     """_summary_

    #     Returns
    #     -------
    #     _type_
    #         _description_
    #     """
    #     return inspect.signature(self.xrl_numba_func)

    # def test_signature(self):
    #     """_summary_

    #     Raises
    #     ------
    #     NotImplementedError
    #         _description_
    #     """
    #     raise NotImplementedError

    # def test_isclose(self):
    #     """_summary_"""
    #     raise NotImplementedError


class XraylibTest(BaseTest):
    """_summary_

    Parameters
    ----------
    BaseTest : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    error
        _description_
    """

    args_dict = {
        "Z": (random.randint, (0, 119)),
        "E": (lambda: random.random() * 1000, ()),
        "theta": (lambda: (random.random() - 0.5) * 2 * np.pi, ()),
        "shell": (random.randint, (1, 30)),  # TODO sensible values
        "auger_trans": (random.randint, (1, 30)),  # TODO sensible values
        "trans": (random.randint, (1, 30)),  # TODO sensible values
        "line": (random.randint, (1, 30)),  # TODO sensible values
        "E0": (lambda: random.random() * 1000, ()),  # TODO sensible values
        "phi": (lambda: (random.random() - 0.5) * 2 * np.pi, ()),
        "pz": (random.random, ()),  # TODO sensible values
        "q": (random.random, ()),  # TODO sensible values
        "PK": (random.random, ()),  # TODO sensible values
        "PL1": (random.random, ()),  # TODO sensible values
        "PL2": (random.random, ()),  # TODO sensible values
        "PL3": (random.random, ()),  # TODO sensible values
        "PM1": (random.random, ()),  # TODO sensible values
        "PM2": (random.random, ()),  # TODO sensible values
        "PM3": (random.random, ()),  # TODO sensible values
        "PM4": (random.random, ()),  # TODO sensible values
    }

    @functools.cached_property
    def args(self) -> tuple[typing.Any]:
        """_summary_

        Returns
        -------
        tuple[typing.Any]
            _description_
        """
        return tuple(
            [
                self.args_dict[arg][0](*self.args_dict[arg][1])
                for arg in self.xrl_sig.parameters.keys()
            ]
        )

    def test_isclose(self):
        try:
            xrl_result = self.xrl_func(*self.args)
        except ValueError:
            xrl_result = 0.0
            warnings.filterwarnings("error", category=XraylibNumbaWarning)
            with pytest.raises(Exception): # TODO: change to ValueError
                self.xrl_numba_func(*self.args)

        # warnings.filterwarnings("ignore", category=XraylibNumbaWarning)
        # assert self.xrl_numba_func(*self.args) == xrl_result

    def test_bare_compile(self):
        warnings.filterwarnings("ignore", category=XraylibNumbaWarning)
        _func = getattr(xraylib, self.func)
        try:
            self.xrl_func(*self.args)
        except ValueError:
            # TODO remove this
            warnings.filterwarnings("error", category=XraylibNumbaWarning)
            with pytest.raises(Exception):
                nb.njit(_func)(*self.args)
        else:
            nb.njit(_func)(*self.args)

    # def test_signature(self):
    #     """_summary_"""
    #     xrl_sig = inspect.signature(self.xrl_func).parameters.keys()
    #     xrl_numba_sig = inspect.signature(self.xrl_numba_func).parameters.keys()
    #     assert xrl_sig == xrl_numba_sig


class XraylibNpTest(BaseTest):
    """_summary_

    Parameters
    ----------
    BaseTest : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    error
        _description_
    """

    args_np_dict = {
        "Z": (lambda *args: rng.integers(*args).astype(np.int_), (0, 119, N)),
        "E": (lambda: rng.random(N) * 1000, ()),
        "theta": (lambda: (rng.random(N) - 0.5) * 2 * np.pi, ()),
        "shell": (
            lambda *args: rng.integers(*args).astype(np.int_),
            (1, 30, N),
        ),  # TODO sensible values
        "auger_trans": (
            lambda *args: rng.integers(*args).astype(np.int_),
            (1, 30, N),
        ),  # TODO sensible values
        "trans": (
            lambda *args: rng.integers(*args).astype(np.int_),
            (1, 30, N),
        ),  # TODO sensible values
        "line": (
            lambda *args: rng.integers(*args).astype(np.int_),
            (1, 30, N),
        ),  # TODO sensible values
        "E0": (lambda: rng.random(N) * 1000, ()),  # TODO sensible values
        "phi": (lambda: (rng.random(N) - 0.5) * 2 * np.pi, ()),
        "pz": (lambda: rng.random(N), ()),  # TODO sensible values
        "q": (lambda: rng.random(N), ()),  # TODO sensible values
    }

    @functools.cached_property
    def args_np(self) -> tuple[typing.Any]:
        """_summary_

        Returns
        -------
        tuple[typing.Any]
            _description_
        """
        return tuple(
            self.args_np_dict[arg][0](*self.args_np_dict[arg][1])
            for arg in self.xrl_sig.parameters.keys()
        )

    @functools.cached_property
    def xrl_np_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        return getattr(xraylib_np, self.func)

    @functools.cached_property
    def xrl_np_numba_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """

        _xrlnp_func = getattr(xraylib_np, self.func)

        def func(*args):
            return _xrlnp_func(*args)

        return nb.njit(func)

    @functools.cached_property
    def xrl_np_result(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self.xrl_np_func(*self.args_np)

    @functools.cached_property
    def xrl_np_numba_result(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        return self.xrl_np_numba_func(*self.args_np)

    def test_dtype(self):
        """_summary_"""
        assert self.xrl_np_result.dtype == self.xrl_np_numba_result.dtype

    def test_allclose(self):
        """_summary_

        Raises
        ------
        error
            _description_
        """
        try:
            assert_allclose(self.xrl_np_result, self.xrl_np_numba_result)
        except AssertionError as error:
            # print(self.xrl_np_result, self.xrl_np_numba_result)
            raise error

    @pytest.mark.skipif(
        nb.__version__ <= "0.58.0",
        reason="doesn't support directly jitting cython function",
    )
    def test_bare_compile_np(self) -> None:
        """
        Apply numba.njit to the cython function directly, without wrapping it in a
        python function.
        """
        _func = getattr(xraylib_np, self.func)
        nb.njit(_func)(*self.args_np)

    # def test_signature_np(self):
    #     """_summary_"""
    #     xrl_sig = inspect.signature(self.xrl_func).parameters.keys()
    #     xrl_numba_sig = inspect.signature(self.xrl_np_numba_func).parameters.keys()
    #     assert xrl_sig == xrl_numba_sig
