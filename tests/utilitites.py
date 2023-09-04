"""_summary_
"""

import inspect
import functools
import random
import typing

import numpy as np
import pytest
import xraylib as xrl
import xraylib_np as xrl_np

# import xraylib_numba.xraylib_numba as xrl_numba
import xraylib_numba as xrl_numba
import xraylib_numba.xraylib_np_numba as xrl_np_numba


rng = np.random.default_rng()
N = 10


def test_function_coverage_xrl():
    """_summary_"""
    xrl_funcs = {
        i
        for i in dir(xrl)
        if callable(getattr(xrl, i)) and not i.startswith("_")
    }
    xrl_numba_funcs = {
        i
        for i in dir(xrl_numba)
        if callable(getattr(xrl_numba, i)) and not i.startswith("_")
    }
    missing_funcs = sorted(xrl_funcs - xrl_numba_funcs)
    try:
        assert len(missing_funcs) == 0
    except AssertionError as error:
        print(missing_funcs)
        raise error

def test_function_coverage_xrlnp():
    """_summary_"""
    xrl_funcs = {
        i
        for i in dir(xrl_np)
        if callable(getattr(xrl_np, i)) and not i.startswith("_")
    }
    xrl_numba_funcs = {
        i
        for i in dir(xrl_np_numba)
        if callable(getattr(xrl_np_numba, i)) and not i.startswith("_")
    }
    missing_funcs = sorted(xrl_funcs - xrl_numba_funcs)
    try:
        assert len(missing_funcs) == 0
    except AssertionError as error:
        print(missing_funcs)
        raise error


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
        return getattr(xrl, self.func)

    @functools.cached_property
    def xrl_numba_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        return getattr(xrl_numba, self.func)

    @functools.cached_property
    def xrl_sig(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return inspect.signature(self.xrl_func)

    @functools.cached_property
    def xrl_numba_sig(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return inspect.signature(self.xrl_numba_func)

    def test_signature(self):
        """_summary_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def test_isclose(self):
        """_summary_"""
        raise NotImplementedError


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
        "pz": (lambda: random.random(), ()),  # TODO sensible values
        "q": (lambda: random.random(), ()),  # TODO sensible values
        "PK": (lambda: random.random(), ()),  # TODO sensible values
        "PL1": (lambda: random.random(), ()),  # TODO sensible values
        "PL2": (lambda: random.random(), ()),  # TODO sensible values
        "PL3": (lambda: random.random(), ()),  # TODO sensible values
        "PM1": (lambda: random.random(), ()),  # TODO sensible values
        "PM2": (lambda: random.random(), ()),  # TODO sensible values
        "PM3": (lambda: random.random(), ()),  # TODO sensible values
        "PM4": (lambda: random.random(), ()),  # TODO sensible values
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
        """_summary_

        Raises
        ------
        error
            _description_
        """
        e_xrl = None
        try:
            xrl_value = self.xrl_func(*self.args)
        except ValueError as error:
            e_xrl = error

        if e_xrl is not None:
            message = "Invalid " + " | ".join(
                self.xrl_numba_sig.parameters.keys()
            )
            with pytest.raises(ValueError) as pytest_error:
                self.xrl_numba_func(*self.args)
            try:
                assert message == pytest_error.value.args[0]
            except AssertionError as error:
                print(message, pytest_error.value.args[0])
                raise error

        else:
            assert np.isclose(xrl_value, self.xrl_numba_func(*self.args))

    def test_signature(self):
        """_summary_"""
        xrl_sig = inspect.signature(self.xrl_func).parameters.keys()
        xrl_numba_sig = inspect.signature(
            self.xrl_numba_func
        ).parameters.keys()
        assert xrl_sig == xrl_numba_sig


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
            [
                self.args_np_dict[arg][0](*self.args_np_dict[arg][1])
                for arg in self.xrl_sig.parameters.keys()
            ]
        )

    @functools.cached_property
    def xrl_np_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        return getattr(xrl_np, self.func)

    @functools.cached_property
    def xrl_np_numba_func(self) -> callable:
        """_summary_

        Returns
        -------
        callable
            _description_
        """
        return getattr(xrl_np_numba, self.func)

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
            assert np.allclose(self.xrl_np_result, self.xrl_np_numba_result)
        except AssertionError as error:
            print(self.xrl_np_result, self.xrl_np_numba_result)
            raise error

    def test_signature_np(self):
        """_summary_"""
        xrl_sig = inspect.signature(self.xrl_func).parameters.keys()
        xrl_numba_sig = inspect.signature(
            self.xrl_np_numba_func
        ).parameters.keys()
        assert xrl_sig == xrl_numba_sig
