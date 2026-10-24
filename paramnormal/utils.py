from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import ParamSpec, TypeVar

import numpy

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

SYMBOLS = {
    "μ": "mu",
    "σ": "sigma",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "λ": "lamda",
    "θ": "theta",
}


def greco_deco(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to let you use greek characters for fxn kwargs."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        sig = signature(func)
        renamed = {SYMBOLS.get(k, k): v for k, v in kwargs.items()}
        bound = sig.bind(*args, **renamed)
        return func(*bound.args, **bound.kwargs)

    return wrapper


def seed(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to seed the RNG before any function."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        numpy.random.seed(0)
        return func(*args, **kwargs)

    return wrapper


def _get_loc_scale_keys(fit: bool = False) -> tuple[str, str]:
    if fit:
        return "floc", "fscale"
    else:
        return "loc", "scale"


def _remove_nones(**kwargs: T) -> dict[str, T]:
    """
    Removes any kwargs whose values are `None`.
    """

    final = kwargs.copy()
    for k in kwargs:
        if kwargs[k] is None:
            final.pop(k)
    return final
