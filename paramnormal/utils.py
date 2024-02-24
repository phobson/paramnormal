from functools import wraps
from inspect import signature

import numpy

SYMBOLS = {
    "μ": "mu",
    "σ": "sigma",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "λ": "lamda",
    "θ": "theta",
}


def greco_deco(func):
    """Decorator to let you use greek characters for fxn kwargs."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        kwargs = {SYMBOLS.get(k, k): v for k, v in kwargs.items()}
        bound = sig.bind(*args, **kwargs)
        return func(**bound.arguments)

    return wrapper


def seed(func):
    """Decorator to seed the RNG before any function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)

    return wrapper


def _get_loc_scale_keys(fit=False):
    if fit:
        return "floc", "fscale"
    else:
        return "loc", "scale"


def _remove_nones(**kwargs):
    """
    Removes any kwargs whose values are `None`.
    """

    final = kwargs.copy()
    for k in kwargs:
        if kwargs[k] is None:
            final.pop(k)
    return final
