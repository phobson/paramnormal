# -*- coding: utf-8 -*-
import sys
from functools import wraps
from inspect import signature

import numpy


SYMBOLS = {
    'μ': 'mu',
    'σ': 'sigma',
    'α': 'alpha',
    'β': 'beta',
    'γ': 'gamma',
    'θ': 'theta'
}


def greco_deco(func):
    """ Decorator to let you use greek characters for fxn kwargs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        kwargs = {SYMBOLS.get(k, k): v for k, v in kwargs.items()}
        bound = sig.bind(*args, **kwargs)
        return func(**bound.arguments)
    return wrapper


def _get_loc_scale_keys(fit=False):
    if fit:
        return 'floc', 'fscale'
    else:
        return 'loc', 'scale'


def _pop_none(**kwargs):
    """
    Removes any kwargs whose values are `None`.
    """

    final = kwargs.copy()
    for k in kwargs:
        if kwargs[k] is None:
            final.pop(k)
    return final


def _fit(scipyname, data, pnormname, **guesses):
    """
    Performs the distribution's MLE fit via scipy.stats
    """

    dist = getattr(stats, scipyname)
    processor = getattr(process_args, pnormname)
    args = _pop_none(**processor(fit=True, **guesses))
    return dist.fit(data, **args)
