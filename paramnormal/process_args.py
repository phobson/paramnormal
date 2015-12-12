# -*- coding: utf-8 -*-
import sys
from functools import wraps

import numpy as np

if sys.version_info.major == 2:
    PY2 = True
else:
    PY2 = False
    from inspect import signature


SYMBOLS = {
    'μ': 'mu',
    'σ': 'sigma',
    'α': 'alpha',
    'β': 'beta',
    'γ': 'gamma',
    'θ': 'theta'
}


def greco_deco(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PY2:
            return func(*args, **kwargs)
        sig = signature(func)
        kwargs = {SYMBOLS.get(k, k): v for k, v in kwargs.items()}
        bound = sig.bind(*args, **kwargs)
        return func(**bound.arguments)
    return wrapper

@greco_deco
def uniform(low=None, high=None):
    params = {}
    if low is not None:
        params['loc'] = low

    if high is not None:
        if low is None:
            raise ValueError('uniform requires both low and high params')
        else:
            params['scale'] = high - low


    return params

@greco_deco
def normal(mu=None, sigma=None):
    return {'loc': mu, 'scale': sigma}


@greco_deco
def lognormal(mu=None, sigma=None, offset=0):
    return {'s': sigma, 'scale': np.exp(mu) if mu is not None else mu, 'loc': offset}


@greco_deco
def beta(alpha=None, beta=None):
    return {'a': alpha, 'b': beta}


@greco_deco
def chi_squared(k=None):
    return {'df': k}


@greco_deco
def pareto(alpha=None):
    return {'b': alpha}


@greco_deco
def gamma(k=None, theta=None):
    return {'a': k, 'scale': theta}

