# -*- coding: utf-8 -*-
import sys
from functools import wraps

import numpy

if sys.version_info.major == 2:  # pragma: no cover
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
        if PY2:  # pragma: no cover
            return func(*args, **kwargs)
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


def uniform(low=None, high=None, width=None, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    params = {}
    if low is not None:
        params[loc_key] = low

    if high is not None:
        if low is None and width is None:
            raise ValueError('`uniform` requires both low and high params.')
        else:
            params[scale_key] = high - low
    else:
        params[scale_key] = width

    return params


@greco_deco
def normal(mu=None, sigma=None, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {loc_key: mu, scale_key: sigma}


@greco_deco
def lognormal(mu=None, sigma=None, offset=0, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    if offset is None:
        raise ValueError("`offset` parameter is required. Recommended value is 0.")
    return {'s': sigma, scale_key: numpy.exp(mu) if mu is not None else mu, loc_key: offset}


@greco_deco
def beta(alpha=None, beta=None, loc=0, scale=1, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'a': alpha, 'b': beta, loc_key: loc, scale_key: scale}


@greco_deco
def chi_squared(k=None, loc=0, scale=1, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'df': k, loc_key: loc, scale_key: 1}


@greco_deco
def pareto(alpha=None, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'b': alpha}


@greco_deco
def gamma(k=None, theta=None, fit=False):
    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'a': k, scale_key: theta}
