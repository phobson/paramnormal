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
    """ Decorator to let you use greek characters for fxn kwargs."""
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
    """
    Process the arguments for uniform distributions.

    Parameters
    ----------
    low : float, optional
        The lower end of the distribution.
    high : float, optional
        The higher end of the distribution.
    width : float, optional
        The width of the distribution. If only one of ``low`` or
        ``high`` is provided, this will be used to compute the missing
        parameter.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    Raises
    ------
    ValueError
        Raised when both ``high`` and ``width`` are not provided.

    """

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
    """
    Process arguments for normal distributions

    Parameters
    ----------
    mu : float, optional
        The mean (expected value) of the distribution.
    sigma : float, optional
        The standard deviation of the distribution.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {loc_key: mu, scale_key: sigma}


@greco_deco
def lognormal(mu=None, sigma=None, offset=0, fit=False):
    """
    Process arguments for normal distributions.

    Parameters
    ----------
    mu : float, optional
        The mean (expected value) of the underlying normal
        distribution.
    sigma : float, optional
        The standard deviation of the underlying normal distribution.
    offset : float, optional
        An aritmetic factor added to the entire distribution. Highly
        recommnded that this value remains zero.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    if offset is None:
        raise ValueError("`offset` parameter is required. Recommended value is 0.")
    return {'s': sigma, scale_key: numpy.exp(mu) if mu is not None else mu, loc_key: offset}


@greco_deco
def beta(alpha=None, beta=None, loc=0, scale=1, fit=False):
    """
    Process arguments for beta distributions.

    Parameters
    ----------
    alpha, beta : float, optional
        The shape parameters of the distribution.
    loc, scale : float, optional
        The location and scale parameters of the distribution.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'a': alpha, 'b': beta, loc_key: loc, scale_key: scale}


@greco_deco
def chi_squared(k=None, loc=0, scale=1, fit=False):
    """
    Process arguments for the chi_squared distribution.

    Parameters
    ----------
    k : float, optional
        The shape parameter of the distribution.
    loc, scale : int, optional
        The location and scale parameters of the distribution.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'df': k, loc_key: loc, scale_key: 1}


@greco_deco
def pareto(alpha=None, fit=False):
    """
    Process arguments for the pareto distribution.

    Parameters
    ----------
    alpha : float, optional
        The shape parameter of the distribution.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'b': alpha}


@greco_deco
def gamma(k=None, theta=None, fit=False):
    """
    Process arguments for the gamma distribution.

    Parameters
    ----------
    k, theta : float, optional
        The shape parameter of the distribution.
    fit : bool, optional
        Whether or not we're processing the arguments to fit the
        distribution.

    Returns
    -------
    params : dict
        The processed parameters that can be fed directly to
        ``scipy.stats`` functions and classes.

    """

    loc_key, scale_key = _get_loc_scale_keys(fit=fit)
    return {'a': k, scale_key: theta}
