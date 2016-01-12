from collections import namedtuple

import numpy
from scipy import stats

from . import process_args


__all__ = [
    'normal',
    'lognormal'
]


_docstring = """\

    Parameters
    ----------
    data : array-like
        Data to be fit to the distribution.
    guesses : keyword arguments, optional
        Initial guesses for the distribution parameters.

    Returns
    -------
    params : namedtuple
        Collection of parameters that define the distribution in a
        manner consistent with ``paramnormal``.

    See also
    --------
    paramnormal.dist.{}

"""


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


def normal(data, **guesses):
    """
    Fit a normal distribution to data.
    {}"""

    params = _fit('norm', data, pnormname='normal', **guesses)
    template = namedtuple('params', ['mu', 'sigma'])
    return template(*params)


def lognormal(data, **guesses):
    """
    Fit a lognormal distribution to data.
    {}"""

    params =  _fit('lognorm', data, pnormname='lognormal', **guesses)
    template = namedtuple('params', ['mu', 'sigma', 'offset'])
    return template(mu=numpy.log(params[2]), sigma=params[0], offset=params[1])


def beta(data, **guesses):
    """
    Fit a beta distribution to data.
    {}
    """
    params = _fit('beta', data, pnormname='beta', **guesses)
    template = namedtuple('params', ['alpha', 'beta', 'loc', 'scale'])
    return template(*params)


def weibull(data, **guess):
    """
    Fit a weibull distribution to data.
    {}
    """
    params  = _fit('weibull_min', data, pnormname='weibull', **guess)
    template = namedtuple('params', ['k', 'loc', 'scale'])
    return template(*params)


fitters = [
    normal,
    lognormal,
    beta,
    weibull,
]

for f in fitters:
    f.__doc__ = f.__doc__.format(_docstring.format(f.__name__))
