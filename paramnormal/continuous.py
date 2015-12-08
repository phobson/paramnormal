import numpy as np
from scipy import stats

from . import process_args


def uniform(low=0, high=1):
    """ Create a frozen uniform distribution

    Parameters
    ----------
    low, high : floats, optional
        Lower and upper limits of the distribution.

    Returns
    -------
    scipy.stats.uniform

    """

    return stats.uniform(**process_args.uniform(low=low, high=high))


def normal(mu=0, sigma=1):
    """ Create a frozen normal distribution

    Parameters
    ----------
    mu : float
        The mean (location) of the distribution.
    sigma : float
        The standard deviation (scale) of the distribution.

    Returns
    -------
    scipy.stats.normal

    """

    return stats.norm(**process_args.normal(mu=mu, sigma=sigma))


def lognormal(mu=0, sigma=1, offset=0):
    """ Create a frozen lognormal distribution

    Parameters
    ----------
    mu : float
        The mean (location) of the underlying normal distribution.
    sigma : float
        The standard deviation (scale) of the underlying normal
        distribution.
    offset : float, optional (default = 0)
        Lateral translation parameter availabe in scipy's implementation
        of the distribution. Highly recommended to leave this at zero.

    Returns
    -------
    scipy.stats.lognormal

    """

    return stats.lognorm(**process_args.lognormal(mu=mu, sigma=sigma, offset=offset))


def beta(alpha, beta):
    """ Create a frozen beta distribution

    Parameters
    ----------
    alpha, beta : floats

    Returns
    -------
    scipy.stats.beta

    """

    return stats.beta(**process_args.beta(alpha=alpha, beta=beta))


def chi_squared(k):
    """ Create a frozen chi_squared distribution

    Parameters
    ----------
    k : float

    Returns
    -------
    scipy.stats.chi_squared

    """

    return stats.chi2(**process_args.chi_squared(k=k))


def pareto(alpha):
    """ Create a frozen pareto distribution

    Parameters
    ----------
    alpha : float

    Returns
    -------
    scipy.stats.pareto

    """

    return stats.pareto(**process_args.pareto(alpha=alpha))


def gamma(k, theta):
    """ Create a frozen gamma distribution

    Parameters
    ----------
    k, theta : floats

    Returns
    -------
    scipy.stats.gamma

    """

    return stats.gamma(**process_args.gamma(k=k, theta=theta))
