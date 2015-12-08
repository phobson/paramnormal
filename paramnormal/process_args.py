import numpy as np


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


def normal(mu=None, sigma=None):
    return {'loc': mu, 'scale': sigma}


def lognormal(mu=None, sigma=None, offset=0):
    return {'s': sigma, 'scale': np.exp(mu) if mu is not None else mu, 'loc': offset}


def beta(alpha=None, beta=None):
    return {'a': alpha, 'b': beta}


def chi_squared(k=None):
    return {'df': k}


def pareto(alpha=None):
    return {'b': alpha}


def gamma(k=None, theta=None):
    return {'a': k, 'scale': theta}