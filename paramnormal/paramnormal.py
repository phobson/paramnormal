from collections import namedtuple

import numpy
from scipy import stats

from . import utils


class BaseDist_Mixin:
    def __new__(cls, **params):
        dist_params = cls._process_args(fit=False, **params)
        return cls.dist(**dist_params)

    @classmethod
    def _fit(cls, data, **guesses):
        args = utils._pop_none(**cls._process_args(fit=True, **guesses))
        _sp_params = cls.dist.fit(data, **args)
        return _sp_params

    @classmethod
    def fit(cls, data, **guesses):
        return cls.param_template(*cls._fit(data, **guesses))


class normal(BaseDist_Mixin):
    dist = stats.norm
    param_template = namedtuple('params', ['mu', 'sigma'])
    name = 'normal'

    @staticmethod
    @utils.greco_deco
    def _process_args(mu=None, sigma=None, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        return {loc_key: mu, scale_key: sigma}


class lognormal(BaseDist_Mixin):
    dist = stats.lognorm
    param_template = namedtuple('params', ['mu', 'sigma', 'offset'])
    name = 'lognormal'

    @staticmethod
    @utils.greco_deco
    def _process_args(mu=None, sigma=None, offset=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 's'
        if offset is None and not fit:
            raise ValueError("`offset` parameter is required. Recommended value is 0.")
        return {key: sigma, scale_key: numpy.exp(mu) if mu is not None else mu, loc_key: offset}

    @classmethod
    def fit(cls, data, **guesses):
        params = cls._fit(data, **guesses)
        return cls.param_template(mu=numpy.log(params[2]), sigma=params[0], offset=params[1])


class weibull(BaseDist_Mixin):
    dist = stats.weibull_min
    param_template = namedtuple('params', ['k', 'loc', 'scale'])
    name = 'weibull'

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 'c'
        return {key: k, loc_key: loc, scale_key: 1}


class alpha(BaseDist_Mixin):
    dist = stats.alpha
    param_template = namedtuple('params', ['alpha', 'loc', 'scale'])

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            alpha_key = 'f0'
        else:
            alpha_key = 'a'
        return {alpha_key: alpha, loc_key: loc, scale_key: scale}


class beta(BaseDist_Mixin):
    dist = stats.beta
    param_template = namedtuple('params', ['alpha', 'beta', 'loc', 'scale'])

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, beta=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            alpha_key = 'f0'
            beta_key = 'f1'
        else:
            alpha_key = 'a'
            beta_key = 'b'
        return {alpha_key: alpha, beta_key: beta, loc_key: loc, scale_key: scale}


class gamma(BaseDist_Mixin):
    dist = stats.gamma
    param_template = namedtuple('params', ['k', 'loc', 'theta'])

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, theta=None, loc=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 'a'
        return {key: k, loc_key: loc, scale_key: theta}


class chi_squared(BaseDist_Mixin):
    dist = stats.chi2
    param_template = namedtuple('params', ['k', 'loc', 'scale'])

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 'df'
        return {key: k, loc_key: loc, scale_key: scale}


class pareto(BaseDist_Mixin):
    dist = stats.pareto
    param_template = namedtuple('params', ['alpha', 'loc', 'scale'])

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 'b'
        return {key: alpha, loc_key: loc, scale_key: scale}
