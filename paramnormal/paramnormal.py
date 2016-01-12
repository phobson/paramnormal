from collections import namedtuple

import numpy
from scipy import stats

from . import utils


class BaseDist_Mixin(object):
    @classmethod
    @utils.greco_deco
    def _fit(cls, data, **guesses):
        args = utils._pop_none(**cls._process_args(fit=True, **guesses))
        _sp_params = cls.dist.fit(data, **args)
        return _sp_params

    @utils.greco_deco
    def __new__(self, **params):
        dist_params = self._process_args(**params['params'], fit=False)
        return self.dist(**dist_params)


class normal(BaseDist_Mixin):
    dist = stats.norm
    params = namedtuple('params', ['mu', 'sigma'])

    @classmethod
    def _process_args(cls, mu=None, sigma=None, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        return {loc_key: mu, scale_key: sigma}

    @classmethod
    def fit(cls, data, **guess):
        return cls.params(*cls._fit(data, **guesses))


class lognormal(BaseDist_Mixin):
    dist = stats.lognorm
    params = namedtuple('params', ['mu', 'sigma', 'offset'])

    @classmethod
    def _process_args(cls, mu=None, sigma=None, offset=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        if fit:
            key = 'f0'
        else:
            key = 's'
        if offset is None:
            raise ValueError("`offset` parameter is required. Recommended value is 0.")
        return {key: sigma, scale_key: numpy.exp(mu) if mu is not None else mu, loc_key: offset}

    @classmethod
    def fit(cls, data, **guesses):
        params = cls._fit(data, **guesses)
        return cls.params(mu=numpy.log(params[2]), sigma=params[0], offset=params[1])
