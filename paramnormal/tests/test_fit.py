from collections import namedtuple
from functools import wraps

import numpy

import nose.tools as nt

from paramnormal import fit


@nt.nottest
def seed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        numpy.random.seed(0)
        return func(*args, **kwargs)
    return wrapper


@nt.nottest
def assert_close_enough(actual, expected):
    nt.assert_almost_equal(actual, expected, places=5)


@nt.nottest
def check_params(*value_pairs):
    for result, expected in value_pairs:
        assert_close_enough(result, expected)


def test__pop_none():
    nt.assert_dict_equal(
        fit._pop_none(a=None, b=1, c=None),
        dict(b=1)
    )


class Test_normal(object):
    @seed
    def setup(self):
        self.data = numpy.random.normal(loc=2.0, scale=6.7, size=37)
        self.fitter = fit.normal

    def test_min_guesses(self):
        params = self.fitter(self.data)
        check_params(
            (params.mu, 4.1709713618),
            (params.sigma, 7.2770395662),
        )


class Test_lognormal(object):
    @seed
    def setup(self):
        self.data = numpy.random.lognormal(mean=2.0, sigma=6.7, size=37)
        self.fitter = fit.lognormal

    def test_min_guesses(self):
        params = self.fitter(self.data)
        check_params(
            (params.mu, 4.1709713618),
            (params.sigma, 7.2770395662),
            (params.offset, 0.0)
        )


class Test_beta(object):
    @seed
    def setup(self):
        self.data = numpy.random.beta(2, 7, size=37)
        self.fitter = fit.beta

    def test_min_guess(self):
        params = self.fitter(self.data)
        check_params(
            (params.alpha, 1.65675833325),
            (params.beta, 5.78176888942),
            (params.loc, 0),
            (params.scale, 1),
        )

    def test_guess_alpha(self):
        params = self.fitter(self.data, alpha=2)
        check_params(
            (params.alpha, 2),
            (params.beta, 6.8812340590409891),
            (params.loc, 0),
            (params.scale, 1),
        )

    def test_guess_beta(self):
        params = self.fitter(self.data, beta=7)
        check_params(
            (params.alpha, 1.91476622934291),
            (params.beta, 7),
            (params.loc, 0),
            (params.scale, 1),
        )


class Test_weibull(object):
    @seed
    def setup(self):
        self.data = numpy.random.weibull(2, size=37)
        self.fitter = fit.weibull

    def test_min_guess(self):
        params = self.fitter(self.data)
        check_params(
            (params.k, 2.1663085937500024),
            (params.loc, 0),
            (params.scale, 1),
        )

