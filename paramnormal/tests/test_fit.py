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

        )
