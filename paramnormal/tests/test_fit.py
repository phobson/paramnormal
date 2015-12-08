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

    @nt.nottest
    @staticmethod
    def check_params(params, expected_mu, expected_sigma):
        assert_close_enough(params.mu, expected_mu)
        assert_close_enough(params.sigma, expected_sigma)

    def test_min_guesses(self):
        self.check_params(
            self.fitter(self.data),
            4.1709713618,
            7.2770395662,
        )

class Test_lognormal(object):
    @seed
    def setup(self):
        self.data = numpy.random.lognormal(mean=2.0, sigma=6.7, size=37)
        self.fitter = fit.lognormal

    @nt.nottest
    @staticmethod
    def check_params(params, expected_mu, expected_sigma, expected_offset):
        assert_close_enough(params.mu, expected_mu)
        assert_close_enough(params.sigma, expected_sigma)
        assert_close_enough(params.offset, expected_offset)

    def test_min_guesses(self):
        self.check_params(
            self.fitter(self.data),
            4.1709713618,
            7.2770395662,
            0.0
        )
