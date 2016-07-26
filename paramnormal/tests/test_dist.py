from functools import wraps

import numpy
from scipy import stats

import pytest
import numpy.testing as nptest

from paramnormal import dist
from paramnormal.utils import seed


@seed
def generate_knowns(np_rand_fxn, size, *args, **kwargs):
    # numpy.random.pareto is actually a Lomax and needs
    # to be shifted by 1
    shift = kwargs.pop('shift', 0)
    kwargs.update(dict(size=size))
    return np_rand_fxn(*args, **kwargs) + shift


@seed
def generate_test_dist(dist, size, *cargs, **ckwargs):
    return dist(*cargs, **ckwargs).rvs(size=size)


def check_params(*value_pairs):
    for result, expected in value_pairs:
        assert (result - expected) < 0.00001


class CheckDist_Mixin(object):

    def do_check(self, size):
        result = generate_test_dist(self.dist, size, *self.cargs, **self.ckwds)
        known = generate_knowns(self.np_rand_fxn, size, *self.npargs, **self.npkwds)

        nptest.assert_array_almost_equal(result, known)

    def test_random_0010(self):
        self.do_check(10)

    def test_random_0037(self):
        self.do_check(37)

    def test_random_0100(self):
        self.do_check(100)

    def test_random_3737(self):
        self.do_check(3737)

    def test_from_params(self):
        data = generate_test_dist(self.dist, 37, *self.cargs, **self.ckwds)
        params = self.dist.fit(data)
        newdist = self.dist.from_params(params)
        assert (isinstance(newdist, stats._distn_infrastructure.rv_frozen))

    def test_xxx(self):
        self.dist(*self.cargs, **self.ckwds)


class Test_normal(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.normal
        self.cargs = []
        self.ckwds = dict(mu=4, sigma=1.75)

        self.np_rand_fxn = numpy.random.normal
        self.npargs = []
        self.npkwds = dict(loc=4, scale=1.75)

    def test_processargs(self):
        result = self.dist._process_args(mu=2, sigma=2.45)
        expected = dict(loc=2, scale=2.45)
        assert result == expected

        result = self.dist._process_args(mu=2, sigma=2.45, fit=True)
        expected = dict(floc=2, fscale=2.45)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.normal(loc=2.0, scale=6.7, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.mu, 4.1709713618),
            (params.sigma, 7.2770395662),
        )


class Test_lognormal(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.lognormal
        self.cargs = []
        self.ckwds = dict(mu=4, sigma=1.75)

        self.np_rand_fxn = numpy.random.lognormal
        self.npargs = self.cargs.copy()
        self.npkwds = dict(mean=4, sigma=1.75)

    def test_process_args(self):
        result = self.dist._process_args(mu=2, sigma=2.45)
        expected = dict(scale=numpy.exp(2), s=2.45, loc=0)
        assert result == expected

        result = self.dist._process_args(mu=2, sigma=2.45, fit=True)
        expected = dict(fscale=numpy.exp(2), f0=2.45, floc=0)
        assert result == expected

    def test_process_args_no_offset(self):
        with pytest.raises(ValueError):
            self.dist._process_args(offset=None)

    @seed
    def test_fit(self):
        data = numpy.random.lognormal(mean=2.0, sigma=6.7, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.mu, 4.1709713618),
            (params.sigma, 7.2770395662),
            (params.offset, 0.0)
        )


class Test_weibull(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.weibull
        self.cargs = []
        self.ckwds = dict(k=2)

        self.np_rand_fxn = numpy.random.weibull
        self.npargs = [2]
        self.npkwds = dict()

    def test_process_args(self):
        result = self.dist._process_args(k=2)
        expected = dict(c=2, loc=0, scale=1)
        assert result == expected

        result = self.dist._process_args(k=2, fit=True)
        expected = dict(f0=2, floc=0, fscale=1)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.weibull(2, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.k, 2.1663085937500024),
            (params.loc, 0),
            (params.scale, 1),
        )


class Test_alpha(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.alpha
        self.cargs = []
        self.ckwds = dict(alpha=2)


    def do_check(self, size):
        pass

    def test_process_args(self):
        result = self.dist._process_args(alpha=2)
        expected = dict(a=2, loc=0, scale=1)
        assert result == expected

        result = self.dist._process_args(alpha=2, fit=True)
        expected = dict(f0=2, floc=0, fscale=1)
        assert result == expected

    @seed
    def test_fit(self):
        data = stats.alpha(5).rvs(size=37)
        params = self.dist.fit(data)
        check_params(
            (params.alpha, 4.8356445312500096),
            (params.loc, 0),
            (params.scale, 1),
        )


class Test_beta(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.beta
        self.cargs = []
        self.ckwds = dict(alpha=2, beta=3)

        self.np_rand_fxn = numpy.random.beta
        self.npargs = [2, 3]
        self.npkwds = dict()

    def test_process_args(self):
        result = self.dist._process_args(alpha=2, beta=5)
        expected = dict(a=2, b=5, loc=0, scale=1)
        assert result == expected

        result = self.dist._process_args(alpha=2, beta=5, fit=True)
        expected = dict(f0=2, f1=5, floc=0, fscale=1)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.beta(2, 7, size=37)

        no_guesses = self.dist.fit(data)
        check_params(
            (no_guesses.alpha, 1.65675833325),
            (no_guesses.beta, 5.78176888942),
            (no_guesses.loc, 0),
            (no_guesses.scale, 1),
        )

        guess_alpha = self.dist.fit(data, alpha=2)
        check_params(
            (guess_alpha.alpha, 2),
            (guess_alpha.beta, 6.8812340590409891),
            (guess_alpha.loc, 0),
            (guess_alpha.scale, 1),
        )

        guess_beta = self.dist.fit(data, beta=7)
        check_params(
            (guess_beta.alpha, 1.91476622934291),
            (guess_beta.beta, 7),
            (guess_beta.loc, 0),
            (guess_beta.scale, 1),
        )


class Test_gamma(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.gamma
        self.cargs = []
        self.ckwds = dict(k=2, theta=1)

        self.np_rand_fxn = numpy.random.gamma
        self.npargs = [2, 1]
        self.npkwds = dict()

    def test_process_args(self):
        result = self.dist._process_args(k=1, theta=2)
        expected = dict(a=1, loc=0, scale=2)
        assert result == expected

        result = self.dist._process_args(k=1, theta=2, fit=True)
        expected = dict(f0=1, floc=0, fscale=2)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.gamma(2, 5, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.k, 1.3379069223213478),
            (params.loc, 0),
            (params.theta, 7.5830062081633587),
        )


class Test_chi_squared(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.chi_squared
        self.cargs = []
        self.ckwds = dict(k=2)

        self.np_rand_fxn = numpy.random.chisquare
        self.npargs = [2]
        self.npkwds = dict()

    def test_process_args(self):
        result = self.dist._process_args(k=5)
        expected = dict(df=5, loc=0, scale=1)
        assert result == expected

        result = self.dist._process_args(k=5, fit=True)
        expected = dict(f0=5, floc=0, fscale=1)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.chisquare(2, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.k, 2.2668945312500028),
            (params.loc, 0),
            (params.scale, 1),
        )


class Test_pareto(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.pareto
        self.cargs = []
        self.ckwds = dict(alpha=2)

        self.np_rand_fxn = numpy.random.pareto
        self.npargs = [2]
        self.npkwds = dict(shift=1)

    def test_process_args(self):
        result = self.dist._process_args(alpha=4.78)
        expected = dict(b=4.78, loc=0, scale=1)
        assert result == expected

        result = self.dist._process_args(alpha=4.78, fit=True)
        expected = dict(f0=4.78, floc=0, fscale=1)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.pareto(a=2, size=37) + 1
        params = self.dist.fit(data)
        check_params(
            (params.alpha, 1.7850585937500019),
            (params.loc, 0),
            (params.scale, 1),
        )


class Test_exponential(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.exponential
        self.cargs = []
        self.ckwds = dict(lamda=2)

        self.np_rand_fxn = numpy.random.exponential
        self.npargs = [0.5]
        self.npkwds = dict()

    def test_process_args(self):
        result = self.dist._process_args(lamda=2.0)
        expected = dict(loc=0, scale=0.5)
        assert result == expected

        result = self.dist._process_args(lamda=2.0, fit=True)
        expected = dict(floc=0, fscale=0.5)
        assert result == expected

    @seed
    def test_fit(self):
        data = numpy.random.exponential(0.5, size=37)
        params = self.dist.fit(data)
        check_params(
            (params.lamda, 1.7849050026146085),
            (params.loc, 0),
        )


class Test_rice(CheckDist_Mixin):
    def setup(self):
        self.dist = dist.rice
        self.cargs = []
        self.ckwds = dict(R=10, sigma=2)

        self.np_rand_fxn = stats.rice.rvs
        self.npargs = [5]
        self.npkwds = dict(loc=0, scale=2)

    def test_processargs(self):
        result = self.dist._process_args(R=10, sigma=2)
        expected = dict(b=5, loc=0, scale=2)
        assert result == expected

        result = self.dist._process_args(R=10, sigma=2, fit=True)
        expected = dict(b=5, floc=0, fscale=2)
        assert result == expected

    @seed
    def test_fit(self):
        data = stats.rice(5, loc=0, scale=2).rvs(size=37)
        params = self.dist.fit(data)
        check_params(
            (params.R, 10.100674084593422),
            (params.sigma, 1.759817171541185),
            (params.loc, 0),
        )
