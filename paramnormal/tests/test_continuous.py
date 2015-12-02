import numpy as np

import nose.tools as nt
from numpy.random import seed
import numpy.testing as nptest

from paramnormal import continuous

@nt.nottest
def generate_knowns(np_rand_fxn, size, *args, **kwargs):
    seed(0)
    kwargs.update(dict(size=size))
    return np_rand_fxn(*args, **kwargs)


@nt.nottest
def generate_test_dist(cont_rand_fxn, size, *cargs, **ckwargs):
    seed(0)
    return cont_rand_fxn(*cargs, **ckwargs).rvs(size=size)


class CheckDist_Mixin(object):
    @nt.nottest
    def do_check(self, size):
        result = generate_test_dist(self.cont_rand_fxn, size, *self.cargs, **self.ckwds)
        known = generate_knowns(self.np_rand_fxn, size, *self.npargs, **self.npkwds)

        nptest.assert_array_almost_equal(result, known)

    def test_0010(self):
        self.do_check(10)

    def test_0037(self):
        self.do_check(37)

    def test_0100(self):
        self.do_check(100)

    def test_3737(self):
        self.do_check(3737)


class Test_uniform(CheckDist_Mixin):
    def setup(self):
        self.cont_rand_fxn = continuous.uniform
        self.cargs = []
        self.ckwds = dict(low=1, high=5)

        self.np_rand_fxn = np.random.uniform
        self.npargs = self.cargs.copy()
        self.npkwds = self.ckwds.copy()


class Test_normal(CheckDist_Mixin):
    def setup(self):
        self.cont_rand_fxn = continuous.normal
        self.cargs = []
        self.ckwds = dict(mu=4, sigma=1.75)

        self.np_rand_fxn = np.random.normal
        self.npargs = []
        self.npkwds = dict(loc=4, scale=1.75)


class Test_lognormal(CheckDist_Mixin):
    def setup(self):
        self.cont_rand_fxn = continuous.lognormal
        self.cargs = []
        self.ckwds = dict(mu=4, sigma=1.75)

        self.np_rand_fxn = np.random.lognormal
        self.npargs = self.cargs.copy()
        self.npkwds = dict(mean=4, sigma=1.75)


class Test_beta(CheckDist_Mixin):
    def setup(self):
        self.cont_rand_fxn = continuous.beta
        self.cargs = [2, 3]
        self.ckwds = dict()

        self.np_rand_fxn = np.random.beta
        self.npargs = self.cargs.copy()
        self.npkwds = self.ckwds.copy()
