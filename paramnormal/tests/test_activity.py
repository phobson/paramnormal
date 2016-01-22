import numpy
from matplotlib import pyplot
from scipy import stats

import nose.tools as nt
import numpy.testing as nptest

from paramnormal import activity
from paramnormal import dist
from paramnormal.utils import seed


def assert_dists_are_equivalent(dist1, dist2):
    numpy.random.seed(0)
    x1 = dist1.rvs(3)

    numpy.random.seed(0)
    x2 = dist2.rvs(3)

    nptest.assert_array_almost_equal(x1, x2)


@nt.nottest
def check_params(*value_pairs):
    for result, expected in value_pairs:
        nt.assert_almost_equal(result, expected, places=5)


class Test__check_distro(object):
    @nt.raises(ValueError)
    def test_string_bad(self):
        activity._check_distro('junk')

    @nt.raises(ValueError)
    def test_number(self):
        activity._check_distro(45)

    def test_pndist_as_class(self):
        nt.assert_equal(activity._check_distro(dist.normal, as_class=True), dist.normal)

    def test_string_good_as_class(self):
        nt.assert_equal(activity._check_distro('normal', as_class=True), dist.normal)

    def test_pndist(self):
        assert_dists_are_equivalent(
            activity._check_distro(dist.normal, mu=0, sigma=1),
            stats.norm(0, 1)
        )

    def test_string(self):
        assert_dists_are_equivalent(
            activity._check_distro('normal', mu=0, sigma=1),
            stats.norm(0, 1)
        )

    def test_scipy_dist(self):
        assert_dists_are_equivalent(
            activity._check_distro(stats.lognorm(s=2)),
            stats.lognorm(s=2)
        )


class Test__check_ax(object):
    def test_None(self):
        fig, ax = activity._check_ax(None)
        nt.assert_true(isinstance(fig, pyplot.Figure))
        nt.assert_true(isinstance(ax, pyplot.Axes))

    def test_ax(self):
        fig, ax = pyplot.subplots()

        fig1, ax1 = activity._check_ax(ax)
        nt.assert_equal(ax, ax1)
        nt.assert_equal(fig, fig1)

    @nt.raises(ValueError)
    def test_error(self):
        activity._check_ax('junk')


class Test_random(object):
    def test_normal(self):
        numpy.random.seed(0)
        x1 = activity.random('normal', mu=0, sigma=1, shape=(3, 4))

        numpy.random.seed(0)
        x2 = numpy.random.normal(0, 1, size=(3, 4))
        nptest.assert_array_almost_equal(x1, x2)

    def test_beta(self):
        numpy.random.seed(0)
        x1 = activity.random('beta', alpha=2, beta=3, shape=(5, 2))

        numpy.random.seed(0)
        x2 = numpy.random.beta(2, 3, size=(5, 2))
        nptest.assert_array_almost_equal(x1, x2)


class Test_fit(object):
    @seed
    def test_normal(self):
        data = numpy.random.normal(loc=2.0, scale=6.7, size=37)
        params = activity.fit('normal', data)
        dist = activity.fit('normal', data, as_params=False)

        check_params(
            (params.mu, 4.1709713618),
            (params.sigma, 7.2770395662),
        )

        assert_dists_are_equivalent(dist, stats.norm(params.mu, params.sigma))