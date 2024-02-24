import numpy
import pytest
from matplotlib import pyplot
from scipy import stats

from paramnormal import activity, dist
from paramnormal.utils import seed

BASELINE_DIR = "baseline_images/test_activity"
TOLERANCE = 15


def assert_dists_are_equivalent(dist1, dist2):
    numpy.random.seed(0)
    x1 = dist1.rvs(3)

    numpy.random.seed(0)
    x2 = dist2.rvs(3)

    assert numpy.all((x1 - x2) < 0.0001)


def check_params(*value_pairs):
    for result, expected in value_pairs:
        assert (result - expected) < 0.00001


def test_string_bad():
    with pytest.raises(ValueError):
        activity._check_distro("junk")


def test_number():
    with pytest.raises(ValueError):
        activity._check_distro(45)


def test_pndist_as_class():
    assert activity._check_distro(dist.normal, as_class=True) == dist.normal


def test_string_good_as_class():
    assert activity._check_distro("normal", as_class=True) == dist.normal


def test_pndist():
    assert_dists_are_equivalent(
        activity._check_distro(dist.normal, mu=0, sigma=1), stats.norm(0, 1)
    )


def test_string():
    assert_dists_are_equivalent(activity._check_distro("normal", mu=0, sigma=1), stats.norm(0, 1))


def test_scipy_dist():
    assert_dists_are_equivalent(activity._check_distro(stats.lognorm(s=2)), stats.lognorm(s=2))


@pytest.mark.parametrize("ax", [None, pyplot.gca(), "junk"])
def test__check_ax(ax):
    if ax == "junk":
        with pytest.raises(AttributeError):
            activity._check_ax(ax)
    else:
        fig, ax1 = activity._check_ax(ax)

        assert isinstance(fig, pyplot.Figure)
        assert isinstance(ax1, pyplot.Axes)
        if ax is not None:
            assert ax == ax1


def test_random_normal():
    numpy.random.seed(0)
    x1 = activity.random("normal", mu=0, sigma=1, shape=(3, 4))

    numpy.random.seed(0)
    x2 = numpy.random.normal(0, 1, size=(3, 4))
    assert numpy.all((x1 - x2) < 0.0001)


def test_random_beta():
    numpy.random.seed(0)
    x1 = activity.random("beta", alpha=2, beta=3, shape=(5, 2))

    numpy.random.seed(0)
    x2 = numpy.random.beta(2, 3, size=(5, 2))
    assert numpy.all((x1 - x2) < 0.0001)


@seed
def test_create_normal():
    data = numpy.random.normal(loc=2.0, scale=6.7, size=37)
    params = activity.fit("normal", data)
    dist = activity.fit("normal", data, as_params=False)

    check_params(
        (params.mu, 4.1709713618),
        (params.sigma, 7.2770395662),
    )

    assert_dists_are_equivalent(dist, stats.norm(params.mu, params.sigma))


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_pdf_basic():
    # first
    fig, ax1 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    ax1 = activity.plot(norm_dist, ax=ax1)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_pdf_fit():
    # second
    fig2, ax2 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    data = activity.random("normal", μ=5.4, σ=2.5, shape=37)
    ax2 = activity.plot(norm_dist, ax=ax2, line_opts=dict(label="Theoretical PDF"))
    ax2 = activity.plot("normal", data=data, ax=ax2, line_opts=dict(label="Fit PDF"))
    ax2.legend()
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_pdf_xlog():
    # first
    fig, ax1 = pyplot.subplots()
    loc_dist = dist.lognormal(μ=1.25, σ=0.75)
    ax1 = activity.plot(loc_dist, ax=ax1, xscale="log")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_cdf_basic():
    # first
    fig, ax1 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    ax1 = activity.plot(norm_dist, ax=ax1, which="cdf")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_cdf_fit():
    # second
    fig2, ax2 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    data = activity.random("normal", μ=5.4, σ=2.5, shape=37)
    ax2 = activity.plot(norm_dist, ax=ax2, line_opts=dict(label="Theoretical CDF"), which="cdf")
    ax2 = activity.plot("normal", data=data, ax=ax2, line_opts=dict(label="Fit CDF"), which="cdf")
    ax2.legend()
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_cdf_xlog():
    # first
    fig, ax1 = pyplot.subplots()
    loc_dist = dist.lognormal(μ=1.25, σ=0.75)
    ax1 = activity.plot(loc_dist, ax=ax1, xscale="log", which="CDF")
    ax1.legend()
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_sf_basic():
    # first
    fig, ax1 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    ax1 = activity.plot(norm_dist, ax=ax1, which="sf")
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_sf_fit():
    # second
    fig2, ax2 = pyplot.subplots()
    norm_dist = dist.normal(μ=5.4, σ=2.5)
    data = activity.random("normal", μ=5.4, σ=2.5, shape=37)
    ax2 = activity.plot(norm_dist, ax=ax2, line_opts=dict(label="Theoretical sf"), which="sf")
    ax2 = activity.plot("normal", data=data, ax=ax2, line_opts=dict(label="Fit sf"), which="sf")
    ax2.legend()
    return fig2


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=TOLERANCE)
@seed
def test_plot_sf_xlog():
    # first
    fig, ax1 = pyplot.subplots()
    loc_dist = dist.lognormal(μ=1.25, σ=0.75)
    ax1 = activity.plot(loc_dist, ax=ax1, xscale="log", which="sf")
    ax1.legend()
    return fig


def test_plot_bad_attribute():
    with pytest.raises(AttributeError):
        loc_dist = dist.lognormal(μ=1.25, σ=0.75)
        activity.plot(loc_dist, xscale="log", which="JUNK")
