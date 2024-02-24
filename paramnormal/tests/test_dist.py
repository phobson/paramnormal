import numpy
import numpy.testing as nptest
import pytest
from scipy import stats

from paramnormal import dist
from paramnormal.utils import seed


@seed
def generate_knowns(np_rand_fxn, size, *args, **kwargs):
    # numpy.random.pareto is actually a Lomax and needs
    # to be shifted by 1
    shift = kwargs.pop("shift", 0)
    kwargs.update(dict(size=size))
    return np_rand_fxn(*args, **kwargs) + shift


@seed
def generate_test_dist(distro, size, *cargs, **ckwargs):
    return distro(*cargs, **ckwargs).rvs(size=size)


def check_params(*value_pairs):
    for result, expected in value_pairs:
        assert (result - expected) < 0.0001


@pytest.mark.parametrize(
    "distro, cargs, ckwds, np_rand_fxn, npargs, npkwds",
    [
        (dist.normal, [], dict(mu=4, sigma=1.75), numpy.random.normal, [], dict(loc=4, scale=1.75)),
        (
            dist.lognormal,
            [],
            dict(mu=4, sigma=1.75),
            numpy.random.lognormal,
            [],
            dict(mean=4, sigma=1.75),
        ),
        (dist.weibull, [], dict(k=2), numpy.random.weibull, [2], dict()),
        (dist.beta, [], dict(alpha=2, beta=3), numpy.random.beta, [2, 3], dict()),
        (dist.gamma, [], dict(k=2, theta=1), numpy.random.gamma, [2, 1], dict()),
        (dist.chi_squared, [], dict(k=2), numpy.random.chisquare, [2], dict()),
        (dist.pareto, [], dict(alpha=2), numpy.random.pareto, [2], dict(shift=1)),
        (dist.rice, [], dict(R=10, sigma=2), stats.rice.rvs, [5], dict(loc=0, scale=2)),
    ],
)
@pytest.mark.parametrize("size", [10, 37, 100, 3737])
def test_random(size, distro, cargs, ckwds, np_rand_fxn, npargs, npkwds):
    result = generate_test_dist(distro, size, *cargs, **ckwds)
    known = generate_knowns(np_rand_fxn, size, *npargs, **npkwds)
    nptest.assert_array_almost_equal(result, known)


@pytest.mark.parametrize(
    "distro, cargs, ckwds",
    [
        (dist.normal, [], dict(mu=4, sigma=1.75)),
        (dist.lognormal, [], dict(mu=4, sigma=1.75)),
        (dist.weibull, [], dict(k=2)),
        (dist.alpha, [], dict(alpha=2)),
        (dist.beta, [], dict(alpha=2, beta=3)),
        (dist.gamma, [], dict(k=2, theta=1)),
        (dist.chi_squared, [], dict(k=2)),
        (dist.pareto, [], dict(alpha=2)),
        (dist.rice, [], dict(R=10, sigma=2)),
    ],
)
def test_from_params(distro, cargs, ckwds):
    data = generate_test_dist(distro, 37, *cargs, **ckwds)
    params = distro.fit(data)
    newdist = distro.from_params(params)
    assert isinstance(newdist, stats._distn_infrastructure.rv_frozen)


@pytest.mark.parametrize(
    "distro, cargs, ckwds",
    [
        (dist.normal, [], dict(mu=4, sigma=1.75)),
        (dist.lognormal, [], dict(mu=4, sigma=1.75)),
        (dist.weibull, [], dict(k=2)),
        (dist.alpha, [], dict(alpha=2)),
        (dist.beta, [], dict(alpha=2, beta=3)),
        (dist.gamma, [], dict(k=2, theta=1)),
        (dist.chi_squared, [], dict(k=2)),
        (dist.pareto, [], dict(alpha=2)),
        (dist.rice, [], dict(R=10, sigma=2)),
    ],
)
def test_xxx(distro, cargs, ckwds):
    distro(*cargs, **ckwds)


@pytest.mark.parametrize(
    "distro, ckwds, fit, expected",
    [
        (dist.normal, dict(mu=2, sigma=2.45), False, dict(loc=2, scale=2.45)),
        (dist.normal, dict(mu=2, sigma=2.45), True, dict(floc=2, fscale=2.45)),
        (dist.lognormal, dict(mu=2, sigma=2.45), False, dict(scale=numpy.exp(2), s=2.45, loc=0)),
        (dist.lognormal, dict(mu=2, sigma=2.45), True, dict(fscale=numpy.exp(2), f0=2.45, floc=0)),
        (dist.weibull, dict(k=2), False, dict(c=2, loc=0, scale=1)),
        (dist.weibull, dict(k=2), True, dict(f0=2, floc=0, fscale=1)),
        (dist.alpha, dict(alpha=2), False, dict(a=2, loc=0, scale=1)),
        (dist.alpha, dict(alpha=2), True, dict(f0=2, floc=0, fscale=1)),
        (dist.beta, dict(alpha=2, beta=5), False, dict(a=2, b=5, loc=0, scale=1)),
        (dist.beta, dict(alpha=2, beta=5), True, dict(f0=2, f1=5, floc=0, fscale=1)),
        (dist.gamma, dict(k=1, theta=2), False, dict(a=1, loc=0, scale=2)),
        (dist.gamma, dict(k=1, theta=2), True, dict(f0=1, floc=0, fscale=2)),
        (dist.chi_squared, dict(k=5), False, dict(df=5, loc=0, scale=1)),
        (dist.chi_squared, dict(k=5), True, dict(f0=5, floc=0, fscale=1)),
        (dist.pareto, dict(alpha=4.78), False, dict(b=4.78, loc=0, scale=1)),
        (dist.pareto, dict(alpha=4.78), True, dict(f0=4.78, floc=0, fscale=1)),
        (dist.exponential, dict(lamda=2.0), False, dict(loc=0, scale=0.5)),
        (dist.exponential, dict(lamda=2.0), True, dict(floc=0, fscale=0.5)),
        (dist.rice, dict(R=10, sigma=2), False, dict(b=5, loc=0, scale=2)),
        (dist.rice, dict(R=10, sigma=2), True, dict(fb=5, floc=0, fscale=2)),
    ],
)
def test_processargs(distro, ckwds, fit, expected):
    result = distro._process_args(**ckwds, fit=fit)
    assert result == expected


def test_process_args_no_offset():
    with pytest.raises(ValueError):
        dist.lognormal._process_args(offset=None)


@seed
def test_fit_normal():
    data = numpy.random.normal(loc=2.0, scale=6.7, size=37)
    params = dist.normal.fit(data)
    check_params(
        (params.mu, 4.1709713618),
        (params.sigma, 7.2770395662),
    )


@seed
def test_fit_lognormal():
    data = numpy.random.lognormal(mean=2.0, sigma=6.7, size=37)
    params = dist.lognormal.fit(data)
    check_params((params.mu, 4.1709713618), (params.sigma, 7.2770395662), (params.offset, 0.0))


@seed
def test_fit_weibull():
    data = numpy.random.weibull(2, size=37)
    params = dist.weibull.fit(data)
    check_params(
        (params.k, 2.1663085937500024),
        (params.loc, 0),
        (params.scale, 1),
    )


@seed
def test_fit_alpha():
    data = stats.alpha(5).rvs(size=37)
    params = dist.alpha.fit(data)
    check_params(
        (params.alpha, 4.8356445312500096),
        (params.loc, 0),
        (params.scale, 1),
    )


@seed
def test_fit_beta():
    data = numpy.random.beta(2, 7, size=37)

    no_guesses = dist.beta.fit(data)
    check_params(
        (no_guesses.alpha, 1.65675833325),
        (no_guesses.beta, 5.78176888942),
        (no_guesses.loc, 0),
        (no_guesses.scale, 1),
    )

    guess_alpha = dist.beta.fit(data, alpha=2)
    check_params(
        (guess_alpha.alpha, 2),
        (guess_alpha.beta, 6.8812340590409891),
        (guess_alpha.loc, 0),
        (guess_alpha.scale, 1),
    )

    guess_beta = dist.beta.fit(data, beta=7)
    check_params(
        (guess_beta.alpha, 1.91476622934291),
        (guess_beta.beta, 7),
        (guess_beta.loc, 0),
        (guess_beta.scale, 1),
    )


@seed
def test_fit_gamma():
    data = numpy.random.gamma(2, 5, size=37)
    params = dist.gamma.fit(data)
    check_params(
        (params.k, 1.3379069223213478),
        (params.loc, 0),
        (params.theta, 7.5830062081633587),
    )


@seed
def test_fit_chi_squareed():
    data = numpy.random.chisquare(2, size=37)
    params = dist.chi_squared.fit(data)
    check_params(
        (params.k, 2.2668945312500028),
        (params.loc, 0),
        (params.scale, 1),
    )


@seed
def test_fit_pareto():
    data = numpy.random.pareto(a=2, size=37) + 1
    params = dist.pareto.fit(data)
    check_params(
        (params.alpha, 1.7850585937500019),
        (params.loc, 0),
        (params.scale, 1),
    )


@seed
def test_fit_exponential():
    data = numpy.random.exponential(0.5, size=37)
    params = dist.exponential.fit(data)
    check_params(
        (params.lamda, 1.785060162078026),
        (params.loc, 0),
    )


@seed
def test_fit_rice():
    data = stats.rice(5, loc=0, scale=2).rvs(size=37)
    params = dist.rice.fit(data)
    check_params(
        (params.R, 10.100674084593422),
        (params.sigma, 1.759817171541185),
        (params.loc, 0),
    )
