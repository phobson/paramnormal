import pytest

from paramnormal import dist, utils


@pytest.mark.parametrize(
    ("distro", "distargs", "expected"),
    [
        (dist.normal, dict(mu=1, sigma=2), dict(loc=1, scale=2)),
        (dist.normal, dict(μ=1, σ=2), dict(loc=1, scale=2)),
        (dist.beta, dict(alpha=1, beta=2), dict(a=1, b=2, loc=0, scale=1)),
        (dist.beta, dict(α=1, β=2), dict(a=1, b=2, loc=0, scale=1)),
        (dist.gamma, dict(k=1, theta=2), dict(a=1, loc=0, scale=2)),
        (dist.gamma, dict(k=1, θ=2), dict(a=1, loc=0, scale=2)),
    ],
)
def test_greco_deco(distro, distargs, expected):
    result = distro._process_args(**distargs)
    assert result == expected


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [
        (1, 2, 3, dict(a=1, b=2, c=3)),
        (1, None, 3, dict(a=1, c=3)),
        (None, None, None, dict()),
    ],
)
def test__remove_nones(a, b, c, expected):
    result = utils._remove_nones(a=a, b=b, c=c)
    assert result == expected


@pytest.mark.parametrize(
    ("fit", "expected"),
    [
        (True, ("floc", "fscale")),
        (False, ("loc", "scale")),
    ],
)
def test__get_loc_scale_keys(fit, expected):
    result = utils._get_loc_scale_keys(fit=fit)
    assert result == expected
