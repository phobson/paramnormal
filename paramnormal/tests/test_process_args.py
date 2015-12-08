import numpy as np

import nose.tools as nt
from numpy.random import seed
import numpy.testing as nptest

from paramnormal import process_args


def test_uniform():
    nt.assert_dict_equal(
        process_args.uniform(low=4, high=9),
        dict(loc=4, scale=5)
    )

@nt.raises(ValueError)
def test_uniform_no_low():
    process_args.uniform(high=9)


def test_normal():
    nt.assert_dict_equal(
        process_args.normal(mu=2, sigma=2.45),
        dict(loc=2, scale=2.45)
    )


def test_lognormal():
    nt.assert_dict_equal(
        process_args.lognormal(mu=2, sigma=2.45),
        dict(scale=np.exp(2), s=2.45, loc=0)
    )


def test_beta():
    nt.assert_dict_equal(
        process_args.beta(alpha=2, beta=5),
        dict(a=2, b=5)
    )


def test_chi_squared():
    nt.assert_dict_equal(
        process_args.chi_squared(k=5),
        dict(df=5)
    )


def test_pareto():
    nt.assert_dict_equal(
        process_args.pareto(alpha=4.78),
        dict(b=4.78)
    )


def test_gamma():
    nt.assert_dict_equal(
        process_args.gamma(k=1, theta=2),
        dict(a=1, scale=2)
    )
