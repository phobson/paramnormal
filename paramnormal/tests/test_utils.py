# -*- coding: utf-8 -*-
import numpy

import nose.tools as nt
from numpy.random import seed
import numpy.testing as nptest

from paramnormal import dist, utils


def test_greco_deco():
    d1 = dist.normal._process_args(mu=1, sigma=2)
    d2 = dist.normal._process_args(μ=1, σ=2)
    expected = dict(loc=1, scale=2)
    nt.assert_dict_equal(d1, expected)
    nt.assert_dict_equal(d2, expected)

    d1 = dist.beta._process_args(alpha=1, beta=2)
    d2 = dist.beta._process_args(α=1, β=2)
    expected = {'a': 1, 'b': 2, 'loc': 0, 'scale': 1}
    nt.assert_dict_equal(d1, expected)
    nt.assert_dict_equal(d2, expected)

    d1 = dist.gamma._process_args(k=1, theta=2)
    d2 = dist.gamma._process_args(k=1, θ=2)
    expected = {'a': 1, 'loc': 0, 'scale': 2}
    nt.assert_dict_equal(d1, expected)
    nt.assert_dict_equal(d2, expected)


def test__pop_none():
    expected_no_Nones = dict(a=1, b=2, c=3)
    expected_some_Nones = dict(a=1, c=3)
    expected_all_Nones = dict()

    nt.assert_dict_equal(utils._pop_none(a=1, b=2, c=3), expected_no_Nones)
    nt.assert_dict_equal(utils._pop_none(a=1, b=None, c=3), expected_some_Nones)
    nt.assert_dict_equal(utils._pop_none(a=None, b=None, c=None), expected_all_Nones)
