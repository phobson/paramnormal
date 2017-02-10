paramnormal: Conventionally parameterized probability distributions
===================================================================
.. image:: https://travis-ci.org/phobson/paramnormal.svg?branch=master
    :target: https://travis-ci.org/phobson/paramnormal

.. image:: https://coveralls.io/repos/phobson/paramnormal/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/phobson/paramnormal?branch=master


Scipy distributions are weird.
Maybe these will be less weird.


The problem this is solving
---------------------------

Let's look at the lognormal distribution.
The `wikipedia article <https://en.wikipedia.org/wiki/Log-normal_distribution>`__ states that they are parameterized by μ and σ, the mean and standard deviation of the underlying normal distribution.
In this case, μ and σ can also be known as the location and scale parameters, respectively.

However, to create a lognormal distribution in scipy, you need three parameters: location, scale, and shape.
The tricky part is, however, is that "location" in scipy refers to an offset from zero, "shape" refers to σ, and the "scale" refers to :math:`e^\mu`.
This is all explained the scipy documentation, but it took me a couple of readings and bad mistakes to figure it out.
It's also never really explicitly stated that scipy's location parameter should be zero 99.999% of the time.
That's a very import point to understand when you're fitting lognormal parameters to a sample and you end with three crazy numbers that don't make any sense and distribution with values less than zero despite the fact that you thought lognormal distribution couldn't have values less than zero.

Point of all of this is that *paramnormal* is trying to make easy what scipy sometimes makes tricky.

So where as in scipy, you would do this:

.. code:: python

    import numpy
    from scipy import stats
    mu = 0.75
    sigma = 1.25
    dist = stats.lognorm(sigma, loc=0, scale=numpy.exp(mu))

In paramnormal, you can do this:

.. code:: python

    import paramnormal
    dist = paramnormal.lognormal(mu=0.75, sigma=1.25)

You can even use Greek letters

.. code:: python

    dist = paramnormal.lognormal(μ=0.75, σ=1.25)

All three snippets return the same scipy distribution objects and have the same numerical methods (e.g., ``cdf``, ``pdf``, ``rvs``, ``ppf``).
Paramnormal just provides a short cut that let's you only specify the traditional distribution parameters you read about in your text book.

Documentation
-------------
We have `HTML docs built with sphinx <http://phobson.github.io/paramnormal/>`_.

Installation
------------
Binaries are available through my conda channel

``conda install --channel=phobson paramnormal``

This is a pure python package, so installation from source should be as easy as running
``pip install .`` from the source directory if you've cloned the repo.

Otherwise, I think ``pip install git+https://github.com/phobson/paramnormal.git`` will work.
(I'll upload to pip after this has sat around for a while.

Development status
------------------
From my perspective this is now feature complete, meaning it has all of the distribution that I use somewhat regularly.
If you want to add a new distribution, get in touch.
Otherwise, I'll just be fixing bugs and typos/ommisions in the documentaion.
