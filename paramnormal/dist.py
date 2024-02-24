from collections import namedtuple

import numpy
from scipy import stats

from paramnormal import utils


class BaseDist_Mixin:
    def __new__(cls, **params):
        dist_params = cls._process_args(fit=False, **params)
        return cls.dist(**dist_params)

    @classmethod
    def _fit(cls, data, **guesses):
        args = utils._remove_nones(**cls._process_args(fit=True, **guesses))
        _sp_params = cls.dist.fit(data, **args)
        return _sp_params

    @classmethod
    def fit(cls, data, **guesses):
        """Fit a distribution to sample using scipy's maximum
        likelihood estimation methods.

        Parameters
        ----------
        data : array-like
            A sample whose distribution parameters will be estimated.
        guesses : named arguments of floats
            Inital guess values for certain parameters of the
            distribution. See the class docstring for more information
            on the parameters.

        Returns
        -------
        params : namedtuple
            A namedtuple containing all of the paramaters of the
            distribution.

        """

        return cls.param_template(*cls._fit(data, **guesses))

    @classmethod
    def from_params(cls, params):
        """Create a distribution from the ``namedtuple``
        result of the :meth:`~fit` method.

        Examples
        --------
        >>> import numpy
        >>> import paramnormal
        >>> # silly fake data
        >>> x = numpy.random.normal(size=37)
        >>> params = paramnormal.normal.fit(x)
        >>> dist = paramnormal.normal.from_params(params)

        """

        kwargs = dict(zip(params._fields, params))
        return cls(**kwargs)


class normal(BaseDist_Mixin):
    """
    Create and fit data to a normal distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    mu : float
        The expected value (mean) of the underlying normal distribution.
        Acts as the location parameter of the distribution.
    sigma : float
        The standard deviation of the underlying normal distribution.
        Also acts as the scale parameter of distribution.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.normal(mu=5, sigma=2).rvs(size=3)
    array([ 8.52810469,  5.80031442,  6.95747597])

    >>> # english names and greek symbols are interchangeable
    >>> numpy.random.seed(0)
    >>> pn.normal(μ=5, σ=2).rvs(size=3)
    array([ 8.52810469,  5.80031442,  6.95747597])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = numpy.random.normal(5, 2, size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.normal.fit(data)
    params(mu=5.6480512782619359, sigma=2.1722505742582769)

    >>> # estimate sigma when mu is fixed a known value:
    >>> pn.normal.fit(data, mu=4.75)
    params(mu=4.75, sigma=2.3505677305181645)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    https://en.wikipedia.org/wiki/normal_distribution

    See Also
    --------
    scipy.stats.norm
    numpy.random.normal

    """

    dist = stats.norm
    param_template = namedtuple("params", ["mu", "sigma"])
    name = "normal"

    @staticmethod
    @utils.greco_deco
    def _process_args(mu=None, sigma=None, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        return {loc_key: mu, scale_key: sigma}


class lognormal(BaseDist_Mixin):
    """
    Create and fit data to a lognormal distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `offset`
        is fixed at 0. Thus, only `mu` and `sigma` are estimated unless
        the `offset` is explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    mu : float
        The expected value (mean) of the underlying normal distribution.
        Acts as the scale parameter of the distribution.
    sigma : float
        The standard deviation of the underlying normal distribution.
        Also acts as the shape parameter of distribution.
    offset : float, optional
        The location parameter of the distribution. It's effectively
        the lower bound of the distribution. In other works, if you're
        investigating some quantity that cannot go below zero (e.g.,
        pollutant concentrations), leave this as the default (zero).

        .. note ::
           When fitting a lognormal distribution to a dataset, this will
           be fixed at its default value unless you explicitly set
           it to another value. Set it to `None` if wish that it be
           estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.lognormal(mu=5, sigma=2).rvs(size=3)
    array([ 5054.85624027,   330.40342795,  1050.97750604])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.lognormal(μ=5, σ=2).rvs(size=3)
    array([ 5054.85624027,   330.40342795,  1050.97750604])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = numpy.random.lognormal(5, 2, size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.lognormal.fit(data)
    params(mu=5.6480511731060181, sigma=2.172250571711877, offset=0)

    >>> # estimate sigma when mu is fixed a known value:
    >>> pn.lognormal.fit(data, mu=4.75)
    params(mu=4.75, sigma=2.3505859375000036, offset=0)

    >>> # include `offset` in the estimate
    >>> pn.lognormal.fit(data, offset=None)
    params(mu=5.6538159643068386, sigma=2.1596452081058795, offset=-0.12039282461824304)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    https://en.wikipedia.org/wiki/lognormal_distribution

    See Also
    --------
    scipy.stats.lognorm
    numpy.random.lognormal

    """

    dist = stats.lognorm
    param_template = namedtuple("params", ["mu", "sigma", "offset"])
    name = "lognormal"

    @staticmethod
    @utils.greco_deco
    def _process_args(mu=None, sigma=None, offset=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        key = "f0" if fit else "s"
        if offset is None and not fit:
            raise ValueError("`offset` parameter is required. Recommended value is 0.")
        return {key: sigma, scale_key: numpy.exp(mu) if mu is not None else mu, loc_key: offset}

    @classmethod
    def fit(cls, data, **guesses):
        params = cls._fit(data, **guesses)
        return cls.param_template(mu=numpy.log(params[2]), sigma=params[0], offset=params[1])


class weibull(BaseDist_Mixin):
    """
    Create and fit data to a weibull distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only `k`
        is estimated unless `loc` or `scale` are explicitly set to
        `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    k : float
        The shape parameter of the distribution.

        .. note ::
           Strictly speaking, the weibull distribution has a second
           shape parameter, lambda. However, it seems to be always
           set to 1. So much so that scipy doesn't give you any other
           option.

    loc, scale : floats, optional
        Location and scale parameters of the distribution. These
        default to, and should probably be left at, 0 and 1,
        respectively.

        .. note ::
           When fitting a weibull distribution to a dataset, these will
           be fixed at their default values unless you explicitly set
           them to other values. Set them to `None` if you wish that
           they be estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.weibull(k=5).rvs(size=3)
    array([ 0.9553641 ,  1.04662991,  0.98415009])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = numpy.random.weibull(5, size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.weibull.fit(data)
    params(k=5.4158203125000091, loc=0, scale=1)

    >>> # include `loc` and `scale` in the estimate
    >>> pn.weibull.fit(data, loc=None, scale=None)
    params(k=14.120107702486127, loc=-1.389856535577052, scale=2.4320324339845572)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html
    https://en.wikipedia.org/wiki/weibull_distribution

    See Also
    --------
    scipy.stats.weibull_min
    scipy.stats.frechet_min
    numpy.random.weibull

    """

    dist = stats.weibull_min
    param_template = namedtuple("params", ["k", "loc", "scale"])
    name = "weibull"

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        key = "f0" if fit else "c"
        return {key: k, loc_key: loc, scale_key: scale}


class alpha(BaseDist_Mixin):
    """
    Create and fit data to a alpha distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only
        `alpha` is estimated unless `loc` or `scale` are explicitly set
        to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    alpha : float
        The shape parameter of the distribution.

    loc, scale : floats, optional
        Location and scale parameters of the distribution. These
        default to, and should probably be left at, 0 and 1,
        respectively.

        .. note ::
           When fitting a alpha distribution to a dataset, these will
           be fixed at their default values unless you explicitly set
           them to other values. Set them to `None` if you wish that
           they be estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> from scipy import stats
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.alpha(alpha=5).rvs(size=3)
    array([ 0.20502995,  0.22566277,  0.21099298])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.alpha(α=5).rvs(size=3)
    array([ 0.20502995,  0.22566277,  0.21099298])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = stats.alpha.rvs(5, size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.alpha.fit(data)
    params(alpha=4.8356445312500096, loc=0, scale=1)

    >>> # include `loc` and `scale` in the estimate
    >>> pn.alpha.fit(data, loc=None, scale=None)
    params(alpha=8.6781299501492342, loc=-0.15002784429644306, scale=3.1262971852456447)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html

    See Also
    --------
    scipy.stats.alpha

    """

    dist = stats.alpha
    param_template = namedtuple("params", ["alpha", "loc", "scale"])
    name = "alpha"

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        alpha_key = "f0" if fit else "a"
        return {alpha_key: alpha, loc_key: loc, scale_key: scale}


class beta(BaseDist_Mixin):
    """
    Create and fit data to a beta distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only
        `alpha` and `beta` are estimated unless `loc` or `scale` are
        explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    alpha, beta : float
        The (positive) shape parameters of the distribution.
    loc, scale : floats, optional
        Location and scale parameters of the distribution. These
        default to, and should probably be left at, 0 and 1,
        respectively.

        .. note ::
           When fitting a beta distribution to a dataset, these will
           be fixed at their default values unless you explicitly set
           them to other values. Set them to `None` if you wish that
           they be estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.beta(alpha=2, beta=5).rvs(size=3)
    array([ 0.47917138,  0.6550558 ,  0.21501632])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.beta(α=2, β=5).rvs(size=3)
    array([ 0.47917138,  0.6550558 ,  0.21501632])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.beta(alpha=2, beta=5).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.beta.fit(data)
    params(alpha=1.6784891179355115, beta=4.2459121691279398, loc=0, scale=1)

    >>> # just estimate beta with a known alpha
    >>> pn.beta.fit(data, alpha=2)
    params(alpha=2, beta=4.9699264393421139, loc=0, scale=1)

    >>> # include `loc` and `scale` in the estimate
    >>> pn.beta.fit(data, loc=None, scale=None)
    params(
        alpha=1.8111139255547926,
        beta=4.6972775768688697,
        loc=-0.0054013993799938431,
        scale=1.0388376932132561
    )

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    https://en.wikipedia.org/wiki/beta_distribution

    See Also
    --------
    scipy.stats.beta
    numpy.random.beta

    """

    dist = stats.beta
    param_template = namedtuple("params", ["alpha", "beta", "loc", "scale"])
    name = "beta"

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, beta=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        alpha_key = "f0" if fit else "a"
        beta_key = "f1" if fit else "b"
        return {alpha_key: alpha, beta_key: beta, loc_key: loc, scale_key: scale}


class gamma(BaseDist_Mixin):
    """
    Create and fit data to a gamma distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only
        `alpha` and `beta` are estimated unless `loc` or `scale` are
        explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    k, theta : float
        The shape and scale parameters of the distribution,
        respectively.
    loc : float, optional
        Location parameter of the distribution. This defaults to, and
        should probably be left at, 0.

        .. note ::
           When fitting a beta distribution to a dataset, this will
           be fixed at its default value unless you explicitly set
           it to other values. Set to `None` if you wish that it be
           estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.gamma(k=2, theta=5).rvs(size=3)
    array([ 25.69414788,  11.19240456,  27.13566137])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.gamma(k=2, θ=5).rvs(size=3)
    array([ 25.69414788,  11.19240456,  27.13566137])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.gamma(k=2, θ=5).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.gamma.fit(data)
    params(k=1.3379069223213471, loc=0, theta=7.5830062081633622)

    >>> # just estimate theta with a known k
    >>> pn.gamma.fit(data, theta=5)
    params(k=1.8060453251225814, loc=0, theta=5)

    >>> # include `loc` in the estimate
    >>> pn.gamma.fit(data, loc=None)
    params(k=1.0996117768860174, loc=0.29914735266576881, theta=8.9542450315590756)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    https://en.wikipedia.org/wiki/gamma_distribution

    See Also
    --------
    scipy.stats.gamma
    numpy.random.gamma

    """

    dist = stats.gamma
    param_template = namedtuple("params", ["k", "loc", "theta"])
    name = "gamma"

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, theta=None, loc=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        key = "f0" if fit else "a"
        return {key: k, loc_key: loc, scale_key: theta}


class chi_squared(BaseDist_Mixin):
    """
    Create and fit data to a chi-squared distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only
        `alpha` and `beta` are estimated unless `loc` or `scale` are
        explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    k : float
        The degrees of freedom of the distribution,
        respectively.
    loc, scale : floats, optional
        Location and scale parameters of the distribution. These
        default to, and should probably be left at, 0 and 1,
        respectively.

        .. note ::
           When fitting a chi-squared distribution to a dataset, these
           will be fixed at their default value unless you explicitly
           set them to other values. Set to `None` if you wish that they
           be estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.chi_squared(k=2).rvs(size=3)
    array([ 1.59174902,  2.51186153,  1.84644629])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.chi_squared(k=2).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.chi_squared.fit(data)
    params(k=2.2668945312500028, loc=0, scale=1)

    >>> # include `loc` in the estimate
    >>> pn.chi_squared.fit(data, loc=None)
    params(k=1.9361813889429524, loc=0.037937143324767775, scale=1)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    https://en.wikipedia.org/wiki/Chi-squared_distribution

    See Also
    --------
    scipy.stats.chi2
    numpy.random.chisquare

    """

    dist = stats.chi2
    param_template = namedtuple("params", ["k", "loc", "scale"])
    nane = "chi_squared"

    @staticmethod
    @utils.greco_deco
    def _process_args(k=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        key = "f0" if fit else "df"
        return {key: k, loc_key: loc, scale_key: 1}


class pareto(BaseDist_Mixin):
    """
    Create and fit data to a pareto distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        and `scale` are fixed at 0 and 1, respectively. Thus, only
        `alpha` is estimated unless `loc` or `scale` are explicitly
        set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    alpha : float
        The shape parameter of the distribution.
    loc, scale : floats, optional
        Location and scale parameters of the distribution. These
        default to, and should probably be left at, 0 and 1,
        respectively.

        .. note ::
           When fitting a pareto distribution to a dataset, this will
           be fixed at its default value unless you explicitly set
           it to other values. Set to `None` if you wish that it be
           estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.pareto(alpha=2).rvs(size=3)
    array([ 1.48875061,  1.87379424,  1.58662889])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.pareto(α=2).rvs(size=3)
    array([ 1.48875061,  1.87379424,  1.58662889])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.pareto(alpha=2).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.pareto.fit(data)
    params(alpha=1.7850585937500019, loc=0, scale=1)

    >>> # include `loc` in the estimate
    >>> pn.pareto.fit(data, loc=None)
    params(alpha=1.8040853559635659, loc=0.009529403810858695, scale=1)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html
    https://en.wikipedia.org/wiki/pareto_distribution

    See Also
    --------
    scipy.stats.pareto
    numpy.random.pareto

    """

    dist = stats.pareto
    param_template = namedtuple("params", ["alpha", "loc", "scale"])
    name = "pareto"

    @staticmethod
    @utils.greco_deco
    def _process_args(alpha=None, loc=0, scale=1, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        key = "f0" if fit else "b"
        return {key: alpha, loc_key: loc, scale_key: scale}


class exponential(BaseDist_Mixin):
    """
    Create and fit data to an exponential distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        is fixed at 0. Thus, only `lamda` is estimated unless `loc` is
        explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    lamda : float
        The shape parameter of the distribution.

        .. note ::
           For our purposes, we spell `lambda` as `lamda` to avoid
           conflicting with the python keyword ``lambda``.

    loc : float, optional
        Location parameter of the distribution. This defaults to, and
        should probably be left at, 0.

        .. note ::
           When fitting an exponential distribution to a dataset, this
           will be fixed at its default value unless you explicitly set
           it to other values. Set to `None` if you wish that it be
           estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.exponential(lamda=2).rvs(size=3)
    array([ 0.39793725,  0.62796538,  0.46161157])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.exponential(λ=2).rvs(size=3)
    array([ 0.39793725,  0.62796538,  0.46161157])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.exponential(λ=2).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.exponential.fit(data)
    params(lamda=1.7849050026146085, loc=0)

    >>> # include `loc` in the estimate
    >>> pn.exponential.fit(data, loc=None)
    params(lamda=1.8154701618164411, loc=0.0094842718426853996)

    References
    ----------
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    https://en.wikipedia.org/wiki/exponential_distribution

    See Also
    --------
    scipy.stats.expon
    numpy.random.exponential

    """

    dist = stats.expon
    param_template = namedtuple("params", ["lamda", "loc"])
    name = "exponential"

    @staticmethod
    @utils.greco_deco
    def _process_args(lamda=None, loc=0, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        return {loc_key: loc, scale_key: lamda**-1 if lamda is not None else lamda}

    @classmethod
    def fit(cls, data, **guesses):
        params = cls._fit(data, **guesses)
        return cls.param_template(loc=params[0], lamda=params[1] ** -1)


class rice(BaseDist_Mixin):
    """
    Create and fit data to a Rice distribution.

    Methods
    -------
    fit
        Use scipy's maximum likelihood estimation methods to estimate
        the parameters of the data's distribution. By default, `loc`
        is fixed at 0. Thus, only `R` and `sigma` are estimated unless
        `loc` is explicitly set to `None`.
    from_params
        Create a new distribution instances from the ``namedtuple``
        result of the :meth:`~fit` method.

    Parameters
    ----------
    R : float
        The shape parameter of the distribution.
    sigma : float
        The standard deviate of the distribution.
    loc : float, optional
        Location parameter of the distribution. This defaults to, and
        should probably be left at, 0.

        .. note ::
           When fitting an Rice distribution to a dataset, this
           will be fixed at its default value unless you explicitly set
           it to other values. Set to `None` if you wish that it be
           estimated entirely from scratch.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal as pn
    >>> numpy.random.seed(0)
    >>> pn.rice(R=10, sigma=2).rvs(size=3)
    array([ 15.67835764,  13.36907874,  10.37753817])

    >>> # you can also use greek letters
    >>> numpy.random.seed(0)
    >>> pn.rice(R=10, σ=2).rvs(size=3)
    array([ 15.67835764,  13.36907874,  10.37753817])

    >>> # silly fake data
    >>> numpy.random.seed(0)
    >>> data = pn.rice(R=10, sigma=2).rvs(size=37)
    >>> # pretend `data` is unknown and we want to fit a dist. to it
    >>> pn.rice.fit(data)
    params(R=10.100674084593422, sigma=1.759817171541185, loc=0)

    >>> # include `loc` in the estimate (bad idea)
    >>> pn.rice.fit(data, loc=None)
    params(R=4.249154300734, sigma=1.862167512728, loc=5.570921659394)

    References
    ----------
    http://scipy.github.io/devdocs/generated/scipy.stats.rice
    https://en.wikipedia.org/wiki/Rice_distribution

    See Also
    --------
    scipy.stats.rice
    numpy.random.exponential

    """

    dist = stats.rice
    param_template = namedtuple("params", ["R", "sigma", "loc"])
    name = "rice"

    @staticmethod
    @utils.greco_deco
    def _process_args(R=None, loc=0, sigma=None, fit=False):
        loc_key, scale_key = utils._get_loc_scale_keys(fit=fit)
        bkey = "fb" if fit else "b"

        b = None
        if R is not None and sigma is not None:
            b = R / sigma
        return {loc_key: loc, scale_key: sigma, bkey: b}

    @classmethod
    def fit(cls, data, **guesses):
        b, loc, sigma = cls._fit(data, **guesses)
        return cls.param_template(R=b * sigma, loc=loc, sigma=sigma)


__all__ = [
    "normal",
    "lognormal",
    "weibull",
    "alpha",
    "beta",
    "gamma",
    "chi_squared",
    "pareto",
    "exponential",
    "rice",
]
