import numpy
from matplotlib import pyplot

from paramnormal import dist, utils


def _check_distro(distro, **params):
    # check if we're returning the class definition or an instance
    as_class = params.pop("as_class", False)
    if hasattr(distro, "pdf"):
        return distro
    elif hasattr(distro, "from_params"):
        return _check_distro(distro.name, as_class=as_class, **params)
    else:
        try:
            distro = getattr(dist, distro)
        except (AttributeError, TypeError):
            raise ValueError(f"{distro} is not a valid paramnormal distribution")

    _params = utils._remove_nones(**params)
    if as_class:
        return distro
    else:
        return distro(**_params)


def _check_ax(ax):
    if ax is None:
        ax = pyplot.gca()
        fig = ax.figure
    else:
        fig = ax.figure

    return fig, ax


def random(distro, **params):
    """
    Generate random data from a probability distribution.

    Parameters
    ----------
    distro : str or paramnormal class
        The distribution from which the random data will be generated.
    params : keyword arguments of floats
        The parameters required to define the distribution. See each
        distribution's docstring for more info.
    shape : int or tuple of ints, optional
        The shape of the array into which the generated data will be
        placed. If `None`, a scalar will be retured.

    Returns
    -------
    random : numpy.array or scalar
        The random array (or scalar) generated.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal
    >>> numpy.random.seed(0)
    >>> # define dist with a string
    >>> paramnormal.activity.random('normal', mu=5, sigma=1.75, shape=1)
    array([ 8.08709161])
    >>> # or you can specify the actual class
    >>> paramnormal.activity.random(paramnormal.normal, mu=5, sigma=1.75, shape=None)
    5.700275114642641
    >>> # greek letters still work
    >>> paramnormal.activity.random('beta', α=2.5, β=1.2, shape=(3,3))
    array([[ 0.43771761,  0.84131634,  0.4390664 ],
           [ 0.7037142 ,  0.88282672,  0.09080825],
           [ 0.98747135,  0.63227551,  0.98108498]])

    """

    shape = params.pop("shape", None)
    distro = _check_distro(distro, **params)
    return distro.rvs(size=shape)


def fit(distro, data, as_params=True, **guesses):
    """
    Estimate the distribution parameters of sample data.

    Parameters
    ----------
    distro : str or paramnormal class
        The distribution from which the random data will be generated.
    data : array-like
        The data from which the distribution will be fit.
    guesses : named arguments of floats
        Inital guess values for certain parameters of the
        distribution. See the class docstring for more information
        on the parameters.

    Returns
    -------
    params : namedtuple
        A namedtuple containing all of the paramaters of the
        distribution.

    Examples
    --------
    >>> import numpy
    >>> import paramnormal
    >>> numpy.random.seed(0)
    >>> x = numpy.random.normal(loc=5.75, scale=2.25, size=37)
    >>> paramnormal.activity.fit('normal', x)
    params(mu=6.4790576880446782, sigma=2.4437818960405617)

    >>> paramnormal.activity.fit('normal', x, sigma=2)
    params(mu=6.4790576880446782, sigma=2)
    """

    distro = _check_distro(distro, as_class=True)
    params = distro.fit(data, **guesses)
    if as_params:
        return params
    else:
        return distro.from_params(params)


def plot(
    distro,
    which="PDF",
    data=None,
    fit_dist=True,
    ax=None,
    pad=0.05,
    xscale="linear",
    line_opts=None,
    **guesses,
):
    """
    Plot the PDF of a dataset and other representations of the
    distribution (histogram, kernel density estimate, and rug plot).

    Parameters
    ----------
    distro : str or distribution
        The (name of) the distribution to be plotted.
    data : array-like, optional
        An array-like object that can be passed to
        :func:`~seaborn.distplot` and :func:`~fit`.
    fit_dist : bool, optional
        Toggles fitting ``distro`` to ``data``. If False, ``distro``
        must be a fully specified distribution so that the PDF can be
        plotted.
    ax : matplotlib.Axes, optional
        Axes on which the everything gets drawn. If not provided, a new
        one is created.
    pad : float, optional
        The fraction of beyond min and max values of data where the PDF
        will be drawn.
    xscale : str, optional
        Specfifies a `'log'` or `'linear'` scale on the plot.
    line_opts : dict, optional
        Plotting options passed to :meth:`~ax.plot` when drawing the PDF.
    distplot : bool, Optional
        Toggles the use of :func:`~seaborn.distplot`. The default is
        `False`.

        .. note:
           ``data`` must not be `None` for this to have an effect.

    distplot_opts : dict, optional
        Dictionary of parameters to be passed to
        :func:`~seaborn.distplot`.
    guesses : keyword arguments, optional
        Additional parameters for specifying the distribution.

    Returns
    -------
    ax : matplotlib.Axes

    See Also
    --------
    seaborn.distplot

    Examples
    --------

    Plot a simple PDF of a fully-specified normal distribution.

    .. plot::
        :context: close-figs

        >>> import numpy
        >>> import seaborn
        >>> import paramnormal
        >>> clean_bkgd = {'axes.facecolor': 'none', 'figure.facecolor': 'none'}
        >>> seaborn.set(style='ticks', rc=clean_bkgd)
        >>> norm_dist = paramnormal.normal(μ=5.4, σ=2.5)
        >>> ax = paramnormal.activity.plot(norm_dist)

    Pass a data sample to fit the distribution on-the-fly.

    .. plot::
        :context: close-figs

        >>> paramnormal.utils.seed(0)
        >>> data = paramnormal.activity.random('normal', μ=5.4, σ=2.5, shape=(37))
        >>> ax = paramnormal.activity.plot('normal', data=data)

    Use seaborn to show other representations of the distribution of
    real data:

    .. plot::
        :context: close-figs

        >>> ax = paramnormal.activity.plot('normal', data=data, distplot=True)
        >>> ax.legend(loc='upper left')

    Use ``line_opts`` and ``distplot_opts`` to customize more complex
    plots.

    .. plot::
        :context: close-figs

        >>> paramnormal.utils.seed(0)
        >>> data = paramnormal.activity.random('lognormal', μ=0.75, σ=1.2, shape=125)
        >>> logdata = numpy.log10(data)
        >>> line_opts = dict(color='firebrick', lw=3.5, label='Fit PDF')
        >>> distplot_opts = dict(rug=True, kde=False, norm_hist=True)
        >>> ax = paramnormal.activity.plot('lognormal', data=data, distplot=True,
        ...                                xscale='log', pad=0.01,
        ...                                line_opts=line_opts,
        ...                                distplot_opts=distplot_opts)



    Notice that the bins in log-space don't work so well. We can
    compute them outselves.

    .. plot::
        :context: close-figs

        >>> paramnormal.utils.seed(0)
        >>> data = paramnormal.activity.random('lognormal', μ=0.75, σ=1.2, shape=125)
        >>> logdata = numpy.log10(data)
        >>> bins = numpy.logspace(logdata.min(), logdata.max(), num=30)
        >>> distplot_opts = dict(rug=True, kde=False, norm_hist=True, bins=bins)
        >>> ax = paramnormal.activity.plot('lognormal', data=data, distplot=True,
        ...                                xscale='log', pad=0.01,
        ...                                line_opts=line_opts,
        ...                                distplot_opts=distplot_opts)

    """

    # validate the axes and distribution function (`which`)
    fig, ax = _check_ax(ax)
    if data is not None:
        distro = fit(distro, data, as_params=False, **guesses)
    else:
        distro = _check_distro(distro, **guesses)
    fxn = getattr(distro, which.lower())

    # determine and set the xlimits of the plot
    xlimits = distro.ppf([pad / 100, 1 - pad / 100])

    # determine the x-values
    if xscale == "log":
        # xlimits = numpy.log10(xlimits)
        x_hat = numpy.logspace(*numpy.log10(xlimits), num=100)
    else:
        x_hat = numpy.linspace(*xlimits, num=100)

    # compute y-values
    y_hat = fxn(x_hat)

    line_opts = dict() if line_opts is None else line_opts
    line_opts["label"] = line_opts.pop("label", which)

    (line,) = ax.plot(x_hat, y_hat, **line_opts)
    ax.set_xscale(xscale)

    return ax
