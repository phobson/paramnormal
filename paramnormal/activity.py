from __future__ import annotations

from typing import Any, Literal, TypeAlias, overload

import numpy
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from scipy.stats.distributions import rv_frozen

from paramnormal import dist, utils
from paramnormal.types import FloatArray, NDArrayLike, ParamsLike

Distro: TypeAlias = str | type[dist.BaseDist_Mixin] | rv_frozen


def _get_class(distro: Distro) -> type[dist.BaseDist_Mixin]:
    if isinstance(distro, type) and issubclass(distro, dist.BaseDist_Mixin):
        return distro
    if isinstance(distro, str):
        cls = getattr(dist, distro, None)
        if isinstance(cls, type) and issubclass(cls, dist.BaseDist_Mixin):
            return cls
    raise ValueError(f"{distro} is not a valid paramnormal distribution")


def _check_distro(distro: Distro, **params: float | None) -> rv_frozen:
    # frozen scipy distributions pass straight through
    if isinstance(distro, rv_frozen):
        return distro
    return _get_class(distro)(**utils._remove_nones(**params))


def _check_ax(ax: Axes | None) -> tuple[Figure | SubFigure, Axes]:
    if ax is None:
        ax = pyplot.gca()
        fig = ax.figure
    else:
        fig = ax.figure

    return fig, ax


def random(
    distro: Distro, *, shape: int | tuple[int, ...] | None = None, **params: float | None
) -> float | FloatArray:
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

    distro = _check_distro(distro, **params)
    return distro.rvs(size=shape)


@overload
def fit(
    distro: Distro, data: NDArrayLike, as_params: Literal[True] = True, **guesses: float | None
) -> ParamsLike: ...
@overload
def fit(
    distro: Distro, data: NDArrayLike, as_params: Literal[False], **guesses: float | None
) -> rv_frozen: ...
def fit(
    distro: Distro, data: NDArrayLike, as_params: bool = True, **guesses: float | None
) -> ParamsLike | rv_frozen:
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

    cls = _get_class(distro)
    params = cls.fit(data, **guesses)
    if as_params:
        return params
    else:
        return cls.from_params(params)


def _draw_distplot(sample: FloatArray, ax: Axes, opts: dict[str, Any] | None) -> None:
    """Histogram, KDE and rug plot of ``sample``.

    Replaces the deprecated ``seaborn.distplot``, accepting its common options:
    ``hist``, ``kde``, ``rug``, ``norm_hist``, ``bins``, ``color``, and the
    ``hist_kws``, ``kde_kws`` and ``rug_kws`` dictionaries.
    """
    try:
        import seaborn
    except ImportError as err:
        raise ImportError(
            "`seaborn` is required for `distplot=True`; install `paramnormal[plot]`"
        ) from err

    o = dict(opts or {})
    hist = o.pop("hist", True)
    kde = o.pop("kde", True)
    rug = o.pop("rug", False)
    norm_hist = o.pop("norm_hist", False)
    bins = o.pop("bins", "auto")
    color = o.pop("color", None)
    hist_kws = dict(o.pop("hist_kws", None) or {})
    kde_kws = dict(o.pop("kde_kws", None) or {})
    rug_kws = dict(o.pop("rug_kws", None) or {})
    if o:
        raise TypeError(f"unsupported `distplot_opts`: {sorted(o)}")

    if color is not None:
        for kws in (hist_kws, kde_kws, rug_kws):
            kws.setdefault("color", color)

    if hist:
        stat = "density" if (norm_hist or kde) else "count"
        seaborn.histplot(x=sample, bins=bins, stat=stat, ax=ax, **hist_kws)
    if kde:
        seaborn.kdeplot(x=sample, ax=ax, **kde_kws)
    if rug:
        seaborn.rugplot(x=sample, ax=ax, **rug_kws)


def plot(
    distro: Distro,
    which: str = "PDF",
    data: NDArrayLike | None = None,
    fit_dist: bool = True,
    ax: Axes | None = None,
    pad: float = 0.05,
    xscale: Literal["linear", "log"] = "linear",
    line_opts: dict[str, str | float] | None = None,
    distplot: bool = False,
    distplot_opts: dict[str, Any] | None = None,
    **guesses: float | None,
) -> Axes:
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

    fig, ax = _check_ax(ax)
    sample = None if data is None else numpy.asarray(data, dtype=float)
    if sample is not None and fit_dist:
        frozen = fit(distro, sample, as_params=False, **guesses)
    else:
        frozen = _check_distro(distro, **guesses)
        if any(v is None for v in (*frozen.args, *frozen.kwds.values())):
            raise ValueError("`distro` must be fully specified when it is not being fit to `data`")
    fxn = getattr(frozen, which.lower())

    if distplot and sample is not None:
        _draw_distplot(sample, ax, distplot_opts)

    # determine the x-values from the distribution's quantiles
    lo, hi = (float(v) for v in frozen.ppf([pad / 100, 1 - pad / 100]))
    if xscale == "log":
        x_hat = numpy.logspace(numpy.log10(lo), numpy.log10(hi), num=100)
    else:
        x_hat = numpy.linspace(lo, hi, num=100)

    y_hat = fxn(x_hat)

    # copy so the caller's dict is never modified
    opts = dict(line_opts or {})
    opts.setdefault("label", which)

    ax.plot(x_hat, y_hat, **opts)  # ty: ignore[invalid-argument-type]
    ax.set_xscale(xscale)

    return ax
