paramnormal.utils.seed(0)
data = paramnormal.activity.random('lognormal', μ=0.75, σ=1.2, shape=125)
logdata = numpy.log10(data)
bins = numpy.logspace(logdata.min(), logdata.max(), num=30)
distplot_opts = dict(rug=True, kde=False, norm_hist=True, bins=bins)
ax = paramnormal.activity.plot('lognormal', data=data, distplot=True,
                               xscale='log', pad=0.01,
                               line_opts=line_opts,
                               distplot_opts=distplot_opts)
