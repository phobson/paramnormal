paramnormal.utils.seed(0)
data = paramnormal.activity.random('lognormal', μ=0.75, σ=1.2, shape=125)
logdata = numpy.log10(data)
line_opts = dict(color='firebrick', lw=3.5, label='Fit PDF')
distplot_opts = dict(rug=True, kde=False, norm_hist=True)
ax = paramnormal.activity.plot('lognormal', data=data, distplot=True,
                               xscale='log', pad=0.01,
                               line_opts=line_opts,
                               distplot_opts=distplot_opts)
