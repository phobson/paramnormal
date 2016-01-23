import numpy
import seaborn
import paramnormal
clean_bkgd = {'axes.facecolor': 'none', 'figure.facecolor': 'none'}
seaborn.set(style='ticks', rc=clean_bkgd)
norm_dist = paramnormal.normal(μ=5.4, σ=2.5)
ax = paramnormal.activity.plot(norm_dist)
