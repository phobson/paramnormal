paramnormal.utils.seed(0)
data = paramnormal.activity.random('normal', μ=5.4, σ=2.5, shape=(37))
ax = paramnormal.activity.plot('normal', data=data)
