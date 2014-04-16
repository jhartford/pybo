import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo.models
import pybo.utils.ldsample as ldsample
import pybo.policies.gpopt.gpentropy as gpentropy


if __name__ == '__main__':
    sn = 0.25
    ell = 0.65
    sf = 1.0

    # sample a function from the GP prior and use that as a model. Note that
    # since we haven't added data, the noise term used by the GP model has no
    # affect whatsoever.
    gp = pygp.BasicGP(sn, ell, sf)
    model = pybo.models.GPModel(gp, [0, 10], sigma=sn, rng=0)

    np.random.seed(0)
    X = ldsample.random(model.bounds, 5)
    y = model.get_all(X)
    gp.add_data(X, y)

    xstar = np.array([2.])
    ell = np.array([ell])

    # get some points for plotting.
    xmin = model.bounds[0,0]
    xmax = model.bounds[0,1]
    x = np.linspace(xmin, xmax, 200)

    # get the mean/variance before conditioning
    muB, s2B = gp.posterior(x[:,None])
    erB = 3*np.sqrt(s2B)

    # after conditioning on the minimum
    muA, s2A = gpentropy.predict_ep(x[:,None], X, y, xstar, ell, sf**2, sn**2)
    erA = 3*np.sqrt(s2A)

    pl.gcf()
    pl.subplot(211)
    pl.cla()
    pl.fill_between(x, muB-erB, muB+erB, color='k', alpha=0.1)
    pl.plot(x, muB, lw=2, color='k')
    pl.scatter(X, y, s=40)
    pl.title('marginal posterior')
    pl.axis('tight')
    pl.axis(xmin=xmin, xmax=xmax)

    axis = pl.axis()

    pl.subplot(212)
    pl.cla()
    pl.fill_between(x, muA-erA, muA+erA, color='k', alpha=0.1)
    pl.plot(x, muA, lw=2, color='k')
    pl.scatter(X, y, s=40)
    pl.axvline(xstar, color='r')
    pl.title('marginal posterior conditioned on the marked minimum')
    pl.axis('tight')
    pl.axis(axis)
    pl.draw()
