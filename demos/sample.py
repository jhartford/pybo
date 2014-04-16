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

    R, alpha = gpentropy.run_ep(X, y, xstar, ell, sf**2, sn**2)
    print R
    print alpha

    pl.gcf()
    pl.cla()
    pygp.gpplot(gp, xmin, xmax, draw=False)
    pl.plot(x, model.f(x[:, None]), 'g-', lw=2)
    pl.axvline(xstar, color='r')
    pl.axis('tight')
    pl.axis(xmin=xmin, xmax=xmax)
    pl.draw()
