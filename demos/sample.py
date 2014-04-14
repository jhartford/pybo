import numpy as np
import matplotlib.pyplot as pl

import pygp
import pybo.models
import pybo.utils.ldsample as ldsample


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

    # get some points for plotting.
    xmin = model.bounds[0,0]
    xmax = model.bounds[0,1]
    x = np.linspace(xmin, xmax, 200)

    pl.gcf()
    pl.cla()
    pygp.gpplot(gp, xmin, xmax, draw=False)
    pl.plot(x, model.f(x[:, None]), 'r-')
    pl.axis('tight')
    pl.draw()
