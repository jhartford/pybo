"""
Class facilitating grid generation and memoization.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# local imports
from mwhutils import random
from .lbfgs import solve_lbfgs
from ..utils import params

# exported symbols
__all__ = ['solve_spray']


@params('ngrid', 'nbest', 'nspray')
class solve_spray(object):
    def __init__(self, bounds=[[0, 1]], ngrid=20000, nbest=10, nspray=10, rng=None):
        self.nbest = nbest
        self.ngrid = ngrid
        self.nspray = nspray
        self.rng = rng

        self.bounds = np.array(bounds, ndmin=2)
        self.grid = random.sobol(bounds, ngrid, rng)

    def __call__(self, f, bounds, data=None):
        """
        Compute the objective function on an initial grid, pick `nbest` points,
        and maximize using LBFGS from these initial points.

        The initial grid is initialized to a Sobol grid and augmented with data
        as well as randomly distributed points around the best observation.

        Args:
            f: function handle that takes an optional `grad` boolean kwarg
               and if `grad=True` returns a tuple of `(function, gradient)`.
               NOTE: this functions is assumed to allow for multiple inputs in
               vectorized form.

            bounds: bounds of the search space.
            data: tuple (X, y) used to sprinkle points around the best
                  observation so far.

        Returns:
            xmin, fmax: location and value of the maximizer.
        """

        bounds = np.array(bounds, ndmin=2)
        if not np.allclose(bounds, self.bounds):
            self.bounds = bounds
            self.grid = random.sobol(bounds, self.ngrid, self.rng)

        if data is not None:
            X, y = data
            width = 0.001 * bounds[:,1] - bounds[:,0]
            xbest = X[y.argmax()]
            xspray = xbest + width[:] * np.random.randn(self.nspray, X.shape[1])
            xgrid = np.r_[self.grid, X, xspray]
        else:
            xgrid = self.grid

        return solve_lbfgs(f, bounds, xgrid=xgrid)
