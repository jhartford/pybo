"""
Wrapper around the covariance computation code for ES/EP.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np

# exported symbols
__all__ = ['get_cov']


# externally defined code for computing the covariance matrix. Note that X
# should be an (n,d) array, xstar and ell should be d-vectors, and K should be
# of size n+d^2+d+1.
cdef extern void computeCovMatrix(double *K, double *X, int n, int d,
                                  double *xstar, double *ell, double sf2,
                                  double sn2)

cdef extern void computeNewColumnCovMatrix(double *ret, double *X, int n, int d,
                                           double *xstar, double *ell,
                                           double sf2, double sn2,
                                           double *Xtest, int m)


def get_cov(double[::1, :] X,
            double[:] xstar,
            double[:] ell,
            double sf2,
            double sn2):

    # get the sizes we'll need
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int r = n + d*d + d + 1

    # create the output array.
    cdef double[::1, :] K = np.empty((r, r), order='F')

    # call the internal code.
    computeCovMatrix(&K[0,0], &X[0,0], n, d, &xstar[0], &ell[0], sf2, sn2)

    # this just makes sure we get returned as an array rather than as a typed
    # memory view.
    return np.asarray(K)


def get_crosscov(double[::1, :] Xtest,
                 double[::1, :] X,
                 double[:] xstar,
                 double[:] ell,
                 double sf2,
                 double sn2):

    # get the sizes we'll need
    cdef int m = Xtest.shape[0]
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int r = n + d + int(d*(d-1)/2) + d + 1

    # create the output array.
    cdef double[::1, :] K = np.empty((m, r), order='F')

    computeNewColumnCovMatrix(&K[0,0], &X[0,0], n, d, &xstar[0], &ell[0], sf2,
                              sn2, &Xtest[0,0], m)

    # this just makes sure we get returned as an array rather than as a typed
    # memory view.
    return np.asarray(K)
