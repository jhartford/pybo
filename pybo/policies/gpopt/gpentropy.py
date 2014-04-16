"""
Policy for solving via "entropy search".
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.stats as ss
import scipy.linalg as sla

# local imports
from ._escov import get_cov

# exported symbols
__all__ = []


def run_ep_iteration(mu, Sigma, ymin, sn2):
    # XXX: compute the inverse. first off: can we avoid calling this? second: if
    # we really do need to call this then we should be using dpotri (or a scipy
    # call to that LAPACK function).
    SigmaFactor = sla.cho_factor(Sigma)
    SigmaInverse = sla.cho_solve(SigmaFactor, np.eye(Sigma.shape[0]))
    muSigma = sla.cho_solve(SigmaFactor, mu)

    # the marginal approximation to our posterior given the current approximate
    # factors.
    tau = 1 / Sigma.diagonal()
    rho = mu / Sigma.diagonal()

    # the current approximate factors.
    tauHat = np.zeros_like(tau)
    rhoHat = np.zeros_like(rho)

    # we won't do any damping at first.
    damping = 1

    while True:
        # eliminate the contribution of the approximate factor.
        v = (tau - tauHat) ** -1
        m = v * (rho - rhoHat)

        s = np.sqrt(v);      s[-1] = np.sqrt(v[-1] + sn2)
        t = m.copy();        t[-1] = ymin - m[-1]
        u = np.ones_like(t); u[-1] = -1

        alpha = t / s
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        beta = ratio * (alpha + ratio) / s / s
        kappa = u * (t / s + ratio) / s

        tauHatNew = beta / (1 - beta*v)
        tauHatNew[np.abs(tauHatNew) < 1e-300] = 1e-300
        rhoHatNew = (m + 1 / kappa) * tauHatNew

        # don't change anything that ends up with a negative variance.
        negv = (v < 0)
        tauHatNew[negv] = tauHat[negv]
        rhoHatNew[negv] = rhoHat[negv]

        while True:
            # mix between the new factors and the old ones. NOTE: in the first
            # iteration damping is 1, so this doesn't do any damping.
            tauHatNew = tauHatNew * damping + tauHat * (1-damping)
            rhoHatNew = rhoHatNew * damping + rhoHat * (1-damping)

            # get the eigenvalues of the new posterior covariance and mix more
            # with the old approximation if they're blowing up.
            vals, _ = np.linalg.eig(np.diag(tauHatNew) + SigmaInverse)

            if any(1/vals <= 1e-10):
                damping *= 0.5
            else:
                break

        # our new approximate factors.
        tauHat = tauHatNew
        rhoHat = rhoHatNew

        # the new posterior.
        V = sla.cho_solve(sla.cho_factor(np.diag(tauHat) + SigmaInverse), np.eye(Sigma.shape[0]))
        m = np.dot(V, rhoHat + muSigma)

        if np.max(np.abs(np.r_[V.diagonal() - 1/tau, m - rho/tau])) >= 1e-6:
            tau = 1 / V.diagonal()
            rho = m / V.diagonal()
            damping *= 0.99
        else:
            break

    vHat = 1 / tauHat
    mHat = rhoHat / tauHat

    return mHat, vHat


def run_ep(X, y, xstar, ell, sf2, sn2):
    # make sure our input array is in fortran order so that we can pass it to
    # Miguel's get_cov method.
    X = np.asfortranarray(X)
    d = X.shape[1]

    # our "observations". first we have the actual observations y, and then the
    # equality constraints of zero for the gradient and the non-diagonal
    # elements of the Hessian evaluated at xstar.
    b = np.r_[y, np.zeros(d + d*(d-1)/2)]
    c = len(b)

    # get the full covariance between the inputs as well as the gradient and
    # Hessian evaluated at xstar. This is blocked such that our constraints
    # (i.e. b) are the first diagonal block and a is the second.
    K = get_cov(X, xstar, ell, sf2, sn2)
    Ka = K[c:, c:]
    Kb = K[:c, :c]
    Kba = K[:c, c:]

    # FIXME: document get_cov better. Also just by naming convention wouldn't it
    # make more sense if a was the first block?

    # these temporaries are just the cholesky of the covariance of b and the
    # forward-solve of this with Kba which we can use to do the conditioning.
    R = sla.cholesky(Kb)
    V = sla.solve_triangular(R, Kba, trans=True)

    # This then results in mu and Sigma being the Gaussian prior we'll hand off
    # to EP.
    mu = np.dot(V.T, sla.solve_triangular(R, b, trans=True))
    Sigma = Ka - np.dot(V.T, V)

    # run EP to get approximate factors for a and update the factors so we can
    # apply them to the full joint.
    m, v = run_ep_iteration(mu, Sigma, min(y), sn2)
    m = np.r_[b, m]
    v = np.r_[np.zeros_like(b), v]

    # NOTE: this replaces the earlier computed cholesky.
    R = sla.cholesky(K + np.diag(v))
    alpha = sla.solve_triangular(R, m, trans=True)

    return R, alpha
