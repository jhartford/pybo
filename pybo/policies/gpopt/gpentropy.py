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
    v = Sigma.diagonal().copy()
    m = mu.copy()

    # the current approximate factors.
    vHat = np.ones_like(v) * np.inf
    mHat = np.zeros_like(m)

    # we won't do any damping at first.
    damping = 1

    while True:
        # eliminate the contribution of the approximate factor.
        vOld = (v**-1 - vHat**-1) ** -1
        mOld = vOld * (m/v - mHat/vHat)
        negv = (vOld < 0)

        # introduce aux versions of m and v because the updates are *almost* the
        # same for the two parts if we do this transformation first.
        vAux = vOld.copy(); vAux[-1] = vOld[-1] + sn2
        mAux = mOld.copy(); mAux[-1] = ymin - mOld[-1]
        sAux = np.sqrt(vAux)

        alpha = mAux / sAux
        ratio = np.exp(ss.norm.logpdf(alpha) - ss.norm.logcdf(alpha))
        beta = ratio * (alpha + ratio) / vAux

        # note that for the evaluation part the kappa gets flipped as well.
        kappa = (mAux / sAux + ratio) / sAux
        kappa[-1] *= -1

        vHatNew = 1 / beta - vOld
        mHatNew = mOld + 1 / kappa

        # don't change anything that ends up with a negative variance.
        vHatNew[negv] = vHat[negv]
        mHatNew[negv] = mHat[negv]

        while True:
            mHatNew = mHatNew / vHatNew * damping + mHat / vHat * (1 - damping)
            vHatNew = 1 / (1 / vHatNew * damping + 1 / vHat * (1 - damping))
            mHatNew = vHatNew * mHatNew

            # XXX: this seems expensive. anything better?
            vals, _ = np.linalg.eig(np.diag(vHatNew**-1) + SigmaInverse)

            if any(1/vals <= 1e-10):
                damping *= 0.5
            else:
                break

        # our new approximate factors.
        vHat = vHatNew
        mHat = mHatNew

        # the new posterior marginals.
        VNew = sla.cho_solve(sla.cho_factor(np.diag(vHat**-1) + SigmaInverse), np.eye(Sigma.shape[0]))
        mNew = np.dot(VNew, mHat / vHat + muSigma)
        vNew = VNew.diagonal().copy()

        if np.max(np.abs(np.r_[m-mNew, v-vNew])) >= 1e-6:
            damping *= 0.99
            v = vNew
            m = mNew
        else:
            break

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
