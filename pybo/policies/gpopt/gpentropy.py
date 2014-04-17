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
import scipy.spatial.distance as ssd

# local imports
from ._escov import get_cov, get_crosscov

# exported symbols
__all__ = []


def run_ep_iteration(mu, Sigma, ymin, sn2):
    # initial marginal approximation to our posterior given the zeroed factors
    # given below. note we're working with the "natural" parameters.
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
            vals, _ = np.linalg.eig(np.diag(1/tauHatNew) + Sigma)

            if any(1/vals <= 1e-10):
                damping *= 0.5
            else:
                break

        # our new approximate factors.
        tauHat = tauHatNew
        rhoHat = rhoHatNew

        # the new posterior.
        R = sla.cholesky(Sigma + np.diag(1/tauHat))
        V = sla.solve_triangular(R, Sigma, trans=True)
        V = Sigma - np.dot(V.T, V)
        m = np.dot(V, rhoHat) + sla.cho_solve((R, False), mu) / tauHat

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
    K = get_cov(X, xstar, 1/ell**2, sf2, sn2)
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

    Xstar = np.asfortranarray(xstar[:,None])
    Kstar = get_crosscov(Xstar, X, xstar, 1/ell**2, sf2, sn2)
    Vstar = sla.solve_triangular(R, Kstar.T, trans=True)

    mustar = float(np.dot(Vstar.T, alpha))
    s2star = float(sf2 - np.sum(Vstar**2, axis=0))

    return R, alpha, Vstar, mustar, s2star


def seard(ell, sf2, X1, X2):
    # XXX: gross! in order to keep this self-contained I'll just reimplement the
    # SEARD kernel here. but ultimately all the cov computations should be moved
    # into the kernel objects in a smart way.
    return sf2 * np.exp(-0.5*ssd.cdist(X1/ell, X2/ell, 'sqeuclidean'))


def predict_ep(Xtest, X, y, xstar, ell, sf2, sn2):
    X = np.asfortranarray(X)
    Xstar = np.asfortranarray(xstar[:,None])
    Xtest = np.asfortranarray(Xtest)

    # get the cholesky and weight terms.
    R, alpha, Vstar, mustar, s2star = run_ep(X, y, xstar, ell, sf2, sn2)

    # get the kernel wrt our inputs and constraints
    Ktest = get_crosscov(Xtest, X, xstar, 1/ell**2, sf2, sn2)
    Vtest = sla.solve_triangular(R, Ktest.T, trans=True)

    # compute the posterior on our test points without the constraint that
    # f(xstar) < f(xtest_i).
    mu = np.dot(Vtest.T, alpha)
    s2 = sf2 - np.sum(Vtest**2, axis=0)

    # the covariance between each test point and xstar.
    rho = np.ravel(seard(ell, sf2, Xtest, Xstar) - np.dot(Vtest.T, Vstar))
    rho *= 1 - 1e-4

    s = s2 + s2star - 2*rho
    a = (mu - mustar) / np.sqrt(s)
    b = np.exp(ss.norm.logpdf(a) - ss.norm.logcdf(a))

    mu += b * (s2-rho) / np.sqrt(s)
    s2 -= b * (b+a) * (s2-rho)**2 / s

    return mu, s2
