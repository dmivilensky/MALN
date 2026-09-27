"""Compact kernel p_n(u) ~ (1-|u|^2)^k 1{|u|<1}, k = ceil(n/2), and exact constants."""
import numpy as np
from scipy.special import betaln, gammaln


def k_of(n):
    return int(np.ceil(n / 2))


def mean_norm(n):
    """E|U| for U with density p_n: |U|^2 ~ Beta(n/2, k+1)."""
    k = k_of(n)
    return np.exp(betaln(n / 2 + 0.5, k + 1) - betaln(n / 2, k + 1))


def grad_integral(n):
    """I_n = sup_|v|=1 int |d_v p_n| = 2k E[|U|/(1-|U|^2)] E|Theta_1|."""
    k = k_of(n)
    e_ratio = np.exp(betaln(n / 2 + 0.5, k) - betaln(n / 2, k + 1))
    e_theta = np.exp(gammaln(n / 2) - gammaln((n + 1) / 2)) / np.sqrt(np.pi)
    return 2 * k * e_ratio * e_theta


def sample_planar(n, size, rng):
    """Samples of (U_1, U_2, |U_rest|) for U with density p_n in R^n."""
    k = k_of(n)
    rad = np.sqrt(rng.beta(n / 2, k + 1, size))
    g1, g2 = rng.standard_normal(size), rng.standard_normal(size)
    c = rng.chisquare(n - 2, size) if n > 2 else np.zeros(size)
    nrm = np.sqrt(g1 ** 2 + g2 ** 2 + c)
    return rad * g1 / nrm, rad * g2 / nrm, rad * np.sqrt(c) / nrm
