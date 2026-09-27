"""Exact reduced simulation of the two-point method of Proposition IV.1 on the instance
f(x) = -(c/R) <u, x>, O(x) = f(x) + delta * sgn(<u, x>),  sgn(t) = 1 for t > 0 and -1 for t <= 0.

For this instance the iteration depends on x_k only through s_k = <u, x_k> and z_k = |x_k - s_k u|,
and on the direction e_k only through e_1 = <u, e_k> and c = <e_k, (x_k - s_k u)/z_k>; the pair
(e_1, c) has the law of the first two coordinates of a uniform point of the sphere S^{n-1}.
The chain (s_k, z_k) is therefore simulated exactly in distribution with O(1) work per step."""
import numpy as np
import numba
from scipy.special import gammaln


def mean_abs_coordinate(n):
    """E|<e, u>| for e uniform on S^{n-1}."""
    return float(np.exp(gammaln(n / 2) - gammaln((n + 1) / 2)) / np.sqrt(np.pi))


def breakdown_level(n, h, slope):
    """Noise level at which the mean increment of <u, x_k> vanishes at the discontinuity."""
    return slope * h / (n * mean_abs_coordinate(n))


@numba.njit(cache=True)
def reduced_run(n, K, h, eta, Rp, slope, delta, seed):
    """Returns <u, xbar> for one run of K iterations started at x_1 = 0."""
    np.random.seed(seed)
    s = 0.0; z = 0.0; acc = 0.0
    for _ in range(K):
        acc += s
        g1 = np.random.standard_normal(); g2 = np.random.standard_normal()
        rest = np.random.gamma((n - 2) / 2.0, 2.0) if n > 2 else 0.0
        nr = np.sqrt(g1 * g1 + g2 * g2 + rest)
        e1 = g1 / nr; c = g2 / nr
        sp = s + h * e1; sm = s - h * e1
        op = -slope * sp + (delta if sp > 0 else -delta)
        om = -slope * sm + (delta if sm > 0 else -delta)
        kap = n / (2 * h) * (op - om)
        s_new = s - eta * kap * e1
        zz = (z - eta * kap * c) ** 2 + (eta * kap) ** 2 * max(0.0, 1.0 - e1 * e1 - c * c)
        r2 = s_new * s_new + zz
        if r2 > Rp * Rp:
            sc = Rp / np.sqrt(r2); s_new *= sc; zz *= sc * sc
        s = s_new; z = np.sqrt(zz)
    return acc / K
