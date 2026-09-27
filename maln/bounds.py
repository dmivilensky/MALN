"""Certified upper bounds of the paper, as explicit functions of the parameters."""
import numpy as np


def belt_smsc_bound(L0, mu, eps, R, theta, grid=60):
    """Smallest D_1 = a (r0 + lam m) arcsin(theta), lam = 1/L0, over parameters (rho, a, m)
    with rho <= R, a <= m and the separation condition of Lemma C.1:
        a rho - mu rho^2/2 - D_0 - lam m^2/2 > eps,  D_0 = a theta r0,
        r0 = rho (m theta + a)/(m - a theta).
    Returns np.inf if no admissible parameters are found on the grid."""
    best = np.inf
    for kappa in np.geomspace(1.2, 60, grid):
        for t in np.geomspace(1e-3, mu * R ** 2 / eps, grid):
            rho = np.sqrt(t * eps / mu)
            a = kappa * eps / rho
            for q in np.linspace(0.02, 1.0, grid):
                m = a / q
                r0 = rho * (theta + q) / (1 - q * theta)
                lam = 1.0 / L0
                gap = a * rho - 0.5 * mu * rho ** 2 - a * theta * r0 - 0.5 * lam * m ** 2
                if gap > eps:
                    best = min(best, a * (r0 + lam * m) * np.arcsin(theta))
    return best


def quadratic_obstruction(mu, eps, R, theta):
    """Theorem III.5(iv): R sqrt(3 mu eps) theta, valid for eps <= mu R^2 / 3."""
    return R * np.sqrt(3 * mu * eps) * theta
