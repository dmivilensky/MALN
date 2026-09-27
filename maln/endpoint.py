"""Simplex reconstruction near the endpoint L = mu and exact constrained minimization."""
import numpy as np


def simplex(n):
    """Unit vertices u_0..u_n of a regular simplex in R^n (rows)."""
    E = np.eye(n + 1) - 1.0 / (n + 1)
    Q, _ = np.linalg.qr(E[:, :n])          # orthonormal basis of the sum-zero subspace
    P = E @ Q
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def ball_quadratic_min(evals, V, b, R):
    """min 0.5 x^T H x + b^T x over |x| <= R for H = V diag(evals) V^T (full orthonormal V)."""
    bt = V.T @ b
    def xnorm(nu):
        return np.linalg.norm(bt / (evals + nu))
    if xnorm(0.0) <= R:
        nu = 0.0
    else:
        lo, hi = 0.0, 1.0
        while xnorm(hi) > R:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if xnorm(mid) > R else (lo, mid)
        nu = hi
    x = -V @ (bt / (evals + nu))
    return x


class NearQuadratic:
    """f(x) = mu/2 |x|^2 + <c,x> + L0/2 |W^T (x - p)|^2, W with orthonormal columns."""

    def __init__(self, mu, L0, c, p, W):
        self.mu, self.L0, self.c, self.p, self.W = mu, L0, c, p, W

    def __call__(self, X):
        X = np.atleast_2d(X)
        D = (X - self.p) @ self.W
        return 0.5 * self.mu * np.sum(X ** 2, 1) + X @ self.c + 0.5 * self.L0 * np.sum(D ** 2, 1)

    def minimum(self, R):
        n, k = self.W.shape
        Qf, _ = np.linalg.qr(np.hstack([self.W, np.random.default_rng(0).standard_normal((n, n - k))]))
        evals = np.full(n, float(self.mu)); evals[:k] += self.L0
        b = self.c - self.L0 * self.W @ (self.W.T @ self.p)
        x = ball_quadratic_min(evals, Qf, b, R)
        return x, self(x)[0]


def simplex_estimate(values, U, r, mu, R):
    n = U.shape[1]
    chat = n / (r * (n + 1)) * (values @ U)
    z = -chat / mu
    nz = np.linalg.norm(z)
    return z if nz <= R else z * (R / nz)


def guaranteed_loss(n, mu, L0, R, r, delta):
    return (np.sqrt(n) * (L0 * r / 4 + delta / r) + L0 * R) ** 2 / (2 * mu)


def best_radius(n, mu, L0, R, delta, floor=1e-6):
    """Query radius minimizing L0 r/4 + delta/r over (0, R]."""
    if L0 <= 0:
        return R
    return float(np.clip(2 * np.sqrt(delta / L0), floor * R, R))
