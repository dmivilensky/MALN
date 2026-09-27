"""Spherical-belt hard family and its Moreau envelope.

Every function of the family is invariant under orthogonal maps fixing the
hidden direction w, so it is evaluated in the two coordinates
    t = <w, x>,   z = ||x - t w||  (z >= 0).
"""
import numpy as np

GOLD = (np.sqrt(5.0) - 1.0) / 2.0


class Belt:
    """g_w(x) = max{ m*sigma_w(x) - b, -a<w,x> },  b = rho*(m*theta + a)."""

    def __init__(self, m, a, rho, theta):
        if not (0 < a <= m and 0 < theta < 1 and rho > 0):
            raise ValueError("parameters must satisfy 0 < a <= m, 0 < theta < 1")
        self.m, self.a, self.rho, self.theta = float(m), float(a), float(rho), float(theta)
        self.b = rho * (m * theta + a)
        self.r0 = self.b / (m - a * theta)
        self.D0 = a * theta * self.r0

    # support function of K_w = {|v| <= 1, |<w,v>| <= theta}
    def sigma(self, t, z):
        t, z = np.asarray(t, float), np.abs(np.asarray(z, float))
        r = np.hypot(t, z)
        vis = np.abs(t) <= self.theta * r
        return np.where(vis, r, self.theta * np.abs(t) + np.sqrt(1 - self.theta ** 2) * z)

    def g(self, t, z):
        return np.maximum(self.m * self.sigma(t, z) - self.b, -self.a * np.asarray(t, float))

    def G(self, r):
        return np.maximum(self.m * np.asarray(r, float) - self.b, 0.0)

    # projection of planar vectors (X1 along w, X2 orthogonal) onto k*K_w
    def _proj(self, X1, X2, k):
        nrm = np.hypot(X1, X2)
        scale = np.where(nrm > k, k / np.maximum(nrm, 1e-300), 1.0)
        P1, P2 = X1 * scale, X2 * scale
        bad = np.abs(P1) > k * self.theta
        c2 = k * np.sqrt(1 - self.theta ** 2)
        Q1 = np.sign(X1) * k * self.theta
        Q2 = np.clip(X2, -c2, c2)
        return np.where(bad, Q1, P1), np.where(bad, Q2, P2)

    def _phi(self, s, t, z, lam):
        X1 = t / lam + (1 - s) * self.a
        X2 = z / lam
        P1, P2 = self._proj(X1, X2, s * self.m)
        q1, q2 = P1 - (1 - s) * self.a, P2
        return q1 * t + q2 * z - 0.5 * lam * (q1 ** 2 + q2 ** 2) - s * self.b, q1, q2

    def moreau(self, t, z, lam, iters=90, return_grad=False):
        """e_lam g_w via the dual formula: a concave maximization over s in [0,1]."""
        t = np.atleast_1d(np.asarray(t, float))
        z = np.atleast_1d(np.abs(np.asarray(z, float)))
        t, z = np.broadcast_arrays(t, z)
        lo, hi = np.zeros_like(t), np.ones_like(t)
        x1 = hi - GOLD * (hi - lo)
        x2 = lo + GOLD * (hi - lo)
        f1 = self._phi(x1, t, z, lam)[0]
        f2 = self._phi(x2, t, z, lam)[0]
        for _ in range(iters):
            left = f1 < f2
            lo = np.where(left, x1, lo)
            hi = np.where(left, hi, x2)
            nx1 = np.where(left, x2, hi - GOLD * (hi - lo))
            nx2 = np.where(left, lo + GOLD * (hi - lo), x1)
            nf1 = np.where(left, f2, self._phi(nx1, t, z, lam)[0])
            nf2 = np.where(left, self._phi(nx2, t, z, lam)[0], f1)
            x1, x2, f1, f2 = nx1, nx2, nf1, nf2
        cands = [0.5 * (lo + hi), np.zeros_like(t), np.ones_like(t)]
        vals = [self._phi(c, t, z, lam) for c in cands]
        best = np.argmax(np.stack([v[0] for v in vals]), axis=0)
        val = np.choose(best, [v[0] for v in vals])
        if not return_grad:
            return val
        q1 = np.choose(best, [v[1] for v in vals])
        q2 = np.choose(best, [v[2] for v in vals])
        return val, q1, q2

    def H(self, r, lam):
        """Common radial branch: the envelope on the equator <w,x> = 0."""
        return self.moreau(np.zeros_like(np.asarray(r, float)), r, lam)

    def D1(self, lam):
        return self.a * (self.r0 + lam * self.m) * np.arcsin(self.theta)


def cap_probability(n, theta):
    """Exact P(|w_1| > theta) for w uniform on S^{n-1}."""
    from scipy.special import betainc
    return betainc((n - 1) / 2.0, 0.5, 1 - theta ** 2)


def hiding_theta(n, T, beta):
    """theta_* = sqrt(2 log(2(T+1)/(1-beta)) / (n-1))."""
    return np.sqrt(2 * np.log(2 * (T + 1) / (1 - beta)) / (n - 1))
