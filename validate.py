"""Independent numerical checks of the statements used in the paper.
Every check raises AssertionError on failure; results/validation.json records the margins."""
import json, numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest
from maln.belt import Belt, cap_probability
from maln.kernel import grad_integral, mean_norm, sample_planar, k_of
from maln.endpoint import simplex, NearQuadratic, simplex_estimate, guaranteed_loss
from maln.bounds import belt_smsc_bound
from maln.algorithms import two_point_lipschitz, two_point_parameters, two_point_run, two_point_smsc
from maln.breakdown import reduced_run, breakdown_level

rng = np.random.default_rng(20260927)
out = {}


def random_belt():
    m = rng.uniform(0.3, 2.0); a = rng.uniform(0.02, 1.0) * m
    return Belt(m, a, rng.uniform(0.2, 2.0), rng.uniform(0.02, 0.25)), 10 ** rng.uniform(-2, 0.5)


# 1. dual formula for the Moreau envelope versus primal minimization in the invariant plane
err = 0.0
for _ in range(40):
    B, lam = random_belt()
    t, z = rng.uniform(-3, 3), abs(rng.uniform(-3, 3))
    e = B.moreau(t, z, lam)[0]
    obj = lambda y: B.g(y[0], y[1]) + ((y[0] - t) ** 2 + (y[1] - z) ** 2) / (2 * lam)
    best = min(minimize(obj, np.array([t, z]) + rng.normal(0, 1, 2), method="Nelder-Mead",
                        options=dict(xatol=1e-11, fatol=1e-13, maxiter=20000)).fun for _ in range(10))
    err = max(err, abs(best - e))
assert err < 1e-8; out["moreau_dual_vs_primal_max_abs_error"] = err

# 2. belt lemma
worst_gap, worst_tail, worst_min = 0.0, 0.0, 0.0
for _ in range(200):
    B, _ = random_belt()
    r = rng.uniform(0, 4, 4000); ph = rng.uniform(-np.arcsin(B.theta), np.arcsin(B.theta), 4000)
    d = np.abs(B.g(r * np.sin(ph), r * np.cos(ph)) - B.G(r))
    worst_gap = max(worst_gap, d.max() / B.D0)
    worst_tail = max(worst_tail, d[r >= B.r0].max(initial=0))
    tt, zz = rng.uniform(-4, 4, 4000), rng.uniform(0, 4, 4000)
    worst_min = max(worst_min, (-B.a * B.rho - B.g(tt, zz)).max())
    assert abs(B.g(B.rho, 0.0) + B.a * B.rho) < 1e-12
assert worst_gap <= 1 + 1e-12 and worst_tail < 1e-12 and worst_min <= 1e-12
out["belt_visible_discrepancy_over_D0_max"] = worst_gap

# 3-4. Moreau hiding: pointwise arc bound, exact tail, tangential gradient
worst_arc, worst_tail, worst_tan = 0.0, 0.0, 0.0
for _ in range(60):
    B, lam = random_belt()
    rmax = 1.3 * (B.r0 + lam * B.m)
    r = rng.uniform(1e-3, rmax, 3000); ph = rng.uniform(-np.arcsin(B.theta), np.arcsin(B.theta), 3000)
    e, q1, q2 = B.moreau(r * np.sin(ph), r * np.cos(ph), lam, return_grad=True)
    d = np.abs(e - B.H(r, lam))
    worst_arc = max(worst_arc, np.max(d / (B.a * r * np.abs(ph) + 1e-300)))
    worst_tail = max(worst_tail, d[r >= B.r0 + lam * B.m].max(initial=0))
    tangential = np.abs(q1 * np.cos(ph) - q2 * np.sin(ph))       # component orthogonal to x
    worst_tan = max(worst_tan, tangential.max() / B.a)
assert worst_arc <= 1 + 1e-6 and worst_tail < 1e-10 and worst_tan <= 1 + 1e-9
out.update(moreau_over_arc_bound_max=worst_arc, moreau_tail_max=worst_tail,
           tangential_gradient_over_a_max=worst_tan)

# 5. kernel constants against Monte Carlo
for n in (8, 32):
    U1, U2, U3 = sample_planar(n, 2_000_000, rng)
    rad2 = U1 ** 2 + U2 ** 2 + U3 ** 2
    mc_I = 2 * k_of(n) * np.mean(np.abs(U1) / (1 - rad2))
    mc_E = np.mean(np.sqrt(rad2))
    assert abs(mc_I / grad_integral(n) - 1) < 0.01 and abs(mc_E / mean_norm(n) - 1) < 0.002
    out[f"kernel_n{n}_MC_over_exact"] = [mc_I / grad_integral(n), mc_E / mean_norm(n)]

# 6. planar reduction of the convolution versus full-dimensional sampling
n, B = 12, Belt(1.0, 0.3, 1.0, 0.2)
x = np.zeros(n); x[0], x[1] = 0.1, 0.5
k = k_of(n)
Z = rng.standard_normal((400_000, n)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
Ufull = Z * np.sqrt(rng.beta(n / 2, k + 1, 400_000))[:, None]
Y = x + 0.3 * Ufull
full = B.g(Y[:, 0], np.linalg.norm(Y[:, 1:], axis=1)).mean()
U1, U2, U3 = sample_planar(n, 400_000, rng)
planar = B.g(0.1 + 0.3 * U1, np.sqrt((0.5 + 0.3 * U2) ** 2 + (0.3 * U3) ** 2)).mean()
assert abs(full - planar) < 3e-3; out["convolution_planar_vs_full"] = [full, planar]

# 7. simplex identities and attainment of n delta^2/(2 mu R^2) for odd n
for n in (3, 7, 15, 31):
    U = simplex(n)
    assert np.allclose(U.sum(0), 0) and np.allclose(U.T @ U, (n + 1) / n * np.eye(n))
    s = np.array([1.0, -1.0] * ((n + 1) // 2))
    mu, R, delta = 1.0, 1.0, 1e-3
    f = NearQuadratic(mu, 0.0, np.zeros(n), np.zeros(n), np.zeros((n, 1)))
    xh = simplex_estimate(f(R * U) + delta * s, U, R, mu, R)
    ratio = f(xh)[0] / (n * delta ** 2 / (2 * mu * R ** 2))
    assert abs(ratio - 1) < 1e-9
out["simplex_attainment_odd_n"] = "ratio 1 for n in (3,7,15,31)"

# 8. near-endpoint guarantee on random quadratic instances
worst = 0.0
for _ in range(400):
    n = int(rng.integers(2, 40)); mu = 1.0; R = 1.0
    L0 = 10 ** rng.uniform(-4, 0); delta = 10 ** rng.uniform(-5, -1); r = rng.uniform(0.05, 1) * R
    kk = int(rng.integers(1, n + 1)); W, _ = np.linalg.qr(rng.standard_normal((n, kk)))
    c = rng.standard_normal(n) * rng.uniform(0, 3); p = rng.standard_normal(n) * rng.uniform(0, 1)
    f = NearQuadratic(mu, L0, c, p, W); xs, fs = f.minimum(R)
    U = simplex(n); s = rng.choice([-1.0, 1.0], n + 1)
    xh = simplex_estimate(f(r * U) + delta * s, U, r, mu, R)
    worst = max(worst, (f(xh)[0] - fs) / guaranteed_loss(n, mu, L0, R, r, delta))
assert worst <= 1 + 1e-9; out["near_endpoint_loss_over_guarantee_max"] = worst

# 9. spherical cap probability
for n, th in ((20, 0.3), (200, 0.15)):
    W = rng.standard_normal((1_000_000, n)); w1 = W[:, 0] / np.linalg.norm(W, axis=1)
    mc = np.mean(np.abs(w1) > th); ex = cap_probability(n, th); bd = 2 * (1 - th ** 2) ** ((n - 1) / 2)
    assert abs(mc - ex) < 5e-3 * max(ex, 1e-3) + 1e-3 and ex <= bd
    out[f"cap_n{n}"] = dict(monte_carlo=mc, exact=float(ex), bound=bd)

# 10. adaptive Gaussian observations: invariance of the transcript under the reflection
N, T, reps = 6, 5, 200_000
Z = rng.standard_normal((reps, N))
A0 = rng.standard_normal(N)
Mat = rng.standard_normal((T, N, N))
def transcript(Z):
    Ys, As = [], []
    a = np.tile(A0, (len(Z), 1))
    for t in range(T):
        y = np.sum(a * Z, 1); Ys.append(y); As.append(a)
        a = np.tanh(y)[:, None] * (Mat[t] @ np.ones(N))[None, :] + np.cos(y)[:, None] * (Mat[t] @ A0)[None, :]
    return np.stack(Ys, 1), np.stack(As, 1)
Y, As = transcript(Z)
Q = np.linalg.qr(np.concatenate([np.transpose(As, (0, 2, 1)), rng.standard_normal((reps, N, 1))], 2))[0]
xi = Q[:, :, T]
g = np.sum(xi * Z, 1)
Zr = Z - 2 * g[:, None] * xi
Yr, _ = transcript(Zr)
assert np.max(np.abs(Yr - Y)) < 1e-9
ks = kstest(g, "norm").pvalue
assert ks > 1e-3
out["gaussian_reflection"] = dict(max_transcript_change=float(np.max(np.abs(Yr - Y))), ks_pvalue_g=ks,
                                  mean_abs_moment_diff=float(np.abs(np.mean(Zr ** 2, 0) - np.mean(Z ** 2, 0)).max()))

# 11. certified smooth strongly convex bounds of Theorem III.4 and Lemma C.1
for L0, eps, th in ((5.0, 1e-2, 0.25), (64.0, 1e-2, 0.25), (400.0, 1e-3, 0.1)):
    mu, R = 1.0, 1.0
    D1 = belt_smsc_bound(L0, mu, eps, R, th, grid=30)
    if L0 >= 36:
        assert D1 <= 18 / 5 * eps * th ** 2 + 14 * eps * th * np.sqrt(mu / L0)
    assert D1 <= 11 * eps * th ** 2 + 30 * eps * th * np.sqrt(mu / L0)
    out[f"smsc_certified_L0_{L0}"] = D1 / eps

# 12. explicit constants and separation conditions of Theorems III.1-III.4, III.5(iv),
#     Corollaries III.6-III.7 and Proposition V.3 on random admissible parameters
tol = 1 + 1e-12
for _ in range(20000):
    th = rng.uniform(1e-3, 0.25); eps = 10 ** rng.uniform(-6, 0)
    M, R = 10 ** rng.uniform(-2, 2), 10 ** rng.uniform(-2, 2)
    if eps <= M * R / 8:                                    # Theorem III.1
        B = Belt(M, 5 * eps / (4 * R), R, th)
        assert B.D0 <= (4 / 3 * eps * th ** 2 + 5 / 3 * eps ** 2 * th / (M * R)) * tol
        assert B.a * R - B.D0 > eps
    L = 10 ** rng.uniform(-2, 2)
    if eps <= L * R ** 2 / 16:                              # Theorem III.2
        m = np.sqrt(L * eps); B = Belt(m, 2 * eps / R, R, th)
        assert B.D1(1 / L) <= (12 / 5 * eps * th ** 2 + 7 * eps ** 1.5 * th / (np.sqrt(L) * R)) * tol
        assert B.a * R - B.D0 - m ** 2 / (2 * L) > eps
    nu = rng.uniform(0.01, 1.0); Lnu = 10 ** rng.uniform(-2, 2)
    mnu = (2 ** (nu - 1) * Lnu * eps ** nu) ** (1 / (1 + nu))
    if 4 * eps <= mnu * R:                                  # Corollary III.6
        lam = eps / mnu ** 2; B = Belt(mnu, 2 * eps / R, R, th)
        bound = 12 / 5 * eps * th ** 2 + 7 * 2 ** ((1 - nu) / (1 + nu)) * eps ** ((2 + nu) / (1 + nu)) * th \
            / (R * Lnu ** (1 / (1 + nu)))
        assert B.D1(lam) <= bound * tol and B.a * R - B.D0 - lam * mnu ** 2 / 2 > eps
        assert (1 / lam) ** nu * (2 * mnu) ** (1 - nu) <= Lnu * tol   # Hoelder constant of the envelope
    mu = 10 ** rng.uniform(-2, 2)
    if eps <= mu * R ** 2 and M - mu * R >= 4 * np.sqrt(mu * eps):   # Theorem III.3
        B = Belt(M - mu * R, 2 * np.sqrt(mu * eps), np.sqrt(eps / mu), th)
        assert B.D0 <= 16 / 7 * (eps * th ** 2 + 2 * np.sqrt(mu) * eps ** 1.5 * th / (M - mu * R)) * tol
        assert 2 * eps - eps / 2 - B.D0 > eps
    p = rng.uniform(2, 6); mup = 10 ** rng.uniform(-2, 2)
    Mp = 2 ** (p - 2) * mup * R ** (p - 1); rhop = (p * eps / (2 ** (p - 1) * mup)) ** (1 / p)
    if rhop <= R and M - Mp >= 4 * eps / rhop:              # Corollary III.7
        B = Belt(M - Mp, 2 * eps / rhop, rhop, th)
        assert B.D0 <= 16 / 7 * (eps * th ** 2 + 2 * eps ** 2 * th / ((M - Mp) * rhop)) * tol
        assert abs(2 ** (p - 2) * mup / p * rhop ** p - eps / 2) < 1e-9 * eps
        assert 2 * eps - eps / 2 - B.D0 > eps
    L0 = mu * 10 ** rng.uniform(np.log10(36), 4)
    if eps <= mu * R ** 2:                                  # Theorem III.4(a)
        B = Belt(np.sqrt(L0 * eps), 3 * np.sqrt(mu * eps), np.sqrt(eps / mu), th)
        assert B.D1(1 / L0) <= (18 / 5 * eps * th ** 2 + 14 * eps * th * np.sqrt(mu / L0)) * tol
        assert 3 * eps - eps / 2 - B.D0 - eps / 2 > eps
    L0 = mu * 10 ** rng.uniform(np.log10(5), 4)
    if eps <= mu * R ** 2 / 4:                              # Theorem III.4(b)
        B = Belt(2 * np.sqrt(L0 * eps), 4 * np.sqrt(mu * eps), 2 * np.sqrt(eps / mu), th)
        assert B.D1(1 / L0) <= (11 * eps * th ** 2 + 30 * eps * th * np.sqrt(mu / L0)) * tol
        assert 8 * eps - 2 * eps - B.D0 - 2 * eps > 1.05 * eps
    if eps <= mu * R ** 2 / 3:                              # Theorem III.5(iv)
        a = np.sqrt(3 * mu * eps)
        assert a / mu <= R * tol and a ** 2 / (2 * mu) * (1 - th ** 2) > eps
    assert 1.5 * eps * (1 - th) > eps                       # Proposition V.3
out["explicit_constants"] = "20000 random admissible parameter draws"

# 13. the two-point method of Proposition IV.1 on a Lipschitz instance with a deterministic perturbation
n, M, R, eps, beta = 4, 1.0, 1.0, 0.5, 0.25
delta = eps ** 2 / (84 * np.sqrt(n) * M * R)
xstar = np.array([0.4, -0.3, 0.2, 0.1])
f = lambda x: M * np.linalg.norm(x - xstar) / 2 + M * abs(x[0]) / 2
def oracle(x):
    hsh = np.sin(np.dot(x, [12.9898, 78.233, 37.719, 4.581]) * 43758.5453) * 43758.5453
    return f(x) + delta * (2 * (hsh - np.floor(hsh)) - 1)
grid = rng.uniform(-1, 1, (200000, n)); grid = grid[np.linalg.norm(grid, axis=1) <= R]
fstar = min(min(f(y) for y in grid[:20000]), f(np.array([0.0, -0.3, 0.2, 0.1])))
succ, calls = 0, 0
for rep in range(8):
    xo, calls = two_point_lipschitz(oracle, n, M, R, eps, beta, np.random.default_rng(rep))
    succ += f(xo) - fstar <= eps
assert succ >= 7
out["prop_IV1_two_point"] = dict(successes=int(succ), runs=8, oracle_calls_per_run=calls)

# 14. reduced chain of the breakdown experiment versus the full n-dimensional method
n, M, R, eps = 4, 1.0, 1.0, 0.5
slope = 2 * eps / R
h, Rp, K, eta = two_point_parameters(n, M, R, eps)
dmf = breakdown_level(n, h, slope)
res14 = {}
for ratio in (0.9, 1.1):
    d = ratio * dmf
    orc = lambda x, d=d: -slope * x[0] + (d if x[0] > 0 else -d)
    full = [two_point_run(orc, n, M, R, eps, np.random.default_rng(100 + r))[0] for r in range(4)]
    red = [reduced_run(n, K, h, eta, Rp, slope, d, 200 + r) for r in range(4)]
    gf = [slope * (R - s) / eps for s in full]; gr = [slope * (R - s) / eps for s in red]
    ok = (max(gf + gr) <= 1) if ratio < 1 else (min(gf + gr) > 1)
    assert ok and abs(np.mean(full) - np.mean(red)) < 0.02
    res14[f"ratio_{ratio}"] = dict(gap_over_eps_full=gf, gap_over_eps_reduced=gr)
out["breakdown_reduced_vs_full"] = res14

# 15. Proposition IV.3 and Corollary IV.4(e): constants, and norm of the minimizers of the hard functions (<= rho)
import math
for _ in range(20000):
    n_ = int(rng.integers(2, 10 ** 6)); mu = 10 ** rng.uniform(-2, 2); L = mu * 10 ** rng.uniform(0, 4)
    R = 10 ** rng.uniform(-2, 2); eps = mu * R ** 2 * 10 ** rng.uniform(-8, 0)
    h = math.sqrt(5 * eps / (16 * L)); d = eps / 5 * math.sqrt(mu / (n_ * L))
    assert h < math.sqrt(eps / L) and n_ * d ** 2 / (mu * h ** 2) <= 16 / 125 * eps * (1 + 1e-12)
    assert d / h <= 0.5 * L * R and 2 * (5 / 32 + 16 / 125 + 1 / 32) * eps + 2 * d < eps
    assert 4 * 6.25 * n_ ** 2 * L ** 2 * R ** 2 / (mu * (800 * n_ ** 2 * L ** 2 * R ** 2 / (mu * eps) + 1)) <= eps / 32 * (1 + 1e-12)
worst = 0.0
for _ in range(40):
    mu = 1.0; L0 = mu * 10 ** rng.uniform(np.log10(5), 3); eps = 10 ** rng.uniform(-3, -1); th = rng.uniform(0.02, 0.25)
    B = Belt(2 * np.sqrt(L0 * eps), 4 * np.sqrt(mu * eps), 2 * np.sqrt(eps / mu), th); lam = 1 / L0
    F = lambda y: B.moreau(y[0], abs(y[1]), lam)[0] + 0.5 * mu * (y[0] ** 2 + y[1] ** 2)
    best = min((minimize(F, rng.normal(0, 1, 2) * np.sqrt(eps / mu), method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000)) for _ in range(6)), key=lambda r: r.fun)
    ratio = np.hypot(*best.x) / (2 * np.sqrt(eps / mu))
    worst = max(worst, ratio)
    assert ratio <= 1 + 1e-6 and 2 + 1 / math.sqrt(6) < 2.41 < math.sqrt(6)
out["prop_IV3_constants"] = "20000 random parameter draws"
out["smsc_hard_minimizer_norm_over_bound_max"] = worst

# 16. the method of Proposition IV.3 on a smooth strongly convex instance with a deterministic perturbation
n, mu, L, R, eps, beta = 3, 1.0, 6.0, 1.0, 0.5, 0.25
delta = eps / 5 * np.sqrt(mu / (n * L))
z0 = np.array([0.5, 0.0, 0.0])
f = lambda x: 0.5 * mu * np.dot(x, x) + 0.5 * (L - mu) * (x[0] - z0[0]) ** 2
xf = np.array([(L - mu) * z0[0] / L, 0.0, 0.0])
assert np.linalg.norm(xf) <= R - np.sqrt(eps / L)
def oracle(x):                                   # discontinuity through the minimizer
    return f(x) + (delta if x[0] > xf[0] else -delta)
succ, gaps = 0, []
for rep in range(8):
    xo, calls = two_point_smsc(oracle, n, L, mu, R, eps, beta, np.random.default_rng(300 + rep))
    gaps.append(f(xo) - f(xf)); succ += gaps[-1] <= eps
assert succ >= 7
out["prop_IV3_two_point"] = dict(successes=int(succ), runs=8, max_gap_over_eps=max(gaps) / eps, oracle_calls_per_run=calls)

json.dump(out, open("results/validation.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
print("ALL CHECKS PASSED")
