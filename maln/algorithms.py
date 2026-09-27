"""Two-point methods of Propositions IV.1 and IV.3 (projected, ball-smoothed, with selection among J runs)."""
import numpy as np


def two_point_parameters(n, M, R, eps):
    """Smoothing radius, projection radius, iterations, step size of Proposition IV.1."""
    h = 7 * eps / (64 * M)
    Rp = R - h
    K = int(np.ceil(1200 * n ** 2 * M ** 2 * R ** 2 / eps ** 2))
    S = 1.16 * n ** 2 * M ** 2
    eta = Rp / np.sqrt(K * S)
    return h, Rp, K, eta


def two_point_run(oracle, n, M, R, eps, rng):
    """One run of 2K queries; returns the averaged iterate."""
    h, Rp, K, eta = two_point_parameters(n, M, R, eps)
    x = np.zeros(n); acc = np.zeros(n)
    for _ in range(K):
        acc += x
        e = rng.standard_normal(n); e /= np.linalg.norm(e)
        G = n / (2 * h) * (oracle(x + h * e) - oracle(x - h * e)) * e
        x = x - eta * G
        nx = np.linalg.norm(x)
        if nx > Rp:
            x *= Rp / nx
    return acc / K


def two_point_lipschitz(oracle, n, M, R, eps, beta, rng):
    """Returns (output, number of oracle calls): J runs and selection by the oracle values."""
    K = two_point_parameters(n, M, R, eps)[2]
    J = int(np.ceil(np.log2(1 / beta)))
    cands = [two_point_run(oracle, n, M, R, eps, rng) for _ in range(J)]
    vals = [oracle(c) for c in cands]
    return cands[int(np.argmin(vals))], J * (2 * K + 1)


def two_point_smsc(oracle, n, L, mu, R, eps, beta, rng):
    """Method of Proposition IV.3: radius h = sqrt(5 eps/(16 L)), steps 4/(mu (k+1)),
    weighted averaging, J runs and selection by the oracle values."""
    h = np.sqrt(5 * eps / (16 * L))
    Rp = R - h
    K = int(np.ceil(800 * n ** 2 * L ** 2 * R ** 2 / (mu * eps)))
    J = int(np.ceil(np.log2(1 / beta)))
    cands = []
    for _ in range(J):
        x = np.zeros(n); acc = np.zeros(n); wsum = 0.0
        for k in range(1, K + 1):
            acc += k * x; wsum += k
            e = rng.standard_normal(n); e /= np.linalg.norm(e)
            G = n / (2 * h) * (oracle(x + h * e) - oracle(x - h * e)) * e
            x = x - 4.0 / (mu * (k + 1)) * G
            nx = np.linalg.norm(x)
            if nx > Rp:
                x *= Rp / nx
        cands.append(acc / wsum)
    vals = [oracle(c) for c in cands]
    return cands[int(np.argmin(vals))], J * (2 * K + 1)
