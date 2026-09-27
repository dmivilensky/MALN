"""Breakdown of the two-point method of Proposition IV.1 under a fixed discontinuous perturbation.

Instance: f(x) = -(2 eps / R) <u, x> on B_R (Lipschitz constant 2 eps / R <= M), and the oracle
O(x) = f(x) + delta * sgn(<u, x>). The method is run with all parameters of Proposition IV.1;
the iteration is simulated exactly through the reduced chain of maln.breakdown."""
import csv, json, time, numpy as np
from maln.algorithms import two_point_parameters
from maln.breakdown import reduced_run, breakdown_level, mean_abs_coordinate

M, R, EPS = 1.0, 1.0, 1.0 / 8
SLOPE = 2 * EPS / R
DIMS = [2, 4, 8, 16, 32, 64]
RATIOS = [0.98, 1.02]
rows, summary = [], {}
for n in DIMS:
    h, Rp, K, eta = two_point_parameters(n, M, R, EPS)
    dmf = breakdown_level(n, h, SLOPE)
    dguar = EPS ** 2 / (84 * np.sqrt(n) * M * R)
    seeds = 4 if n <= 32 else 2
    t0 = time.time()
    gaps = {}
    for ratio in RATIOS:
        g = []
        for sd in range(seeds):
            sbar = reduced_run(n, K, h, eta, Rp, SLOPE, ratio * dmf, 10_000 * n + 100 * int(ratio * 100) + sd)
            g.append(SLOPE * (R - sbar) / EPS)
            rows.append(dict(n=n, K=K, ratio=ratio, seed=sd, delta=ratio * dmf, gap_over_eps=g[-1]))
        gaps[ratio] = g
    summary[n] = dict(K=K, h=h, delta_breakdown=dmf, delta_guaranteed=dguar,
                      breakdown_over_guaranteed=dmf / dguar,
                      breakdown_sqrt_n_MR_over_eps2=dmf * np.sqrt(n) * M * R / EPS ** 2,
                      mean_abs_coordinate=mean_abs_coordinate(n),
                      max_gap_below=max(gaps[RATIOS[0]]), min_gap_above=min(gaps[RATIOS[1]]),
                      seconds=time.time() - t0)
    print(n, summary[n], flush=True)
    assert summary[n]["max_gap_below"] <= 1 < summary[n]["min_gap_above"]
with open("results/breakdown.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(dict(M=M, R=R, eps=EPS, slope=SLOPE, ratios=RATIOS, dims=summary),
          open("results/breakdown.json", "w"), indent=2)
