"""Certified hiding level after Moreau smoothing versus universal convolution.

usage: python exp_dimension.py [plot]   ('plot' redraws from results/dimension.csv)
"""
import csv, json, sys, numpy as np, matplotlib.pyplot as plt
from maln.belt import Belt, hiding_theta
from maln.kernel import grad_integral, mean_norm, sample_planar
from maln import style

EPS, L, R, T, BETA = 3e-5, 1.0, 1.0, 10 ** 6, 0.25
DIMS = [2 ** k for k in range(9, 18)]
SAMPLES = 400_000


def compute(seed=20260925):
    rng = np.random.default_rng(seed)
    rows = []
    for n in DIMS:
        th = hiding_theta(n, T, BETA)
        a = 2 * EPS / R
        m, lam = np.sqrt(L * EPS), 1.0 / L                    # Moreau: bias eps/2, smoothness L
        Bm = Belt(m, a, R, th)
        r = np.linspace(1e-5, 1.05 * (Bm.r0 + lam * m), 400)
        ph = np.linspace(-np.arcsin(th), np.arcsin(th), 61)
        Rg, Pg = np.meshgrid(r, ph)
        dm = np.abs(Bm.moreau(Rg * np.sin(Pg), Rg * np.cos(Pg), lam) - Bm.H(r, lam)[None, :]).max()
        In, En = grad_integral(n), mean_norm(n)                # convolution: bias eps/2, smoothness L
        mc = np.sqrt(L * EPS / (2 * In * En)); gam = EPS / (2 * mc * En)
        Bc = Belt(mc, a, R, th)
        U1, U2, U3 = sample_planar(n, SAMPLES, rng)
        dc, se = 0.0, 0.0
        for ri in np.linspace(1e-5, 1.1 * (Bc.r0 + gam), 70):
            def conv(p):
                t, z = ri * np.sin(p), ri * np.cos(p)
                return Bc.g(t + gam * U1, np.sqrt((z + gam * U2) ** 2 + (gam * U3) ** 2))
            base = conv(0.0)
            for p in np.linspace(-np.arcsin(th), np.arcsin(th), 11):
                diff = conv(p) - base                          # common samples
                if abs(diff.mean()) > dc:
                    dc, se = abs(diff.mean()), diff.std() / np.sqrt(SAMPLES)
        rows.append(dict(n=n, theta=th, D1_certified=Bm.D1(lam), moreau_measured=dm,
                         conv_measured=dc, conv_standard_error=se, m_moreau=m, m_conv=mc, gamma_conv=gam,
                         a=a, conv_D0=Bc.D0, I_n_over_sqrt_n=In / np.sqrt(n)))
        print(rows[-1], flush=True)
    with open("results/dimension.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plot():
    style.apply()
    rows = list(csv.DictReader(open("results/dimension.csv")))
    n = np.array([float(x["n"]) for x in rows])
    dmv = np.array([float(x["moreau_measured"]) for x in rows]) / EPS
    dcv = np.array([float(x["conv_measured"]) for x in rows]) / EPS
    d1 = np.array([float(x["D1_certified"]) for x in rows]) / EPS
    sm = np.polyfit(np.log(n[-4:]), np.log(dmv[-4:]), 1)[0]
    sc = np.polyfit(np.log(n[-4:]), np.log(dcv[-4:]), 1)[0]
    sd = np.polyfit(np.log(n[-4:]), np.log(d1[-4:]), 1)[0]
    d0c = np.array([float(x["conv_D0"]) for x in rows])
    s0 = np.polyfit(np.log(n[-4:]), np.log(d0c[-4:]), 1)[0]
    rel_se = max(float(x["conv_standard_error"]) / float(x["conv_measured"]) for x in rows)
    ins = np.array([float(x["I_n_over_sqrt_n"]) for x in rows])
    json.dump({"slope_moreau_last4": sm, "slope_conv_last4": sc, "slope_D1_last4": sd,
               "slope_conv_D0_last4": s0, "max_relative_standard_error_conv": rel_se,
               "I_n_over_sqrt_n_min": float(ins.min()), "I_n_over_sqrt_n_max": float(ins.max()),
               "min_measured_over_certified_moreau": float(np.min(dmv / d1)),
               "ratio_first": dcv[0] / dmv[0], "ratio_last": dcv[-1] / dmv[-1],
               "max_measured_over_certified_moreau": float(np.max(dmv / d1))},
              open("results/dimension_summary.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.loglog(n, d1, color=style.BLUE, lw=1.0, ls="--", label=r"certified level $D_1$ (Lemma B.1)")
    ax.loglog(n, dmv, "o", color=style.BLUE, label="Moreau envelope (measured)")
    ax.loglog(n, dcv, "s", color=style.ORANGE, label="universal convolution (measured)")
    ax.set_xlabel(r"dimension $n$"); ax.set_ylabel(r"visible discrepancy $/\varepsilon$")
    ax.set_ylim(4e-4, 2.5)
    ax.legend(loc="upper right")
    fig.savefig("figures/dimension.pdf")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "plot":
        compute()
    plot()
