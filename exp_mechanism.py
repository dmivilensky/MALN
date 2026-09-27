"""Leakage of the smoothed hard family into the visible sector (Lemma B.1)."""
import json, numpy as np, matplotlib.pyplot as plt
from maln.belt import Belt
from maln import style

style.apply()
eps, L, R, theta = 1e-2, 1.0, 1.0, 0.2
m, lam, a = np.sqrt(L * eps), 1.0 / L, 2 * eps / R
B = Belt(m, a, R, theta)
rmax = 1.25 * (B.r0 + lam * m)
r = np.linspace(1e-4, rmax, 500)
phi = np.linspace(-np.arcsin(theta), np.arcsin(theta), 241)
Rg, Pg = np.meshgrid(r, phi)
Tg, Zg = Rg * np.sin(Pg), Rg * np.cos(Pg)
env = B.moreau(Tg, Zg, lam)
Hr = B.H(r, lam)
disc_env = np.abs(env - Hr[None, :])
disc_belt = np.abs(B.g(Tg, Zg) - B.G(r)[None, :])
arcbound = a * Rg * np.abs(Pg)

res = {
    "eps": eps, "L": L, "R": R, "theta": theta, "m": m, "lambda": lam, "a": a,
    "r0": B.r0, "tail_radius": B.r0 + lam * m,
    "D0_certified": B.D0, "D1_certified": B.D1(lam),
    "belt_sup_discrepancy": float(disc_belt.max()),
    "envelope_sup_discrepancy": float(disc_env.max()),
    "max_ratio_to_pointwise_arc_bound": float(np.max(disc_env[arcbound > 0] / arcbound[arcbound > 0])),
    "envelope_sup_beyond_tail": float(disc_env[:, r >= B.r0 + lam * m].max()),
}
k = np.unravel_index(np.argmax(disc_env), disc_env.shape)
res["argmax_radius"] = float(r[k[1]]); res["argmax_latitude_deg"] = float(np.degrees(phi[k[0]]))
res["ratio_to_arc_bound_at_argmax"] = float(disc_env[k] / arcbound[k])
tol = 1e-12
for name, D in (("belt", disc_belt), ("envelope", disc_env)):
    for side, rows in (("plus_w", phi > 0), ("minus_w", phi < 0)):
        sup = D[rows].max(0) > tol
        res[f"{name}_support_end_{side}"] = float(r[sup].max())
res["b_over_m"] = B.b / B.m
json.dump(res, open("results/mechanism.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
im = ax[0].pcolormesh(r, np.degrees(phi), disc_env / eps, cmap="Blues", shading="auto",
                      vmin=0, vmax=disc_env.max() / eps, rasterized=True)
ax[0].axvline(B.r0 + lam * m, color=style.ORANGE, lw=1.4, ls="--")
ax[0].text(B.r0 + lam * m + 0.01, np.degrees(np.arcsin(theta)) * 0.78,
           r"$r_0+\lambda m$", color=style.ORANGE, fontsize=8)
ax[0].set_xlabel(r"radius $\|x\|$")
ax[0].set_ylabel(r"latitude $\arcsin(\langle w,x\rangle/\|x\|)$  [deg]")
ax[0].set_title(r"$|e_\lambda g_w(x)-H_\lambda(\|x\|)|/\varepsilon$ on the visible sector")
ax[0].grid(False)
cb = fig.colorbar(im, ax=ax[0], pad=0.02)
cb.outline.set_visible(False)

ax[1].plot(r, disc_belt.max(0) / eps, color=style.INK2, lw=1.2, label=r"belt $g_w$ vs $G$")
ax[1].plot(r, disc_env.max(0) / eps, color=style.BLUE, label=r"envelope $e_\lambda g_w$ vs $H_\lambda$")
ax[1].plot(r, np.minimum(a * r * np.arcsin(theta), B.D1(lam)) / eps, color=style.ORANGE,
           ls="--", lw=1.2, label=r"$\min\{a\|x\|\arcsin\theta,\ D_1\}$")
ax[1].set_xlabel(r"radius $\|x\|$")
ax[1].set_ylabel(r"maximum over latitude $/\varepsilon$")
ax[1].set_title("radial profile of the discrepancy")
ax[1].legend(loc="upper left")
ax[1].set_xlim(0, rmax); ax[1].set_ylim(0, 1.35 * B.D1(lam) / eps)
fig.subplots_adjust(wspace=0.38)
fig.savefig("figures/mechanism.pdf")
print(json.dumps(res, indent=2))
