"""Where does the extra loss near stall come from? Entropy analysis of the CFD solutions.

1. Pitchwise mass-averaged entropy rise on the meridional plane (peak, near stall, difference).
2. Spanwise efficiency change between the two points at station 2: CFD vs NASA.
3. Entropy jump across the passage shock at mid-pitch, per span, for both points: how much of the
   extra loss near stall is generated directly by the stronger shock.

Entropy rise s - s_ref = cp ln(T/T_ref) - R ln(p/p_ref), with the inlet stagnation state as reference.
Efficiency penalty of an entropy rise ds (small-loss approximation): d_eta ~ T_t2 ds / (cp (T_t2 - T_t1)).
Run from the project root:  python scripts/07_loss.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import FIG, MESH, ROOT, Flowpath  # noqa: E402

GAM, RGAS = 1.4, 287.058
CP = GAM * RGAS / (GAM - 1)
PREF, TREF = 101325.0, 288.15
Z_ST2 = 11.011
NASA_DATA = Path.home() / "transonic-fan" / "data"

M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm, TH = M["Z"], M["R"], B["TH"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
JB, NJ = int(B["JB"]), int(B["NJ"])
NI, NJ1, NK = TH.shape
SPAN = Flowpath().span(Zm, Rm)


def load(d):
    f = np.load(ROOT / "cases" / d / "field.npz")
    p, T, rho, V = f["p"], f["T"], f["rho"], f["V"]
    s = CP * np.log(T / TREF) - RGAS * np.log(p / PREF)
    W = V - f["Vgrid"]
    mw = np.linalg.norm(W, axis=-1) / np.sqrt(GAM * p / rho)
    tt = T + (V ** 2).sum(-1) / (2 * CP)
    pt = p * (tt / T) ** (GAM / (GAM - 1))
    flux = np.abs(rho * V[..., 2])                   # axial mass flux as averaging weight
    return dict(s=s, mw=mw, tt=tt, pt=pt, flux=flux)


def pitch_avg(c, q):
    """Mass-weighted pitchwise average of q at every (i, k) (pitch widths r*dtheta)."""
    dth = np.gradient(TH, axis=1)
    w = c["flux"] * dth
    w = np.where(np.isnan(q), 0.0, w)
    return np.nansum(w * q, axis=1) / np.maximum(w.sum(axis=1), 1e-30)


def station2(c):
    """Spanwise mass-averaged Pt, Tt, entropy and efficiency at z = Z_ST2."""
    r = np.zeros(NK)
    out = {q: np.zeros(NK) for q in ("pt", "tt", "s")}
    for k in range(NK):
        i = np.searchsorted(Zm[:, k], Z_ST2) - 1
        w = (Z_ST2 - Zm[i, k]) / (Zm[i + 1, k] - Zm[i, k])
        g = lambda a: (1 - w) * a[i, :, k] + w * a[i + 1, :, k]  # noqa: E731
        wt = g(c["flux"]) * np.gradient(g(TH))
        r[k] = (1 - w) * Rm[i, k] + w * Rm[i + 1, k]
        for q in out:
            out[q][k] = (wt * g(c[q])).sum() / wt.sum() if wt.sum() > 0 else np.nan
    eta = ((out["pt"] / PREF) ** ((GAM - 1) / GAM) - 1) / (out["tt"] / TREF - 1)
    return r, out, eta


def shock_jump(c):
    """Entropy and Mach on both sides of the passage shock along the mid-pitch line, per k."""
    jm = JB + (NJ - JB) // 2
    res = []
    for k in range(1, K_TIP):
        mw, s, z = c["mw"][:I_TE, jm, k], c["s"][:I_TE, jm, k], Zm[:I_TE, k]
        if np.nanmax(mw) < 1.05:                    # no shock on this layer
            res.append((SPAN[I_LE, k], np.nan, np.nan, np.nan, np.nan))
            continue
        n = int(np.nanargmin(np.gradient(mw, z)))
        a, b = slice(max(n - 8, 0), max(n - 3, 1)), slice(n + 3, n + 8)
        res.append((SPAN[I_LE, k], np.nanmean(s[b]) - np.nanmean(s[a]), np.nanmax(mw[max(n - 6, 0):n + 1]),
                    np.nanmin(mw[n:n + 7]), 100 * (z[n] - Zm[I_LE, k]) / (Zm[I_TE, k] - Zm[I_LE, k])))
    return np.array(res)


cases = {"peak": load("peak"), "stall": load("stall")}
sm = {n: pitch_avg(c, c["s"]) for n, c in cases.items()}
st = {n: station2(c) for n, c in cases.items()}
sj = {n: shock_jump(c) for n, c in cases.items()}

# ----------------------------------------------------------------------------- figure 1: meridional entropy
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
vmax = np.nanpercentile(sm["stall"], 99)
for a, (title, q, cmap, vr) in zip(ax, [("near peak efficiency", sm["peak"], "magma_r", (0, vmax)),
                                        ("near stall", sm["stall"], "magma_r", (0, vmax)),
                                        ("near stall - near peak", sm["stall"] - sm["peak"], "RdBu_r", (-15, 15))]):
    cf = a.contourf(Zm, Rm, q, levels=np.linspace(*vr, 31), cmap=cmap, extend="both")
    fig.colorbar(cf, ax=a, label="pitch-averaged s - s_ref [J/kg/K]")
    a.plot(Zm[I_LE, :K_TIP + 1], Rm[I_LE, :K_TIP + 1], "k-", lw=1)
    a.plot(Zm[I_TE, :K_TIP + 1], Rm[I_TE, :K_TIP + 1], "k-", lw=1)
    a.plot(Zm[I_LE:I_TE + 1, K_TIP], Rm[I_LE:I_TE + 1, K_TIP], "k--", lw=1)
    a.axvline(Z_ST2, color="g", lw=1, ls=":")
    a.set_xlim(-3, 14)
    a.set_title(title)
    a.set_xlabel("z [cm]")
    a.set_aspect("equal")
ax[0].set_ylabel("r [cm]")
fig.suptitle("Entropy rise, pitchwise mass-averaged (dotted green: NASA station 2)")
fig.tight_layout()
fig.savefig(FIG / "loss_meridional.png", dpi=140)

# ----------------------------------------------------------------------------- figure 2: spanwise
nasa = pd.read_csv(NASA_DATA / "rotor67_station2.csv", comment="#")
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
r, _, eta_p = st["peak"]
_, _, eta_s = st["stall"]
ax[0].plot(100 * (eta_s - eta_p), r, "-", color="tab:blue", label="CFD (m/m_choke 0.986 -> 0.956)")
npk = nasa[nasa.point == "peak_efficiency"].set_index("radius_cm")
nst = nasa[nasa.point == "near_stall"].set_index("radius_cm")
e = lambda d: (d.p0_ratio ** ((GAM - 1) / GAM) - 1) / (d.t0_ratio - 1)  # noqa: E731
de_n = (e(nst) - e(npk)).dropna()
ax[0].plot(100 * de_n.values, de_n.index, "ko-", mfc="none", label="NASA (m/m_choke 0.989 -> 0.924)")
ax[0].axvline(0, color="0.5", lw=0.8)
ax[0].set_xlabel("efficiency change near stall - near peak [points]")
ax[0].set_ylabel("radius at station 2 [cm]")
ax[0].set_title("Where the efficiency is lost (station 2)")
ax[0].legend(fontsize=8)
ax[0].set_xlim(-12, 6)
for n, col in (("peak", "tab:green"), ("stall", "tab:red")):
    j = sj[n]
    ax[1].plot(j[:, 1], 100 * j[:, 0], "-", color=col, label=f"near {n}")
    ax[2].plot(j[:, 4], 100 * j[:, 0], "-", color=col, label=f"near {n}")
ax[1].set_xlabel("entropy jump across the passage shock at mid-pitch [J/kg/K]")
ax[1].set_ylabel("span from hub [%]")
ax[1].set_title("Shock loss")
ax[2].set_xlabel("shock position at mid-pitch [% axial chord from LE]")
ax[2].set_title("Shock position")
for a in ax:
    a.grid(alpha=0.3)
ax[1].legend(fontsize=8)
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "loss_spanwise.png", dpi=140)

# ----------------------------------------------------------------------------- table
tt1 = TREF
print("span   shock pos [% chord]   M before->after          ds_shock [J/kg/K]        d_eta from shock")
print("       peak   stall          peak         stall       peak   stall  diff     [points, approx]")
for s0 in (0.5, 0.6, 0.7, 0.8, 0.9):
    row = []
    for n in ("peak", "stall"):
        j = sj[n]
        row.append(j[np.nanargmin(np.abs(j[:, 0] - s0))])
    p, q = row
    tt2 = np.nanmean(cases["stall"]["tt"][I_TE + 5, :, :])
    deta = 100 * tt2 * (q[1] - p[1]) / (CP * (tt2 - tt1))
    print(f"{s0*100:3.0f}%   {p[4]:+5.1f}  {q[4]:+5.1f}        {p[2]:.2f}->{p[3]:.2f}    {q[2]:.2f}->{q[3]:.2f}   "
          f"{p[1]:5.1f}  {q[1]:5.1f}  {q[1]-p[1]:+5.1f}      {-deta:+.1f}")
print(f"figures: {FIG / 'loss_meridional.png'}, {FIG / 'loss_spanwise.png'}")
