"""Relative Mach number on surfaces of constant span (NASA TP-2879 figure 17 style) and shock position.

Span fractions are measured from the hub (NASA quotes span from the shroud: 90 % from hub = 10 % from shroud).
x = z (axial), y = r*theta: flow enters from the lower left, the rotor moves downwards (-theta), as in NASA's figure.
Run from the project root:  python scripts/06_mach_contours.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import FIG, MESH, PITCH, ROOT, Flowpath  # noqa: E402

GAM = 1.4
SPANS = [0.9, 0.7, 0.3]
CASES = [("near peak efficiency (m/m_choke 0.986)", "peak"), ("near stall (m/m_choke 0.956)", "stall")]
LEVELS = np.arange(0.40, 1.551, 0.05)

M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm, TH = M["Z"], M["R"], B["TH"]
I_LE, I_TE = int(M["I_LE"]), int(M["I_TE"])
JB, NJ = int(B["JB"]), int(B["NJ"])
NI, NJ1, NK = TH.shape
S = Flowpath().span(Zm, Rm)                      # span fraction from hub at every (i, k)


def relative_mach(f):
    W = f["V"] - f["Vgrid"]
    return np.linalg.norm(W, axis=-1) / np.sqrt(GAM * f["p"] / f["rho"])


def layer(q, s0):
    """Interpolate a node field q(i, j, k) to the surface of constant span s0."""
    z, r = np.empty(NI), np.empty(NI)
    th, v = np.empty((NI, NJ1)), np.empty((NI, NJ1))
    for i in range(NI):
        k = min(max(np.searchsorted(S[i], s0) - 1, 0), NK - 2)
        w = (s0 - S[i, k]) / (S[i, k + 1] - S[i, k])
        z[i] = (1 - w) * Zm[i, k] + w * Zm[i, k + 1]
        r[i] = (1 - w) * Rm[i, k] + w * Rm[i, k + 1]
        th[i] = (1 - w) * TH[i, :, k] + w * TH[i, :, k + 1]
        v[i] = (1 - w) * q[i, :, k] + w * q[i, :, k + 1]
    return z, r, th, v


def shock(z, mw, i0, i1):
    """Location of the steepest Mach drop along a streamwise grid line between i0 and i1."""
    g = np.gradient(mw[i0:i1], z[i0:i1])
    n = int(np.nanargmin(g)) + i0
    up = np.nanmax(mw[max(n - 6, 0):n + 1])
    dn = np.nanmin(mw[n:n + 7])
    return n, up, dn


fig, axes = plt.subplots(len(SPANS), 2, figsize=(13, 6.2 * len(SPANS)))
print(f"{'case':42s} {'span':>5s}  shock at mid-pitch: z - z_LE [% axial chord]   M before -> after")
for col, (title, d) in enumerate(CASES):
    f = np.load(ROOT / "cases" / d / "field.npz")
    mw_all = relative_mach(f)
    for row, s0 in enumerate(SPANS):
        ax = axes[row, col]
        z, r, th, mw = layer(mw_all, s0)
        X = np.broadcast_to(z[:, None], th.shape)
        Y = r[:, None] * th
        for shift in (0.0, 1.0):
            Ys = Y + shift * r[:, None] * PITCH
            cf = ax.contourf(X, Ys, np.ma.masked_invalid(mw), levels=LEVELS, cmap="RdYlBu_r", extend="both")
            ax.contour(X, Ys, np.ma.masked_invalid(mw), levels=LEVELS, colors="k", linewidths=0.3)
            ax.contour(X, Ys, np.ma.masked_invalid(mw), levels=[1.0], colors="k", linewidths=1.6)
            bz = z[I_LE:I_TE + 1]
            ax.fill_between(bz, Ys[I_LE:I_TE + 1, 0], Ys[I_LE:I_TE + 1, JB], color="k")
        # shock position along the mid-passage grid line
        jm = JB + (NJ - JB) // 2
        n, up, dn = shock(z, mw[:, jm], 0, I_TE)
        chord = z[I_TE] - z[I_LE]
        ax.plot(z[n], Y[n, jm], "w*", ms=14, mec="k")
        print(f"{title:42s} {s0*100:4.0f}%  {100 * (z[n] - z[I_LE]) / chord:+6.1f} %"
              f"                         {up:.2f} -> {dn:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(z[I_LE] - 2.5, z[I_TE] + 1.5)
        ax.set_ylim(np.nanmin(Y[I_LE - 10:I_TE + 5, 0]) - 1.0, np.nanmax(Y[I_LE:I_TE + 10, NJ]) + r[I_TE] * PITCH * 0.6)
        ax.set_title(f"{title}\n{s0*100:.0f} % span from hub ({(1 - s0)*100:.0f} % from shroud, NASA)", fontsize=10)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel(r"$r\,\theta$ [cm]")
        ax.annotate("", xy=(0.93, 0.08), xytext=(0.93, 0.22), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(0.95, 0.15, "rotation", transform=ax.transAxes, rotation=90, va="center", fontsize=8)
fig.colorbar(cf, ax=axes, shrink=0.5, label="relative Mach number", ticks=np.arange(0.4, 1.6, 0.1))
fig.suptitle("CFD relative Mach number (thick line: M = 1, star: steepest Mach drop at mid-pitch)", y=0.995)
fig.savefig(FIG / "mach_b2b.png", dpi=130, bbox_inches="tight")
print(f"figure: {FIG / 'mach_b2b.png'}")
