"""Step 2 of the mesh: the blade-to-blade (pitchwise) distribution on every k-layer.

Topology (H-grid, one blade pitch per layer, theta increasing with j):

  j = 0 .......... JB ........................ NJ
  surface 2        surface 1                   surface 2 of the next blade (theta + PITCH)
  |-- blade strip --|-------- passage ----------|

  Inside the blade (I_LE <= i <= I_TE, k < K_TIP) the strip cells are blade material and will be
  removed in the next step. In the tip gap (k >= K_TIP) the strip cells are fluid: the tip
  leakage flow crosses the blade there. Upstream and downstream the strip and passage lines
  follow the camber line at the edges and relax to axial, uniformly spaced lines far away.

Output:
  mesh/b2b.npz               theta(i, j, k)
  figures/mesh_b2b.png

Run from the project root:  python scripts/03_mesh_b2b.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import FIG, MESH, PITCH, Blade, Flowpath  # noqa: E402

# ----------------------------------------------------------------------------- parameters
JB = 16                # pitchwise cells across the blade strip
NJ_PASS = 80           # pitchwise cells across the passage
DS_PITCH = 1.5e-3      # first cell height at the blade surfaces [cm]
RATIO = 1.20           # max growth ratio away from the blade surfaces
LB = 1.5               # decay length of the edge influence upstream/downstream [cm]
NJ = JB + NJ_PASS

n_fallback = {"strip": 0, "passage": 0}


def clustered_n(n, length, ds, ratio, tag):
    """n cells, first and last spacing ds, geometric growth <= ratio, uniform core. Returns [0, 1]."""
    if ds * n >= length:
        return np.linspace(0.0, 1.0, n + 1)
    for g in range(1, n // 2 + 1):
        grow = ds * ratio ** np.arange(g)
        nc = n - 2 * g
        rest = length - 2.0 * grow.sum()
        if rest <= 0.0 or nc <= 0:
            break
        core = rest / nc
        if ds * ratio ** g >= core:
            d = np.concatenate([grow, np.full(nc, core), grow[::-1]])
            x = np.concatenate([[0.0], np.cumsum(d)])
            return x / x[-1]
    # growth ratio cannot be respected with n cells: pure two-sided geometric distribution with the
    # smallest growth ratio q that fills the length (spacing ds * q**min(m, n-1-m))
    n_fallback[tag] += 1
    expo = np.minimum(np.arange(n), n - 1 - np.arange(n))
    q = brentq(lambda q: (ds * q ** expo).sum() - length, 1.0 + 1e-12, 20.0)
    x = np.concatenate([[0.0], np.cumsum(ds * q ** expo)])
    return x / x[-1]


# ----------------------------------------------------------------------------- inputs
M = np.load(MESH / "meridional.npz")
Zm, Rm = M["Z"], M["R"]
layer_a, layer_b, u_blade = M["layer_a"], M["layer_b"], M["u_blade"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
NI, NK = Zm.shape

fp = Flowpath()
bl = Blade(fp)
_, _, t1_tip, t2_tip = bl.on_layer(1.0, 0.0, u_blade)

TH = np.zeros((NI, NJ + 1, NK))
w_far = PITCH * JB / NJ
z_err = 0.0

for k in range(NK):
    z, r = Zm[:, k], Rm[:, k]
    if k <= K_TIP:
        zb, _, t1, t2 = bl.on_layer(layer_a[k], layer_b[k], u_blade)
        z_err = max(z_err, np.abs(zb - z[I_LE:I_TE + 1]).max())
    else:
        t1, t2 = t1_tip, t2_tip          # the strip above the tip continues the tip section
    lo = np.empty(NI)                    # theta at j = 0
    mid = np.empty(NI)                   # theta at j = JB
    e = np.ones(NI)                      # 1 near the blade, -> 0 far upstream/downstream
    lo[I_LE:I_TE + 1], mid[I_LE:I_TE + 1] = t2, t1
    zb = z[I_LE:I_TE + 1]
    tc = 0.5 * (t1 + t2)

    # upstream: camber-line extension relaxing to axial, strip width relaxing to uniform spacing
    slope = np.polyfit(zb[:4] - zb[0], tc[:4], 1)[0]
    iu = np.arange(0, I_LE)
    ee = np.exp(-(z[I_LE] - z[iu]) / LB)
    c = tc[0] - slope * LB * (1.0 - ee)
    w = (t1[0] - t2[0]) * ee + w_far * (1.0 - ee)
    lo[iu], mid[iu], e[iu] = c - 0.5 * w, c + 0.5 * w, ee

    # downstream
    slope = np.polyfit(zb[-4:] - zb[-1], tc[-4:], 1)[0]
    idn = np.arange(I_TE + 1, NI)
    ee = np.exp(-(z[idn] - z[I_TE]) / LB)
    c = tc[-1] + slope * LB * (1.0 - ee)
    w = (t1[-1] - t2[-1]) * ee + w_far * (1.0 - ee)
    lo[idn], mid[idn], e[idn] = c - 0.5 * w, c + 0.5 * w, ee

    for i in range(NI):
        width_s = mid[i] - lo[i]
        width_p = lo[i] + PITCH - mid[i]
        # spacing at j = 0, JB, NJ: DS_PITCH at the blade, relaxing to the uniform spacing far away.
        # Strip and passage share it, so cell sizes are continuous across j = JB.
        ds = e[i] * DS_PITCH + (1.0 - e[i]) * r[i] * PITCH / NJ
        f_s = clustered_n(JB, r[i] * width_s, ds, RATIO, "strip")
        f_p = clustered_n(NJ_PASS, r[i] * width_p, ds, RATIO, "passage")
        TH[i, :JB + 1, k] = lo[i] + f_s * width_s
        TH[i, JB:, k] = mid[i] + f_p * width_p

np.savez(MESH / "b2b.npz", TH=TH, JB=JB, NJ=NJ)

# ----------------------------------------------------------------------------- checks
RT = Rm[:, None, :] * TH                                   # arc length r*theta [cm]
dj = np.diff(RT, axis=1)
print(f"blade-to-blade grid: {NI} x {NJ + 1} x {NK} nodes (strip {JB} + passage {NJ_PASS} cells)")
print(f"blade z consistency with meridional grid: max |dz| = {z_err:.2e} cm")
print(f"min pitchwise spacing = {dj.min()*1e4:.1f} um, max = {dj.max():.3f} cm, any <= 0: {(dj <= 0).any()}")
# cell area in the (z, r*theta) plane: negative = folded cell
Zb = np.broadcast_to(Zm[:, None, :], RT.shape)
ax_ = Zb[1:, 1:] - Zb[:-1, :-1]
ay_ = RT[1:, 1:] - RT[:-1, :-1]
bx_ = Zb[:-1, 1:] - Zb[1:, :-1]
by_ = RT[:-1, 1:] - RT[1:, :-1]
area = 0.5 * (ax_ * by_ - ay_ * bx_)
print(f"min cell area / max cell area = {area.min():.2e} / {area.max():.2e} cm2, "
      f"folded cells: {(area <= 0).sum()}")
ratio_j = dj[:, 1:, :] / dj[:, :-1, :]               # ratio between cell j+1 and cell j
ratio_j = np.maximum(ratio_j, 1.0 / ratio_j)
r_pass = ratio_j[:, JB:, :].max()                    # pairs inside the passage
fluid = np.ones(ratio_j.shape, bool)                 # strip pairs and pairs across j = JB,
fluid[I_LE:I_TE + 1, :JB, :K_TIP] = False            # except those touching blade material
fluid[:, JB:, :] = False
r_strip = ratio_j[fluid].max()
print(f"max pitchwise growth ratio: passage = {r_pass:.3f}, fluid strip (gap, up/downstream) = {r_strip:.3f}")
print(f"geometric fallbacks (growth > {RATIO}): passage = {n_fallback['passage']}, "
      f"strip = {n_fallback['strip']} of {NI * NK} nodes")
# where is the largest strip ratio?
idx = np.unravel_index(np.argmax(np.where(fluid, ratio_j, 0.0)), ratio_j.shape)
print(f"largest fluid-strip ratio at i = {idx[0]}, j-pair = {idx[1]}, k = {idx[2]} "
      f"(LE i = {I_LE}, TE i = {I_TE}, tip k = {K_TIP}, JB = {JB})")
n_all = (NI - 1) * NJ * (NK - 1)
n_blade = (I_TE - I_LE) * JB * K_TIP
print(f"3D cells: {n_all} total - {n_blade} inside the blade = {n_all - n_blade} "
      f"(~{(n_all - n_blade)/1e6:.2f} M)")

# ----------------------------------------------------------------------------- figure
layers = [(0, "hub (k = 0)"), (K_TIP // 2, f"mid-span (k = {K_TIP // 2})"),
          (K_TIP, f"blade tip (k = {K_TIP})"), (NK - 1 - (NK - 1 - K_TIP) // 2, "middle of tip gap")]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for col, (k, name) in enumerate(layers):
    for row in range(2):
        ax = axes[row, col]
        z = Zm[:, k]
        y = RT[:, :, k]
        for j in range(NJ + 1):
            ax.plot(z, y[:, j], "-", color="0.6", lw=0.3)
        for i in range(NI):
            ax.plot(np.full(NJ + 1, z[i]), y[i, :], "-", color="0.6", lw=0.3)
        bz = z[I_LE:I_TE + 1]
        style = "k-" if k <= K_TIP else "k--"
        ax.plot(bz, y[I_LE:I_TE + 1, 0], style, lw=1.2)
        ax.plot(bz, y[I_LE:I_TE + 1, JB], style, lw=1.2)
        ax.plot(bz, y[I_LE:I_TE + 1, NJ], style, lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("z [cm]")
        if row == 0:
            ax.set_title(name)
            ax.set_xlim(z[I_LE] - 3.0, z[I_TE] + 3.0)
        else:
            ax.set_title(name + " - leading edge zoom")
            zc, yc = z[I_LE], y[I_LE, JB]
            ax.set_xlim(zc - 0.4, zc + 0.4)
            ax.set_ylim(yc - 0.4, yc + 0.4)
axes[0, 0].set_ylabel(r"$r\,\theta$ [cm]")
axes[1, 0].set_ylabel(r"$r\,\theta$ [cm]")
fig.tight_layout()
fig.savefig(FIG / "mesh_b2b.png", dpi=130)
print(f"figure: {FIG / 'mesh_b2b.png'}")
