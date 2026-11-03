"""Step 2 of the mesh: the blade-to-blade (pitchwise) distribution on every k-layer.

Topology (H-grid, one blade pitch per layer, theta increasing with j):

  j = 0 .......... JB ........................ NJ
  surface 2        surface 1                   surface 2 of the next blade (theta + PITCH)
  |-- blade strip --|-------- passage ----------|

Inside the blade (I_LE <= i <= I_TE, k < K_TIP) the strip cells are blade material and are
removed in step 3. Above the tip they are fluid (tip leakage).

Distribution, as fractions F(i, j) of the pitch measured from the j = 0 line:
  blade row : wall-clustered strip and passage distributions (first cell DS_PITCH, growth <= RATIO),
              smoothed along i so that integer changes in the number of growth cells do not
              create kinks
  up/down   : F = w * F(edge) + (1 - w) * j / NJ, with w a cosine in the cell index
              (1 at the edge, 0 at inlet/outlet, zero slope at both ends). Strip and passage are
              blended together, so cell sizes stay continuous across j = JB.
  j = 0 line: the strip is centred on the camber line, extended from the edge with the edge slope
              and relaxing to axial over LB.

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
LB = 1.5               # camber-line extension: decay length to axial [cm]
N_SMOOTH = 30          # 1-2-1 smoothing passes of the blade-row distribution along i
TE_TAPER_U = 0.05      # trailing-edge taper: last 5 % of the chord parameter u
TE_THICK = 1.0         # trailing-edge taper disabled (1.0 cm > any TE): true NASA geometry
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
    # growth ratio cannot be respected with n cells: two-sided geometric with the smallest ratio
    n_fallback[tag] += 1
    expo = np.minimum(np.arange(n), n - 1 - np.arange(n))
    q = brentq(lambda q: (ds * q ** expo).sum() - length, 1.0 + 1e-12, 20.0)
    x = np.concatenate([[0.0], np.cumsum(ds * q ** expo)])
    return x / x[-1]


def smooth_i(f, n_pass):
    """1-2-1 smoothing along axis 0, end rows included (zero-gradient ends). Columns 0 and -1
    are 0 and 1 in every row, so the wall and j = JB nodes stay on the blade surfaces."""
    f = f.copy()
    for _ in range(n_pass):
        g = np.concatenate([f[:1], f, f[-1:]])
        f = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
    return f


# ----------------------------------------------------------------------------- inputs
M = np.load(MESH / "meridional.npz")
Zm, Rm = M["Z"], M["R"]
layer_a, layer_b, u_blade = M["layer_a"], M["layer_b"], M["u_blade"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
NI, NK = Zm.shape
NB = I_TE - I_LE + 1

fp_ = Flowpath()
bl = Blade(fp_)
_, _, t1_tip, t2_tip = bl.on_layer(1.0, 0.0, u_blade)

TH = np.zeros((NI, NJ + 1, NK))
uni = np.arange(NJ + 1) / NJ
z_err = 0.0
iu = np.arange(0, I_LE)
idn = np.arange(I_TE + 1, NI)
LW = 2.0               # blend length of the edge distribution towards uniform [cm]

for k in range(NK):
    z, r = Zm[:, k], Rm[:, k]
    if k <= K_TIP:
        zb, _, t1, t2 = bl.on_layer(layer_a[k], layer_b[k], u_blade)
        z_err = max(z_err, np.abs(zb - z[I_LE:I_TE + 1]).max())
    else:
        t1, t2 = t1_tip, t2_tip          # the strip above the tip continues the tip section
    zb, rb = z[I_LE:I_TE + 1], r[I_LE:I_TE + 1]
    # trailing-edge taper: thickness reduced quadratically over the last TE_TAPER_U of u,
    # symmetric about the camber line, to a TE_THICK blunt face (H-grid cannot model the
    # TE circle; a full-thickness blunt face gives a 90 deg corner expansion that diverges)
    tcam = 0.5 * (t1 + t2)
    half = 0.5 * (t1 - t2)
    s_t = np.clip((u_blade - (1.0 - TE_TAPER_U)) / TE_TAPER_U, 0.0, 1.0)
    f_end = min(1.0, TE_THICK / (rb[-1] * 2.0 * half[-1]))
    fac = 1.0 - (1.0 - f_end) * s_t ** 2
    t1, t2 = tcam + half * fac, tcam - half * fac
    ws = t1 - t2                         # strip (blade) width [rad]
    wp = PITCH - ws                      # passage width [rad]

    # blade row: clustered distributions, smoothed along i
    # wall spacing: DS_PITCH, but never larger than a uniform strip cell (thin tapered TE)
    ds_b = np.minimum(DS_PITCH, rb * ws / JB)
    fs = np.array([clustered_n(JB, rb[n] * ws[n], ds_b[n], RATIO, "strip") for n in range(NB)])
    fq = np.array([clustered_n(NJ_PASS, rb[n] * wp[n], ds_b[n], RATIO, "passage") for n in range(NB)])
    fs, fq = smooth_i(fs, N_SMOOTH), smooth_i(fq, N_SMOOTH)
    F = np.empty((NI, NJ + 1))
    lo = np.empty(NI)
    F[I_LE:I_TE + 1, :JB + 1] = ws[:, None] * fs / PITCH
    F[I_LE:I_TE + 1, JB:] = (ws[:, None] + wp[:, None] * fq) / PITCH
    lo[I_LE:I_TE + 1] = t2
    tc = 0.5 * (t1 + t2)

    # blend weights: Gaussian in the physical distance from the edge, so the tiny edge cells
    # see (almost) no change and the distribution relaxes smoothly towards inlet/outlet
    w_up = np.exp(-((z[I_LE] - z[iu]) / LW) ** 2)
    w_dn = np.exp(-((z[idn] - z[I_TE]) / LW) ** 2)

    # upstream
    F[iu] = w_up[:, None] * F[I_LE] + (1.0 - w_up)[:, None] * uni
    slope = np.polyfit(zb[:4] - zb[0], tc[:4], 1)[0]
    c = tc[0] - slope * LB * (1.0 - np.exp(-(z[I_LE] - z[iu]) / LB))
    lo[iu] = c - 0.5 * PITCH * F[iu, JB]

    # downstream
    F[idn] = w_dn[:, None] * F[I_TE] + (1.0 - w_dn)[:, None] * uni
    slope = np.polyfit(zb[-4:] - zb[-1], tc[-4:], 1)[0]
    c = tc[-1] + slope * LB * (1.0 - np.exp(-(z[idn] - z[I_TE]) / LB))
    lo[idn] = c - 0.5 * PITCH * F[idn, JB]

    TH[:, :, k] = lo[:, None] + PITCH * F

np.savez(MESH / "b2b.npz", TH=TH, JB=JB, NJ=NJ)

# ----------------------------------------------------------------------------- checks
RT = Rm[:, None, :] * TH
dj = np.diff(RT, axis=1)
print(f"blade-to-blade grid: {NI} x {NJ + 1} x {NK} nodes (strip {JB} + passage {NJ_PASS} cells)")
print(f"blade z consistency with meridional grid: max |dz| = {z_err:.2e} cm")
print(f"min pitchwise spacing = {dj.min()*1e4:.1f} um, max = {dj.max():.3f} cm, any <= 0: {(dj <= 0).any()}")
Zb = np.broadcast_to(Zm[:, None, :], RT.shape)
ax_ = Zb[1:, 1:] - Zb[:-1, :-1]
ay_ = RT[1:, 1:] - RT[:-1, :-1]
bx_ = Zb[:-1, 1:] - Zb[1:, :-1]
by_ = RT[:-1, 1:] - RT[1:, :-1]
area = 0.5 * (ax_ * by_ - ay_ * bx_)
print(f"min cell area / max cell area = {area.min():.2e} / {area.max():.2e} cm2, folded cells: {(area <= 0).sum()}")
ratio_j = dj[:, 1:, :] / dj[:, :-1, :]
ratio_j = np.maximum(ratio_j, 1.0 / ratio_j)
fluid = np.ones(ratio_j.shape, bool)
fluid[I_LE:I_TE + 1, :JB, :K_TIP] = False
print(f"max pitchwise growth ratio: passage = {ratio_j[:, JB:, :].max():.3f}, "
      f"fluid strip and j = JB = {ratio_j[fluid & (np.arange(NJ - 1)[None, :, None] < JB)].max():.3f}")
print(f"geometric fallbacks: passage = {n_fallback['passage']}, strip = {n_fallback['strip']}")
n_all = (NI - 1) * NJ * (NK - 1)
n_blade = (I_TE - I_LE) * JB * K_TIP
print(f"3D cells: {n_all - n_blade} (~{(n_all - n_blade)/1e6:.2f} M)")

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
        for jj in (0, JB, NJ):
            ax.plot(bz, y[I_LE:I_TE + 1, jj], style, lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("z [cm]")
        if row == 0:
            ax.set_title(name)
            ax.set_xlim(z[I_LE] - 3.0, z[I_TE] + 3.0)
        else:
            ax.set_title(name + " - trailing edge zoom")
            zc, yc = z[I_TE], y[I_TE, JB]
            ax.set_xlim(zc - 0.6, zc + 0.6)
            ax.set_ylim(yc - 0.6, yc + 0.6)
fig.tight_layout()
fig.savefig(FIG / "mesh_b2b.png", dpi=130)
print(f"figure: {FIG / 'mesh_b2b.png'}")
