"""Step 1 of the mesh: the meridional (z, r) grid of the Rotor 67 single-passage H-grid.

Index i runs streamwise (inlet -> LE -> TE -> outlet), index k spanwise (hub -> blade tip -> casing).
Every k-layer is a surface of revolution s = a * s_tip(z) + b, so the running blade tip
(casing offset by the 0.1016 cm clearance) is exactly a grid surface.

Output:
  mesh/meridional.npz        grid arrays for the next steps
  figures/mesh_meridional.png

Run from the project root:  python scripts/02_mesh_meridional.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import FIG, MESH, TIP_GAP, Blade, Flowpath, vinokur, wall_clustered  # noqa: E402

# ----------------------------------------------------------------------------- parameters
Z_IN, Z_OUT = -8.0, 16.0      # domain inlet / outlet [cm] (hub LE at z = 0)
NI_UP, NI_BLADE, NI_DOWN = 28, 96, 32
NK_GAP = 16                   # cells in the tip gap
DS_SPAN = 2.0e-3              # first cell height at hub, blade tip and casing [cm]
SPAN_RATIO = 1.25             # spanwise growth ratio away from the walls
SPAN_DMAX = 0.30              # largest spanwise cell height [cm] (~2 % span)
DU_LE, DU_TE = 0.004, 0.006   # first / last chordwise spacing as fraction of the chord parameter
NJ_ESTIMATE = 72              # pitchwise cells (next step), only for the size estimate here

MESH.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
fp = Flowpath()
bl = Blade(fp)

# ----------------------------------------------------------------------------- span layers
span_mid = fp.cas(4.5) - fp.hub(4.5)
gap_mid = TIP_GAP
eta = wall_clustered(span_mid, DS_SPAN, SPAN_RATIO, SPAN_DMAX)
NK_BLADE = len(eta) - 1                  # cells hub->tip (follows from the spacing inputs)
g = vinokur(NK_GAP, DS_SPAN / gap_mid, DS_SPAN / gap_mid)
layer_a = np.concatenate([eta, 1.0 - g[1:]])
layer_b = np.concatenate([np.zeros_like(eta), g[1:]])
NK = len(layer_a)                       # number of k-nodes
K_TIP = NK_BLADE                         # k index of the blade tip layer

# ----------------------------------------------------------------------------- streamwise
u_blade = vinokur(NI_BLADE, DU_LE, DU_TE)
z_tip_blade, _, _, _ = bl.on_layer(1.0, 0.0, u_blade)
NI = NI_UP + NI_BLADE + NI_DOWN + 1
I_LE, I_TE = NI_UP, NI_UP + NI_BLADE

Zm = np.zeros((NI, NK))
Rm = np.zeros((NI, NK))
for k in range(NK):
    a, b = layer_a[k], layer_b[k]
    if k <= K_TIP:
        zb, _, _, _ = bl.on_layer(a, b, u_blade)
    else:
        zb = z_tip_blade.copy()                    # gap layers continue the tip i-lines
    # upstream: inlet -> LE, last spacing matches the first blade spacing
    L = zb[0] - Z_IN
    t = vinokur(NI_UP, 1.5 / NI_UP, (zb[1] - zb[0]) / L)
    z_up = Z_IN + L * t[:-1]
    # downstream: TE -> outlet, first spacing matches the last blade spacing
    L = Z_OUT - zb[-1]
    t = vinokur(NI_DOWN, (zb[-1] - zb[-2]) / L, 1.5 / NI_DOWN)
    z_dn = zb[-1] + L * t[1:]
    z = np.concatenate([z_up, zb, z_dn])
    s = a * fp.s_tip(z) + b
    Zm[:, k] = z
    Rm[:, k] = fp.r_of(z, s)

np.savez(MESH / "meridional.npz", Z=Zm, R=Rm, layer_a=layer_a, layer_b=layer_b, u_blade=u_blade,
         I_LE=I_LE, I_TE=I_TE, K_TIP=K_TIP)

# ----------------------------------------------------------------------------- checks
dz = np.diff(Zm, axis=0)
dr = np.diff(Rm, axis=1)
print(f"meridional grid: {NI} x {NK} nodes  (i: {NI_UP}+{NI_BLADE}+{NI_DOWN} cells, "
      f"k: {NK_BLADE} blade + {NK_GAP} gap cells)")
print(f"LE at i = {I_LE}, TE at i = {I_TE}, blade tip at k = {K_TIP}")
print(f"min dz = {dz.min():.4f} cm, max dz = {dz.max():.4f} cm, any dz <= 0: {(dz <= 0).any()}")
print(f"min dr = {dr.min():.5f} cm, max dr = {dr.max():.4f} cm, any dr <= 0: {(dr <= 0).any()}")
ratio_k = dr[:, 1:] / dr[:, :-1]
print(f"max spanwise growth ratio = {max(ratio_k.max(), 1 / ratio_k.min()):.3f}")
gap_now = Rm[:, -1] - Rm[:, K_TIP]
print(f"radial tip gap over the blade: {gap_now[I_LE:I_TE+1].min():.4f} ... "
      f"{gap_now[I_LE:I_TE+1].max():.4f} cm")
print(f"blade tip LE/TE z: {Zm[I_LE, K_TIP]:.3f} / {Zm[I_TE, K_TIP]:.3f} cm, "
      f"hub LE/TE z: {Zm[I_LE, 0]:.3f} / {Zm[I_TE, 0]:.3f} cm")
n_cells = (NI - 1) * (NK - 1) * NJ_ESTIMATE
print(f"estimated 3D size with {NJ_ESTIMATE} pitchwise cells: ~{n_cells/1e3:.0f}k hexahedra")

# ----------------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})
for ax in axes:
    for k in range(0, NK, 2):
        ax.plot(Zm[:, k], Rm[:, k], "-", color="0.55", lw=0.4)
    for i in range(0, NI, 2):
        ax.plot(Zm[i, :], Rm[i, :], "-", color="0.55", lw=0.4)
    ax.plot(Zm[:, 0], Rm[:, 0], "k-", lw=1.5)
    ax.plot(Zm[:, -1], Rm[:, -1], "k-", lw=1.5)
    ax.plot(Zm[I_LE:I_TE + 1, K_TIP], Rm[I_LE:I_TE + 1, K_TIP], "b-", lw=1.5, label="blade tip")
    ax.plot(Zm[I_LE, :K_TIP + 1], Rm[I_LE, :K_TIP + 1], "r-", lw=1.5, label="leading edge")
    ax.plot(Zm[I_TE, :K_TIP + 1], Rm[I_TE, :K_TIP + 1], "g-", lw=1.5, label="trailing edge")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("r [cm]")
    ax.set_aspect("equal")
axes[0].set_title("Meridional grid (every 2nd line)")
axes[0].legend(fontsize=8, loc="lower right")
zt, rt = Zm[I_LE, K_TIP], Rm[I_LE, K_TIP]
axes[1].set_xlim(zt - 0.6, zt + 0.6)
axes[1].set_ylim(rt - 0.5, rt + 0.25)
axes[1].set_title("Zoom: blade tip leading edge and tip gap")
fig.tight_layout()
fig.savefig(FIG / "mesh_meridional.png", dpi=150)
print(f"figure: {FIG / 'mesh_meridional.png'}")
