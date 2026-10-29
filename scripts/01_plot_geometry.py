"""Plot the NASA Rotor 67 geometry transcribed from NASA TP-2879 (Tables I and II).

Produces three figures in figures/:
  geometry_meridional.png    hub/casing flowpath with the 14 blade sections, LE and TE lines
  geometry_blade_to_blade.png  blade sections (hub, mid, tip) unwrapped as (m, r*theta)
  geometry_3d.png            3D view of both blade surfaces

Run from the project root:  python scripts/01_plot_geometry.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed in WSL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

N_BLADES = 22
TIP_GAP_CM = 0.1016  # cold tip clearance, NASA TP-2879

blade = pd.read_csv(DATA / "rotor67_blade_sections.csv", comment="#")
flow = pd.read_csv(DATA / "rotor67_flowpath.csv", comment="#")
sections = [g.sort_values("k") for _, g in blade.groupby("section")]
n_sec = len(sections)


def meridional_coordinate(z, r):
    """Arc length along the section in the (z, r) plane, starting at k = 1."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(z), np.diff(r)))])


# ---------------------------------------------------------------- summary
print(f"Blade sections: {n_sec}, points per section: {len(sections[0])}")
print(f"Flowpath points: {len(flow)}  (z = {flow.z.min():.2f} ... {flow.z.max():.2f} cm)")
print()
print(" sec   r_LE    r_TE   chord_m  max t   pitch_LE  solidity")
for g in sections:
    z, r = g.z.to_numpy(), g.r.to_numpy()
    m = meridional_coordinate(z, r)
    # true chord: distance between LE and TE points in 3D
    th_mean = 0.5 * (g.thsp1.to_numpy() + g.thsp2.to_numpy())
    x = r * np.cos(th_mean)
    y = r * np.sin(th_mean)
    chord = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2 + (z[-1] - z[0]) ** 2)
    r_mid = 0.5 * (r[0] + r[-1])
    pitch = 2 * np.pi * r_mid / N_BLADES
    print(f"{int(g.section.iloc[0]):4d}  {r[0]:6.2f}  {r[-1]:6.2f}  {m[-1]:6.2f}  "
          f"{g.tnpc.max():5.3f}   {2*np.pi*r[0]/N_BLADES:6.2f}   {chord/pitch:5.2f}")

# ---------------------------------------------------------------- meridional view
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(flow.z, flow.r_hub, "k-", lw=2, label="hub")
ax.plot(flow.z, flow.r_casing, "k-", lw=2, label="casing")
for g in sections:
    ax.plot(g.z, g.r, "-", color="tab:blue", lw=0.8)
le = np.array([[g.z.iloc[0], g.r.iloc[0]] for g in sections])
te = np.array([[g.z.iloc[-1], g.r.iloc[-1]] for g in sections])
ax.plot(le[:, 0], le[:, 1], "r-o", ms=3, label="leading edge")
ax.plot(te[:, 0], te[:, 1], "g-o", ms=3, label="trailing edge")
ax.axvspan(-1.374, 9.365, color="0.9", zorder=0, label="rotating hub")
ax.set_xlabel("z [cm]")
ax.set_ylabel("r [cm]")
ax.set_title("NASA Rotor 67 - meridional view (NASA TP-2879, Tables I and II)")
ax.set_aspect("equal")
ax.set_xlim(-6, 14)
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "geometry_meridional.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- blade-to-blade view
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, idx, name in zip(axes, [0, n_sec // 2, n_sec - 1], ["hub (sec 1)", f"mid (sec {n_sec//2+1})", "tip (sec 14)"]):
    g = sections[idx]
    z, r = g.z.to_numpy(), g.r.to_numpy()
    m = meridional_coordinate(z, r)
    r_ref = r.mean()
    pitch_ang = 2 * np.pi / N_BLADES
    for shift, alpha in [(0.0, 1.0), (pitch_ang, 0.35)]:  # show neighbouring blade too
        ax.plot(m, r_ref * (g.thsp1 + shift), "b-", alpha=alpha, label="surface 1" if shift == 0 else None)
        ax.plot(m, r_ref * (g.thsp2 + shift), "r-", alpha=alpha, label="surface 2" if shift == 0 else None)
    ax.set_title(name)
    ax.set_xlabel("m [cm]")
    ax.set_ylabel(r"$r_{ref}\,\theta$ [cm]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Blade sections unwrapped on surfaces of revolution (two adjacent blades)")
fig.tight_layout()
fig.savefig(FIG / "geometry_blade_to_blade.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3D view
Z = np.array([g.z.to_numpy() for g in sections])
R = np.array([g.r.to_numpy() for g in sections])
T1 = np.array([g.thsp1.to_numpy() for g in sections])
T2 = np.array([g.thsp2.to_numpy() for g in sections])
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
for T, c in [(T1, "tab:blue"), (T2, "tab:red")]:
    ax.plot_surface(R * np.cos(T), R * np.sin(T), Z, color=c, alpha=0.7, linewidth=0)
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_zlabel("z [cm]")
ax.set_title("NASA Rotor 67 blade (cold geometry)")
ax.view_init(elev=20, azim=-60)
ax.set_box_aspect((np.ptp(R * np.cos(T1)), np.ptp(R * np.sin(T1)), np.ptp(Z)))
fig.tight_layout()
fig.savefig(FIG / "geometry_3d.png", dpi=150)
plt.close(fig)

print()
print(f"Figures written to {FIG}")
