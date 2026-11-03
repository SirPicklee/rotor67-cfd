"""Compare the CFD with NASA TP-2879: speedline and spanwise profiles at station 2.

Station 2 is z = 11.011 cm (rotor hub LE at z = 0). Pitchwise averages are mass-weighted.
Run from the project root:  python scripts/05_compare_nasa.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import FIG, MESH, N_BLADES, ROOT  # noqa: E402

GAM, RGAS = 1.4, 287.058
CP = GAM * RGAS / (GAM - 1)
PREF, TREF = 101325.0, 288.15
Z_ST2 = 11.011            # cm
M_CHOKE = 34.17           # kg/s, CFD choke flow (outlet 115 kPa)
NASA_DATA = Path.home() / "transonic-fan" / "data"
CASES = {"choke": "choke", "peak efficiency": "peak", "near stall": "stall"}
NASA_R2 = np.array([23.2258, 22.0320, 20.8280, 19.6164, 18.3947, 17.1526, 15.8902, 14.6126, 13.3299])
NASA_A2 = np.array([241.362, 165.832, 158.120, 150.038, 142.327, 134.895, 126.813, 117.615, 138.240])

M = np.load(MESH / "meridional.npz")
TH = np.load(MESH / "b2b.npz")["TH"]
Zm, Rm = M["Z"], M["R"]
NI, NJ1, NK = TH.shape


def station(f, z0):
    """Interpolate a case to the plane z = z0 and return pitch-averaged spanwise profiles."""
    out = {k: np.zeros(NK) for k in ("r", "pt", "tt", "ang", "w")}
    tot = {"m": 0.0, "pt": 0.0, "tt": 0.0}
    for k in range(NK):
        i = np.searchsorted(Zm[:, k], z0) - 1
        w = (z0 - Zm[i, k]) / (Zm[i + 1, k] - Zm[i, k])
        g = lambda a: (1 - w) * a[i, :, k] + w * a[i + 1, :, k]  # noqa: E731
        r = ((1 - w) * Rm[i, k] + w * Rm[i + 1, k]) * 0.01
        th = g(TH)
        rho, p, T = g(f["rho"]), g(f["p"]), g(f["T"])
        V = (1 - w) * f["V"][i, :, k] + w * f["V"][i + 1, :, k]
        vz = V[:, 2]
        vr = V[:, 0] * np.cos(th) + V[:, 1] * np.sin(th)
        vt = -V[:, 0] * np.sin(th) + V[:, 1] * np.cos(th)
        v2 = (V ** 2).sum(axis=1)
        tt = T + v2 / (2 * CP)
        pt = p * (tt / T) ** (GAM / (GAM - 1))
        dth = np.gradient(th)                       # pitchwise node widths (periodic ends: half cells)
        dth[0] = dth[-1] = 0.5 * (th[1] - th[0] + th[-1] - th[-2])
        dm = rho * vz * r * dth                     # per unit radial height
        mk = dm.sum()
        out["r"][k] = r * 100
        out["pt"][k] = (dm * pt).sum() / mk / PREF
        out["tt"][k] = (dm * tt).sum() / mk / TREF
        out["ang"][k] = np.degrees(np.arctan2(-(dm * vt).sum(), (dm * np.hypot(vz, vr)).sum()))
        out["w"][k] = mk
    dr = np.gradient(out["r"] * 0.01)
    ok = out["w"] > 0                              # wall layers carry no mass flow (0/0 otherwise)
    mflow = np.where(ok, out["w"] * dr, 0.0)
    for q in ("pt", "tt", "ang"):
        out[q][~ok] = np.nan
    tot["m"] = mflow.sum() * N_BLADES
    tot["pt"] = np.nansum(mflow * out["pt"]) / mflow.sum()
    tot["tt"] = np.nansum(mflow * out["tt"]) / mflow.sum()
    out["eta"] = (out["pt"] ** ((GAM - 1) / GAM) - 1) / (out["tt"] - 1)
    tot["eta"] = (tot["pt"] ** ((GAM - 1) / GAM) - 1) / (tot["tt"] - 1)
    # NASA-style average (TP-2879 table IV): profile sampled at the 9 design-streamline radii,
    # weighted by local mass flux x NASA area element (endwall blockage excluded, as in the test)
    flux = np.where(ok, out["w"] / np.maximum(out["r"] * 0.01 * (2 * np.pi / N_BLADES), 1e-12), np.nan)
    o = np.argsort(out["r"][ok])
    rr = out["r"][ok][o]
    f_i = lambda q: np.interp(NASA_R2, rr, q[ok][o])  # noqa: E731
    wgt = f_i(flux) * NASA_A2
    tot["pt_n"] = (wgt * f_i(out["pt"])).sum() / wgt.sum()
    tot["tt_n"] = (wgt * f_i(out["tt"])).sum() / wgt.sum()
    tot["eta_n"] = (tot["pt_n"] ** ((GAM - 1) / GAM) - 1) / (tot["tt_n"] - 1)
    return out, tot


def history(path, n=1000):
    h = pd.read_csv(path)
    h.columns = [c.strip().strip('"') for c in h.columns]
    t = h.tail(n)
    m = 0.5 * (t.MassFlowIn_1 + t.MassFlowOut_1).mean() * N_BLADES
    eta = t.TotTotEff.mean()
    return m, (t.TotPressureOut_1 / t.TotPressureIn_1).mean(), eta / 100 if eta > 2 else eta


# ----------------------------------------------------------------------------- data
nasa = pd.read_csv(NASA_DATA / "rotor67_overall.csv", comment="#")
nasa2 = pd.read_csv(NASA_DATA / "rotor67_station2.csv", comment="#")
speed = [("115 kPa (choke)", None, (M_CHOKE, 1.519, 0.890)),
         ("121 kPa", "cases/peak/history_peak1.csv", None),
         ("123 kPa", "cases/peak/history_peak3.csv", None),
         ("126 kPa", "cases/peak/history_peak2.csv", None),
         ("128.5 kPa", "cases/stall/history_stall3.csv", None)]
line = []
for name, path, fixed in speed:
    m, pr, eta = fixed if fixed else history(ROOT / path)
    line.append((name, m / M_CHOKE, pr, eta))
st2 = {}
for name, d in CASES.items():
    f = np.load(ROOT / "cases" / d / "field.npz")
    st2[name] = station(f, Z_ST2)

# ----------------------------------------------------------------------------- table
print("CFD speedline (SU2 mixed-out values at the domain outlet, z = 16 cm):")
for name, mr, pr, eta in line:
    print(f"  {name:16s} m/m_choke {mr:.3f}  PR {pr:.4f}  eta {eta:.4f}")
print("CFD at station 2 (z = 11.011 cm, mass-averaged):")
for name, (_, tot) in st2.items():
    print(f"  {name:16s} m = {tot['m']:.2f} kg/s  m/m_choke {tot['m']/M_CHOKE:.3f}  "
          f"PR {tot['pt']:.4f}  TR {tot['tt']:.4f}  eta {tot['eta']:.4f}   |  NASA-style (9 radii): "
          f"PR {tot['pt_n']:.4f}  TR {tot['tt_n']:.4f}  eta {tot['eta_n']:.4f}")
print("NASA TP-2879 (station 2):")
for _, row in nasa.iterrows():
    print(f"  {row.point:16s} m = {row.orifice_flow_kg_s:.2f} kg/s  m/m_choke {row.orifice_flow_kg_s/34.96:.3f}  "
          f"PR {row.pressure_ratio:.4f}  TR {row.temperature_ratio:.4f}  eta {row.efficiency:.4f}")

# ----------------------------------------------------------------------------- speedline figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
mr = [l[1] for l in line]
ax[0].plot(mr, [l[2] for l in line], "o-", color="tab:blue", label="CFD, SU2 outlet (z = 16 cm)")
ax[1].plot(mr, [l[3] for l in line], "o-", color="tab:blue", label="CFD, SU2 outlet (z = 16 cm)")
for name, (_, tot) in st2.items():
    ax[0].plot(tot["m"] / M_CHOKE, tot["pt"], "s", mfc="none", color="tab:blue", ms=9)
    ax[1].plot(tot["m"] / M_CHOKE, tot["eta"], "s", mfc="none", color="tab:blue", ms=9)
    ax[0].plot(tot["m"] / M_CHOKE, tot["pt_n"], "s", color="tab:orange", ms=7)
    ax[1].plot(tot["m"] / M_CHOKE, tot["eta_n"], "s", color="tab:orange", ms=7)
ax[0].plot([], [], "s", mfc="none", color="tab:blue", label="CFD, station 2 (z = 11 cm)")
ax[1].plot([], [], "s", mfc="none", color="tab:blue", label="CFD, station 2 (z = 11 cm)")
ax[0].plot([], [], "s", color="tab:orange", label="CFD, station 2, NASA radii/areas")
ax[1].plot([], [], "s", color="tab:orange", label="CFD, station 2, NASA radii/areas")
for _, row in nasa.iterrows():
    for a, v in ((ax[0], row.pressure_ratio), (ax[1], row.efficiency)):
        a.plot(row.orifice_flow_kg_s / 34.96, v, "k*", ms=13)
ax[0].plot([], [], "k*", ms=13, label="NASA TP-2879")
ax[1].plot([], [], "k*", ms=13, label="NASA TP-2879")
ax[0].set_ylabel("total pressure ratio")
ax[1].set_ylabel("adiabatic efficiency")
for a in ax:
    a.set_xlabel("m / m_choke  (CFD: 34.17 kg/s, NASA: 34.96 kg/s)")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
ax[0].annotate("CFD numerical stall\nbelow ~0.955", xy=(0.956, line[-1][2]), xytext=(0.925, 1.56),
               arrowprops=dict(arrowstyle="->"), fontsize=8)
fig.suptitle("NASA Rotor 67, 100 % speed: CFD speedline vs. measurement")
fig.tight_layout()
fig.savefig(FIG / "speedline.png", dpi=150)

# ----------------------------------------------------------------------------- spanwise figure
fig, ax = plt.subplots(2, 4, figsize=(16, 9), sharey=True)
qty = [("pt", "p0_ratio", "total pressure / p_ref"), ("tt", "t0_ratio", "total temperature / T_ref"),
       ("eta", None, "adiabatic efficiency"), ("ang", "flow_angle_deg", "abs. flow angle [deg]")]
for row, (name, key) in enumerate((("peak efficiency", "peak_efficiency"), ("near stall", "near_stall"))):
    prof, _ = st2[name]
    n = nasa2[nasa2.point == key]
    n_eta = (n.p0_ratio ** ((GAM - 1) / GAM) - 1) / (n.t0_ratio - 1)
    for col, (q, nk, lab) in enumerate(qty):
        a = ax[row, col]
        a.plot(prof[q], prof["r"], "-", color="tab:blue", label="CFD")
        a.plot(n[nk] if nk else n_eta, n.radius_cm, "ko", mfc="none", label="NASA")
        a.set_xlabel(lab)
        a.grid(alpha=0.3)
        if col == 0:
            a.set_ylabel(f"{name}\nradius [cm]")
    ax[row, 0].legend(fontsize=8)
ax[0, 2].set_xlim(0.6, 1.0)
ax[1, 2].set_xlim(0.6, 1.0)
fig.suptitle("Rotor exit (station 2, z = 11.011 cm): CFD vs. NASA TP-2879, mass-averaged pitchwise")
fig.tight_layout()
fig.savefig(FIG / "spanwise_station2.png", dpi=150)
print(f"figures: {FIG / 'speedline.png'}, {FIG / 'spanwise_station2.png'}")
