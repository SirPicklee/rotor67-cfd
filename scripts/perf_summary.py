"""Average the turbomachinery performance of an SU2 run over the last N iterations.

Usage (from the project root):  python scripts/perf_summary.py cases/peak/history.csv [N]
Mass flow is per passage in SU2; the total is x22. m/m_choke uses the CFD choke flow.
"""
import sys

import pandas as pd

N_BLADES = 22
M_CHOKE_CFD = 34.17      # kg/s, second-order choke solution (outlet 115 kPa)
NASA = {"peak efficiency": (34.57, 0.989, 1.642, 0.930), "near stall": (32.31, 0.924, 1.728, 0.901)}

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
h = pd.read_csv(path)
h.columns = [c.strip().strip('"') for c in h.columns]
t = h.tail(n)
m = 0.5 * (t["MassFlowIn_1"] + t["MassFlowOut_1"]) * N_BLADES
pr = t["TotPressureOut_1"] / t["TotPressureIn_1"]
eta = t["TotTotEff"]
eta = eta / 100.0 if eta.mean() > 2 else eta
imb = (t["MassFlowOut_1"] - t["MassFlowIn_1"]) / t["MassFlowIn_1"] * 100

print(f"{path}: {len(h)} iterations, statistics over the last {len(t)}")
print(f"  mass flow      {m.mean():7.3f} kg/s  +- {m.std():.3f}    m/m_choke = {m.mean() / M_CHOKE_CFD:.4f}")
print(f"  pressure ratio {pr.mean():7.4f}       +- {pr.std():.4f}")
print(f"  efficiency     {eta.mean():7.4f}       +- {eta.std():.4f}   (min {eta.min():.4f}, max {eta.max():.4f})")
print(f"  in/out mass imbalance {imb.mean():+.2f} %")
print("NASA TP-2879 reference:")
for name, (mn, ratio, prn, etan) in NASA.items():
    print(f"  {name:16s} m = {mn:.2f} kg/s, m/m_choke = {ratio:.3f}, PR = {prn:.3f}, eta = {etan:.3f}")
