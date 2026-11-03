"""Locate the largest residuals and the lowest pressure/density in an SU2 volume CSV solution
and map those points onto the H-grid indices (i streamwise, j pitchwise, k spanwise).

Usage (from the project root):  python scripts/diag_residual.py cases/choke/diag.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import MESH  # noqa: E402

path = Path(sys.argv[1])
df = pd.read_csv(path)
df.columns = [c.strip().strip('"') for c in df.columns]
print(f"{path}: {len(df)} points, {len(df.columns)} columns")
print("columns:", ", ".join(df.columns))
pts = df[["x", "y", "z"]].to_numpy()

M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm, TH = M["Z"], M["R"], B["TH"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
JB, NJ = int(B["JB"]), int(B["NJ"])
NI, NJ1, NK = TH.shape
R3 = np.broadcast_to(Rm[:, None, :], TH.shape)
Z3 = np.broadcast_to(Zm[:, None, :], TH.shape)
X = np.stack([R3 * np.cos(TH), R3 * np.sin(TH), Z3], axis=-1).reshape(-1, 3) * 0.01
tree = cKDTree(X)


def where(i, j, k):
    s = "upstream" if i < I_LE else ("downstream" if i > I_TE else "blade row")
    s += ", " + ("strip" if j < JB else "passage")
    if j in (0, NJ):
        s += " [j=0/NJ]"
    if j == JB:
        s += " [j=JB surface 1]"
    s += ", " + (f"tip gap k={k}" if k > K_TIP else f"k={k}")
    if k == K_TIP:
        s += " [tip]"
    if k == 0:
        s += " [hub]"
    if k == NK - 1:
        s += " [casing]"
    if i == I_LE:
        s += " [LE]"
    if i == I_TE:
        s += " [TE]"
    return s


def report(title, values, order, n=12):
    print(f"\n=== {title}")
    d, idx = tree.query(pts[order[:n]])
    for rank_, (p, q, dist) in enumerate(zip(order[:n], idx, d)):
        i, rem = divmod(int(q), NJ1 * NK)
        j, k = divmod(rem, NK)
        print(f"{rank_ + 1:2d}  {values[p]: .3e}   i={i:3d} j={j:3d} k={k:3d}   {where(i, j, k)}"
              f"   (match {dist * 1e6:.1f} um)")


for name in [c for c in df.columns if c.lower().startswith("residual")]:
    v = np.abs(df[name].to_numpy())
    report(f"|{name}|  max {v.max():.3e}, median {np.median(v):.3e}", v, np.argsort(v)[::-1])

for name in ("Pressure", "Density", "Temperature"):
    if name in df.columns:
        v = df[name].to_numpy()
        report(f"lowest {name}  (min {v.min():.4g}, median {np.median(v):.4g})", v, np.argsort(v), n=6)
