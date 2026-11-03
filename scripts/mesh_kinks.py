"""Find kinks in the H-grid: turning angle of grid lines between consecutive segments,
measured in the (z, r*theta) blade-to-blade plane. Large turning angles mean skewed cells."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import MESH  # noqa: E402

M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm, TH = M["Z"], M["R"], B["TH"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
JB, NJ = int(B["JB"]), int(B["NJ"])
Z = np.broadcast_to(Zm[:, None, :], TH.shape)
Y = Rm[:, None, :] * TH


def turning(dz, dy, axis):
    a = np.arctan2(np.take(dy, range(1, dy.shape[axis]), axis), np.take(dz, range(1, dz.shape[axis]), axis))
    b = np.arctan2(np.take(dy, range(0, dy.shape[axis] - 1), axis), np.take(dz, range(0, dz.shape[axis] - 1), axis))
    return np.degrees(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi))


# j = const lines along i (streamwise lines)
ti = turning(np.diff(Z, axis=0), np.diff(Y, axis=0), 0)       # (NI-2, NJ+1, NK), node i+1
# skew angle of cells: angle between i-direction and j-direction edges at each node
ez, ey = np.diff(Z, axis=0)[:, :-1, :], np.diff(Y, axis=0)[:, :-1, :]
fz, fy = np.diff(Z, axis=1)[:-1, :, :], np.diff(Y, axis=1)[:-1, :, :]
cosang = (ez * fz + ey * fy) / np.hypot(ez, ey) / np.hypot(fz, fy)
skew = np.degrees(np.arccos(np.clip(cosang, -1, 1)))           # 90 = orthogonal


def region(i):
    return "upstream" if i < I_LE else ("downstream" if i > I_TE else "blade")


print(f"LE i = {I_LE}, TE i = {I_TE}, JB = {JB}, NJ = {NJ}, tip k = {K_TIP}")
for name, arr, off in [("streamwise line turning angle [deg]", ti, 1)]:
    order = np.argsort(arr, axis=None)[::-1]
    print(f"\n=== {name}: max {arr.max():.1f}, 99.9 % {np.percentile(arr, 99.9):.1f}")
    seen = set()
    for q in order:
        i, j, k = np.unravel_index(q, arr.shape)
        key = (i + off, k // 10)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {arr[i, j, k]:5.1f} deg  at i={i + off:3d} j={j:3d} k={k:3d}  ({region(i + off)})")
        if len(seen) == 10:
            break
dev = np.abs(skew - 90)
order = np.argsort(dev, axis=None)[::-1]
print(f"\n=== cell corner angle deviation from 90 deg: max {dev.max():.1f}, 99.9 % {np.percentile(dev, 99.9):.1f}")
seen = set()
for q in order:
    i, j, k = np.unravel_index(q, dev.shape)
    key = (i, k // 10)
    if key in seen:
        continue
    seen.add(key)
    print(f"  corner angle {skew[i, j, k]:5.1f} deg  at i={i:3d} j={j:3d} k={k:3d}  ({region(i)})")
    if len(seen) == 10:
        break
