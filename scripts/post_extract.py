"""Map an SU2 legacy VTK volume solution onto the structured H-grid (i, j, k) and save it as npz.

Usage (from the project root):  python scripts/post_extract.py cases/peak/post.vtk cases/peak/field.npz
Saved arrays have shape (NI, NJ+1, NK) [i streamwise, j pitchwise, k spanwise]; nodes inside the
blade are NaN. SU2's rotating frame stores the ABSOLUTE velocity; the relative velocity is
V - Grid_Velocity. Units: SI.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import MESH  # noqa: E402

DT = {"float": ">f4", "double": ">f8", "int": ">i4", "unsigned_int": ">u4", "long": ">i8", "vtkidtype": ">i8"}
buf = open(sys.argv[1], "rb").read()
pos = 0


def line():
    global pos
    e = buf.index(b"\n", pos)
    s = buf[pos:e].decode(errors="replace").strip()
    pos = e + 1
    return s


def arr(n, typ):
    global pos
    dt = np.dtype(DT[typ.lower()])
    a = np.frombuffer(buf, dt, n, pos).astype(np.float64)
    pos += n * dt.itemsize
    while pos < len(buf) and buf[pos:pos + 1] in (b"\n", b"\r"):
        pos += 1
    return a


for _ in range(4):
    line()
npts, data, pts = 0, {}, None
while pos < len(buf):
    s = line()
    if not s:
        continue
    t = s.split()
    key = t[0].upper()
    if key == "POINTS":
        npts = int(t[1])
        pts = arr(3 * npts, t[2]).reshape(npts, 3)
    elif key == "CELLS":
        arr(int(t[2]), "int")
    elif key == "CELL_TYPES":
        arr(int(t[1]), "int")
    elif key == "POINT_DATA":
        continue
    elif key == "CELL_DATA":
        break
    elif key == "SCALARS":
        nc = int(t[3]) if len(t) > 3 else 1
        line()
        data[t[1]] = arr(npts * nc, t[2]).reshape(npts, nc)[:, 0]
    elif key == "VECTORS":
        data[t[1]] = arr(3 * npts, t[2]).reshape(npts, 3)
    elif key == "FIELD":
        for _ in range(int(t[2])):
            name, nc, nt, typ = line().split()
            a = arr(int(nc) * int(nt), typ).reshape(int(nt), int(nc))
            data[name] = a[:, 0] if int(nc) == 1 else a
    else:
        break
print("fields:", ", ".join(data))

# structured node coordinates (same construction as 04_mesh_write.py)
M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm, TH = M["Z"], M["R"], B["TH"]
shape = TH.shape
R3 = np.broadcast_to(Rm[:, None, :], shape)
Z3 = np.broadcast_to(Zm[:, None, :], shape)
X = np.stack([R3 * np.cos(TH), R3 * np.sin(TH), Z3], axis=-1).reshape(-1, 3) * 0.01
dist, idx = cKDTree(pts).query(X)
inside = dist > 1e-7                       # structured nodes without an SU2 point (blade interior)
print(f"mapped {np.sum(~inside)} of {len(X)} structured nodes, max match distance "
      f"{dist[~inside].max():.1e} m, unmapped (blade interior) {inside.sum()}")


def grid(a):
    g = a[idx].astype(np.float32)
    g[inside] = np.nan
    return g.reshape(shape + a.shape[1:])


out = {"rho": grid(data["Density"]), "p": grid(data["Pressure"]), "T": grid(data["Temperature"]),
       "V": grid(data["Velocity"]), "Vgrid": grid(data["Grid_Velocity"]),
       "mut": grid(data["Eddy_Viscosity"]), "mu": grid(data["Laminar_Viscosity"])}
np.savez_compressed(sys.argv[2], **out)

# sanity check: absolute flow at the inlet plane should be nearly axial
V, Vg = out["V"][0], out["Vgrid"][0]
th = TH[0]
vt = -V[..., 0] * np.sin(th) + V[..., 1] * np.cos(th)
gt = -Vg[..., 0] * np.sin(th) + Vg[..., 1] * np.cos(th)
print(f"inlet plane: mean axial V = {np.nanmean(V[..., 2]):.1f} m/s, mean tangential V = {np.nanmean(vt):.2f} m/s, "
      f"mean blade speed (grid tangential) = {np.nanmean(gt):.1f} m/s")
print("written", sys.argv[2])
