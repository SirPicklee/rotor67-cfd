"""Step 3 of the mesh: assemble the 3D hexahedral mesh, tag the boundaries, write SU2 and VTK files.

Index directions: i streamwise (z), j pitchwise (theta), k spanwise (r).
(k, j, i) ~ (r, theta, z) is right-handed, so the hexahedra below have positive volume.

Boundary markers
  BLADE      pressure/suction surfaces, flat leading/trailing edge faces, blade tip
  HUB_ROT    rotating hub (face centroid within z = -1.374 ... 9.365 cm)
  HUB_STAT   stationary hub upstream and downstream
  SHROUD     casing
  INFLOW     inlet plane, OUTFLOW outlet plane
  PER1       periodic face at j = 0, PER2 at j = NJ (= PER1 rotated by +360/22 deg about z)

Output (coordinates in metres):
  mesh/rotor67.su2              SU2 native mesh
  mesh/rotor67_volume.vtk       volume mesh for ParaView (cell data: volume)
  mesh/rotor67_boundaries.vtk   boundary faces for ParaView (cell data: marker)

Run from the project root:  python scripts/04_mesh_write.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r67geom import MESH, N_BLADES, PITCH, Z_ROTATING_HUB, Flowpath  # noqa: E402

CM_TO_M = 0.01
t0 = time.time()

# ----------------------------------------------------------------------------- load grid
M = np.load(MESH / "meridional.npz")
B = np.load(MESH / "b2b.npz")
Zm, Rm = M["Z"], M["R"]
I_LE, I_TE, K_TIP = int(M["I_LE"]), int(M["I_TE"]), int(M["K_TIP"])
TH = B["TH"]
JB, NJ = int(B["JB"]), int(B["NJ"])
NI, NK = Zm.shape

R3 = np.broadcast_to(Rm[:, None, :], TH.shape)
Z3 = np.broadcast_to(Zm[:, None, :], TH.shape)
X = np.stack([R3 * np.cos(TH), R3 * np.sin(TH), Z3], axis=-1) * CM_TO_M     # (NI, NJ+1, NK, 3)
pts_all = X.reshape(-1, 3)


def nid(i, j, k):
    return (i * (NJ + 1) + j) * NK + k


# ----------------------------------------------------------------------------- cells
fluid = np.ones((NI - 1, NJ, NK - 1), bool)
fluid[I_LE:I_TE, :JB, :K_TIP] = False                     # blade material below the tip
ci, cj, ck = np.nonzero(fluid)
# VTK hexahedron order with (first, second, third) = (k, j, i)
corner = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
conn = np.stack([nid(ci + di, cj + dj, ck + dk) for di, dj, dk in corner], axis=1)

# outward faces of a VTK hexahedron (local node indices)
FACE = {"i-": (0, 3, 2, 1), "i+": (4, 5, 6, 7),
        "k-": (0, 4, 7, 3), "k+": (1, 2, 6, 5),
        "j-": (0, 1, 5, 4), "j+": (3, 7, 6, 2)}


def faces(sel, side):
    return conn[sel][:, list(FACE[side])]


in_blade_ik = (ci >= I_LE) & (ci < I_TE) & (ck < K_TIP)
markers = {
    "INFLOW": faces(ci == 0, "i-"),
    "OUTFLOW": faces(ci == NI - 2, "i+"),
    "PER1": faces(cj == 0, "j-"),
    "PER2": faces((cj == NJ - 1) & ~in_blade_ik, "j+"),
}
blade = [
    faces((cj == NJ - 1) & in_blade_ik, "j+"),                                   # surface 2 (next blade)
    faces((cj == JB) & in_blade_ik, "j-"),                                       # surface 1
    faces((ci == I_LE - 1) & (cj < JB) & (ck < K_TIP), "i+"),                     # leading edge face
    faces((ci == I_TE) & (cj < JB) & (ck < K_TIP), "i-"),                         # trailing edge face
    faces((ck == K_TIP) & (cj < JB) & (ci >= I_LE) & (ci < I_TE), "k-"),          # blade tip
]
markers["BLADE"] = np.concatenate(blade)
hub = faces(ck == 0, "k-")
z_hub = pts_all[hub][:, :, 2].mean(axis=1) / CM_TO_M
rot = (z_hub > Z_ROTATING_HUB[0]) & (z_hub < Z_ROTATING_HUB[1])
markers["HUB_ROT"] = hub[rot]
markers["HUB_STAT"] = hub[~rot]
markers["SHROUD"] = faces(ck == NK - 2, "k+")

# ----------------------------------------------------------------------------- renumber nodes
used = np.unique(conn)
new = np.full(len(pts_all), -1, np.int64)
new[used] = np.arange(len(used))
pts = pts_all[used]
conn = new[conn]
markers = {name: new[f] for name, f in markers.items()}

# ----------------------------------------------------------------------------- checks
def hex_volumes(p, c):
    v = np.zeros(len(c))
    for a, b, d in [(1, 2, 6), (2, 3, 6), (3, 7, 6), (7, 4, 6), (4, 5, 6), (5, 1, 6)]:
        p0 = p[c[:, 0]]
        v += np.einsum("ij,ij->i", p[c[:, a]] - p0, np.cross(p[c[:, b]] - p0, p[c[:, d]] - p0)) / 6.0
    return v


def quad_area(p, f):
    a = np.cross(p[f[:, 2]] - p[f[:, 0]], p[f[:, 3]] - p[f[:, 1]])
    return 0.5 * np.linalg.norm(a, axis=1)


vol = hex_volumes(pts, conn)
print(f"nodes: {len(pts)}   hexahedra: {len(conn)}")
print(f"cell volume: min {vol.min():.3e} m3, max {vol.max():.3e} m3, negative: {(vol <= 0).sum()}")
print(f"domain volume x {N_BLADES} blades = {vol.sum() * N_BLADES * 1e3:.3f} litres")
print("markers:")
for name, f in markers.items():
    print(f"  {name:9s} {len(f):7d} faces, area {quad_area(pts, f).sum() * 1e4:9.3f} cm2")
print(f"periodic faces match: {len(markers['PER1']) == len(markers['PER2'])}")
# PER1 rotated by +PITCH must land on PER2
c, s = np.cos(PITCH), np.sin(PITCH)
p1 = pts[np.unique(markers["PER1"])]
p1r = np.column_stack([c * p1[:, 0] - s * p1[:, 1], s * p1[:, 0] + c * p1[:, 1], p1[:, 2]])
p2 = pts[np.unique(markers["PER2"])]
key = lambda p: np.lexsort((p[:, 1].round(7), p[:, 0].round(7), p[:, 2].round(7)))  # noqa: E731
d_per = np.abs(p1r[key(p1r)] - p2[key(p2)]).max() if len(p1r) == len(p2) else np.nan
print(f"periodic node mismatch after rotation: {d_per:.2e} m")
fp = Flowpath()
z_in = Zm[0, 0]
a_exact = np.pi * (fp.cas(z_in) ** 2 - fp.hub(z_in) ** 2) / N_BLADES
print(f"inlet area: mesh {quad_area(pts, markers['INFLOW']).sum() * 1e4:.3f} cm2, "
      f"exact annulus/{N_BLADES} {a_exact:.3f} cm2")

# ----------------------------------------------------------------------------- SU2
su2 = MESH / "rotor67.su2"
with open(su2, "w") as f:
    f.write("% NASA Rotor 67 single passage H-grid, coordinates in metres, rotation axis z\n")
    f.write(f"NDIME= 3\nNELEM= {len(conn)}\n")
    np.savetxt(f, np.column_stack([np.full(len(conn), 12), conn, np.arange(len(conn))]), fmt="%d")
    f.write(f"NPOIN= {len(pts)}\n")
    np.savetxt(f, np.column_stack([pts, np.arange(len(pts))]), fmt="%.10e %.10e %.10e %d")
    f.write(f"NMARK= {len(markers)}\n")
    for name, fc in markers.items():
        f.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(fc)}\n")
        np.savetxt(f, np.column_stack([np.full(len(fc), 9), fc]), fmt="%d")
print(f"written {su2}  ({su2.stat().st_size / 1e6:.0f} MB)")


# ----------------------------------------------------------------------------- VTK (ParaView)
def write_vtk(path, p, cells, ctype, data):
    n, npc = cells.shape
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nNASA Rotor 67 mesh\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(p)} double\n".encode())
        f.write(p.astype(">f8").tobytes() + b"\n")
        f.write(f"CELLS {n} {n * (npc + 1)}\n".encode())
        f.write(np.column_stack([np.full(n, npc), cells]).astype(">i4").tobytes() + b"\n")
        f.write(f"CELL_TYPES {n}\n".encode())
        f.write(np.full(n, ctype).astype(">i4").tobytes() + b"\n")
        f.write(f"CELL_DATA {n}\n".encode())
        for name, (vals, typ) in data.items():
            f.write(f"SCALARS {name} {'double' if typ == 'f' else 'int'} 1\nLOOKUP_TABLE default\n".encode())
            f.write(vals.astype(">f8" if typ == "f" else ">i4").tobytes() + b"\n")


write_vtk(MESH / "rotor67_volume.vtk", pts, conn, 12, {"volume": (vol, "f")})
names = list(markers)
allf = np.concatenate([markers[n] for n in names])
mid = np.concatenate([np.full(len(markers[n]), m) for m, n in enumerate(names)])
uf, inv = np.unique(allf, return_inverse=True)
write_vtk(MESH / "rotor67_boundaries.vtk", pts[uf], inv.reshape(allf.shape), 9, {"marker": (mid, "i")})
print("marker ids in rotor67_boundaries.vtk: " + ", ".join(f"{m}={n}" for m, n in enumerate(names)))
print(f"done in {time.time() - t0:.0f} s")
