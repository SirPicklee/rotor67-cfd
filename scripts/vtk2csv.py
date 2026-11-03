"""Convert an SU2 legacy binary VTK volume file (PARAVIEW_LEGACY) to CSV (points + point data).

Usage:  python scripts/vtk2csv.py input.vtk output.csv
"""
import sys

import numpy as np
import pandas as pd

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
    a = np.frombuffer(buf, dt, n, pos).astype(float)
    pos += n * dt.itemsize
    while pos < len(buf) and buf[pos:pos + 1] in (b"\n", b"\r"):
        pos += 1
    return a


for _ in range(4):
    print("header:", line())
npts, data = 0, {}
while pos < len(buf):
    s = line()
    if not s:
        continue
    t = s.split()
    key = t[0].upper()
    if key == "POINTS":
        npts = int(t[1])
        p = arr(3 * npts, t[2]).reshape(npts, 3)
        data["x"], data["y"], data["z"] = p[:, 0], p[:, 1], p[:, 2]
    elif key == "CELLS":
        arr(int(t[2]), "int")
    elif key == "CELL_TYPES":
        arr(int(t[1]), "int")
    elif key in ("POINT_DATA", "CELL_DATA"):
        print(s)
        if key == "CELL_DATA":
            break
    elif key == "SCALARS":
        nc = int(t[3]) if len(t) > 3 else 1
        line()  # LOOKUP_TABLE default
        a = arr(npts * nc, t[2]).reshape(npts, nc)
        data[t[1]] = a[:, 0]
    elif key == "VECTORS":
        a = arr(3 * npts, t[2]).reshape(npts, 3)
        data[t[1] + "_mag"] = np.linalg.norm(a, axis=1)
    elif key == "FIELD":
        for _ in range(int(t[2])):
            name, nc, nt, typ = line().split()
            a = arr(int(nc) * int(nt), typ).reshape(int(nt), int(nc))
            data[name] = a[:, 0] if int(nc) == 1 else np.linalg.norm(a, axis=1)
    else:
        print("skipping unknown line:", s[:60])
        break
print("fields:", ", ".join(data))
pd.DataFrame(data).to_csv(sys.argv[2], index=False)
print("written", sys.argv[2])
