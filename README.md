# NASA Rotor 67 — 3D RANS CFD

Single-passage 3D RANS analysis of **NASA Rotor 67**, the low-aspect-ratio transonic fan rotor
of NASA TP-2879 (Strazisar et al., 1989), with an open-source toolchain:
a structured H-grid written from scratch in Python and the **SU2** compressible solver.

This is the follow-up to [transonic-fan](https://github.com/SirPicklee/transonic-fan), where a
meanline / radial-equilibrium model of the same rotor predicted the near-stall pressure ratio
within 0.3 % but **not** the efficiency drop from 0.930 (peak efficiency) to 0.906 (near stall).
The NASA laser anemometer data show the passage shock detaching and moving upstream as the rotor
approaches stall — a shock-position effect a one-dimensional model cannot represent.

**Goal:** solve the peak-efficiency (34.57 kg/s) and near-stall (32.31 kg/s) points in 3D, compare
them with the NASA measurements (overall performance, spanwise profiles, laser Mach contours) and
show where the extra loss near stall comes from.

> **Status:** geometry and mesh done; first SU2 solution (choke point) running.

---

## Geometry

The blade and flowpath were transcribed from NASA TP-2879 Tables I and II
(scanned tables → two OCR passes → smoothness and physics checks → visual check of every flagged cell):

| File | Content |
|---|---|
| `data/rotor67_blade_sections.csv` | 14 sections × 35 points: z, r, θ of both blade surfaces, thickness, camber angles |
| `data/rotor67_flowpath.csv` | hub and casing contours (80 points) |
| `data/rotor67_blade_edges.csv` | leading/trailing edge circles |

Independent checks: section 1 lies on the hub within 0.012 mm and section 14 on the casing within 0.045 mm;
computed solidity (1.29 tip, 3.11 hub) and maximum thickness match the published design values.
The table describes the cold blade; the running tip is the casing offset by the 1.016 mm clearance.

<p align="center"><img src="figures/geometry_meridional.png" width="48%"> <img src="figures/geometry_blade_to_blade.png" width="48%"></p>

## Mesh

Structured single-passage **H-grid**, generated in Python (`scripts/02-04`) and written directly in SU2 format.

| | |
|---|---|
| Cells | 1.38 M hexahedra (156 streamwise × 96 pitchwise × 101 spanwise, blade interior removed) |
| Tip gap | 16 cells across the 1.016 mm clearance, blade strip meshed above the tip for the leakage flow |
| Wall spacing | 15 µm on the blade, 20 µm on hub and casing (SA + wall functions) |
| Quality | no negative volumes, pitchwise growth ≤ 1.20 in the passage, periodic faces match to 1e-16 m |
| Limitation | blunt (flat) leading/trailing edge faces instead of the 0.2–0.5 mm edge circles |

## Solver setup (SU2 8.5)

| | |
|---|---|
| Model | steady RANS, Spalart–Allmaras, standard wall functions |
| Frame | rotating frame, 16 043 rpm (1680 rad/s) |
| Inlet | Giles, total conditions 101 325 Pa / 288.15 K, axial |
| Outlet | Giles, radial equilibrium, static pressure prescribed at mid-span |
| Walls | blade and rotating hub turn with the rotor; casing and stationary hub (`MARKER_SHROUD`) |
| Numerics | Roe, first order start → second order MUSCL + Venkatakrishnan limiter |

Compute: one desktop (Ryzen 7 5700, 24 GB RAM, WSL2). The implicit solver is memory-bandwidth bound:
dual-channel memory and SU2's mixed-precision build (`-Denable-mixedprec=true`) together cut the time per
iteration from 6.6 s to 3.0 s.

## Repository

```
data/       NASA TP-2879 geometry (CSV)
scripts/    r67geom.py (shared geometry), 01 plot geometry, 02-04 mesh, run_su2.sh
cases/      SU2 configuration files (smoke test, choke, operating points)
figures/    plots
```

Reproduce the mesh:

```bash
python scripts/02_mesh_meridional.py
python scripts/03_mesh_b2b.py
python scripts/04_mesh_write.py        # -> mesh/rotor67.su2
```

## Reference

Strazisar, A. J., Wood, J. R., Hathaway, M. D., Suder, K. L., *Laser Anemometer Measurements in a
Transonic Axial-Flow Fan Rotor*, NASA TP-2879, 1989.
[NTRS 19900001929](https://ntrs.nasa.gov/citations/19900001929)
