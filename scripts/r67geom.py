"""Shared geometry and grid utilities for the NASA Rotor 67 CFD mesh.

Units: lengths in cm, angles in rad (as in NASA TP-2879).

Conventions
-----------
span fraction  s = (r - r_hub(z)) / (r_casing(z) - r_hub(z))   (0 at hub, 1 at casing)
chord parameter u = (k - 1) / 34, k = 1..35 the NASA tabulation index
                  (NASA points are ~equally spaced in meridional distance, u = 0 at LE, 1 at TE)
grid layer      s = a * s_tip(z) + b
                  blade layers: a = eta in [0, 1], b = 0      (hub ... running blade tip)
                  gap layers:   a = 1 - g,        b = g       (blade tip ... casing)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MESH = ROOT / "mesh"
FIG = ROOT / "figures"

N_BLADES = 22
PITCH = 2.0 * np.pi / N_BLADES          # [rad]
TIP_GAP = 0.1016                         # cold tip clearance [cm], NASA TP-2879
Z_ROTATING_HUB = (-1.374, 9.365)         # rotating part of the hub [cm]


# ----------------------------------------------------------------------------- grid spacing
def vinokur(n, ds0, ds1):
    """Two-sided stretching (Vinokur 1983).

    Returns n + 1 points on [0, 1] with first spacing ~ds0 and last spacing ~ds1
    (both given as fractions of the total length).
    """
    xi = np.arange(n + 1) / n
    s0 = 1.0 / (n * ds0)
    s1 = 1.0 / (n * ds1)
    a = np.sqrt(s0 / s1)
    b = np.sqrt(s0 * s1)
    if b > 1.001:
        dy = brentq(lambda y: np.sinh(y) / y - b, 1e-8, 60.0)
        u = 0.5 * (1.0 + np.tanh(dy * (xi - 0.5)) / np.tanh(0.5 * dy))
    elif b < 0.999:
        dy = brentq(lambda y: np.sin(y) / y - b, 1e-8, np.pi - 1e-8)
        u = 0.5 * (1.0 + np.tan(dy * (xi - 0.5)) / np.tan(0.5 * dy))
    else:
        u = xi.copy()
    t = u / (a + (1.0 - a) * u)
    t[0], t[-1] = 0.0, 1.0
    return t


def wall_clustered(length, ds, ratio, dmax):
    """Point distribution on [0, 1] clustered at both ends.

    Spacing starts at ds at each wall, grows geometrically by `ratio` until it reaches dmax,
    and is uniform (<= dmax) in the core. ds, dmax and length in the same units.
    The number of points follows from the inputs.
    """
    n_grow = int(np.ceil(np.log(dmax / ds) / np.log(ratio)))
    grow = np.minimum(ds * ratio ** np.arange(n_grow), dmax)
    rest = length - 2.0 * grow.sum()
    if rest <= 0.0:
        raise ValueError("wall_clustered: length too short for ds, ratio and dmax")
    n_core = int(np.ceil(rest / dmax))
    d = np.concatenate([grow, np.full(n_core, rest / n_core), grow[::-1]])
    x = np.concatenate([[0.0], np.cumsum(d)])
    return x / x[-1]


# ----------------------------------------------------------------------------- flowpath
class Flowpath:
    """Hub and casing contours (NASA TP-2879 Table II) and the running blade tip line."""

    def __init__(self):
        f = pd.read_csv(DATA / "rotor67_flowpath.csv", comment="#")
        self.z_min, self.z_max = f.z.min(), f.z.max()
        self.hub = CubicSpline(f.z.to_numpy(), f.r_hub.to_numpy())
        self.cas = CubicSpline(f.z.to_numpy(), f.r_casing.to_numpy())

    def r_tip(self, z):
        """Running blade tip: casing offset inward by TIP_GAP along the casing normal.

        At fixed z the radial distance to the casing is TIP_GAP * sqrt(1 + (dr_c/dz)^2).
        """
        slope = self.cas(z, 1)
        return self.cas(z) - TIP_GAP * np.sqrt(1.0 + slope ** 2)

    def span(self, z, r):
        h = self.hub(z)
        return (r - h) / (self.cas(z) - h)

    def s_tip(self, z):
        return self.span(z, self.r_tip(z))

    def r_of(self, z, s):
        h = self.hub(z)
        return h + s * (self.cas(z) - h)


# ----------------------------------------------------------------------------- blade
class Blade:
    """Blade surfaces interpolated in span (across the 14 NASA sections) and chord (u)."""

    def __init__(self, fp):
        b = pd.read_csv(DATA / "rotor67_blade_sections.csv", comment="#").sort_values(["section", "k"])
        ns, nk = b.section.nunique(), b.k.nunique()
        Z = b.z.to_numpy().reshape(ns, nk)
        R = b.r.to_numpy().reshape(ns, nk)
        self.T1 = b.thsp1.to_numpy().reshape(ns, nk)
        self.T2 = b.thsp2.to_numpy().reshape(ns, nk)
        self.Z, self.R = Z, R
        self.S = fp.span(Z, R)
        self.nk = nk
        self.u = np.linspace(0.0, 1.0, nk)
        self.fp = fp
        # one spanwise spline per chordwise station
        self._spl = {
            "z": [CubicSpline(self.S[:, k], Z[:, k]) for k in range(nk)],
            "t1": [CubicSpline(self.S[:, k], self.T1[:, k]) for k in range(nk)],
            "t2": [CubicSpline(self.S[:, k], self.T2[:, k]) for k in range(nk)],
        }

    def at(self, s, u):
        """z, theta1, theta2 at span fractions s and chord parameters u (1-D arrays, same length)."""
        s = np.atleast_1d(np.asarray(s, float))
        u = np.atleast_1d(np.asarray(u, float))
        out = []
        for key in ("z", "t1", "t2"):
            cols = np.array([f(s) for f in self._spl[key]])        # (nk, n): column j is section at s_j
            vals = CubicSpline(self.u, cols, axis=0)(u)            # (n, n): [i, j] = column j at u_i
            out.append(np.diagonal(vals).copy())
        return out

    def on_layer(self, a, b, u, iters=5):
        """Blade points (z, r, theta1, theta2) at chord parameters u on grid layer s = a*s_tip(z) + b."""
        u = np.asarray(u, float)
        s = np.full_like(u, a * 0.993 + b)
        for _ in range(iters):
            z, _, _ = self.at(s, u)
            s = a * self.fp.s_tip(z) + b
        z, t1, t2 = self.at(s, u)
        return z, self.fp.r_of(z, s), t1, t2
