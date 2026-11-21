#!/usr/bin/env python3
"""Ring-holder fingers that stand in a candlestick cup like candles.

Two kinds:
  candle  - a wax finger with a nail, knuckle wrinkles and a wick hole
  bone    - a skeletal finger: metacarpal head, proximal, middle and distal phalanx

Every finger stands on a plug that pushes into a candle cup (default 22 mm,
the common 7/8 in taper size; the uploaded baroque candlestick's cup is
22.3 mm at the bottom and 23.5 mm at the top).

The shapes are built as signed distance fields, meshed with marching cubes,
simplified, checked for watertightness, and then a ring is slid down each
finger to find where every ring size comes to rest.

    python fingers.py                 # build every preset into out/
    python fingers.py --only candle-crooked --cup 21.5
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- SDF tools

def smin(a, b, k):
    """Polynomial smooth minimum (union with a fillet of size about k)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


class Frame:
    """Local frame: t along the finger, n towards the back (dorsal), b sideways."""

    def __init__(self, origin, t, n):
        self.o = np.asarray(origin, float)
        self.t = np.asarray(t, float) / np.linalg.norm(t)
        n = np.asarray(n, float)
        n = n - self.t * (n @ self.t)
        self.n = n / np.linalg.norm(n)
        self.b = np.cross(self.n, self.t)

    def local(self, P):
        d = P - self.o
        return d @ self.b, d @ self.n, d @ self.t


def profile(points):
    """Pchip curve through (u, value) pairs, u in [0, 1]."""
    u, v = zip(*points)
    return PchipInterpolator(np.array(u, float), np.array(v, float), extrapolate=True)


def cap_profile(body, u_cap, n=12):
    """Append an elliptical end cap that closes the profile at u = 1."""
    v0 = body[-1][1]
    pts = [p for p in body if p[0] < u_cap - 1e-6] + [(u_cap, v0)]
    for i in range(1, n + 1):
        s = i / n
        u = u_cap + (1 - u_cap) * (1 - math.cos(s * math.pi / 2))
        frac = (u - u_cap) / (1 - u_cap)
        pts.append((u, max(v0 * math.sqrt(max(1 - frac * frac, 0.0)), 0.25)))
    return pts


def tube(P, fr, L, fa, fcd, fcp, fyo=None, m_pal=2.0, cap_end=True, cap_start=True, m_dor=2.0):
    """Generalised elliptic tube along the frame axis from 0 to L.

    fa: half width, fcd: dorsal half depth, fcp: palmar half depth, fyo:
    dorsal offset of the centre line; all functions of u = z / L.
    m_pal > 2 makes the palmar half boxier (bone shafts)."""
    x, y, z = fr.local(P)
    zc = np.clip(z, 0.0, L)
    u = zc / L
    a = np.maximum(fa(u), 0.25)
    cd = np.maximum(fcd(u), 0.25)
    cp = np.maximum(fcp(u), 0.25)
    yy = y - (fyo(u) if fyo is not None else 0.0)
    c = np.where(yy >= 0, cd, cp)
    m = np.where(yy >= 0, m_dor, m_pal)
    ax, ay = np.abs(x) / a, np.abs(yy) / c
    q = (ax ** m + ay ** m) ** (1.0 / m)
    qs = np.maximum(q, 1e-6)
    gx = ax ** (m - 1) / a / qs ** (m - 1)
    gy = ay ** (m - 1) / c / qs ** (m - 1)
    g = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-6)
    d = (q - 1.0) / g
    d = np.maximum(d, -np.minimum(a, c))
    if cap_start:
        d = np.maximum(d, -z)
    if cap_end:
        d = np.maximum(d, z - L)
    return d


def ellipsoid(P, fr, rx, ry_d, ry_p, rz):
    """Ellipsoid in a frame, dorsal and palmar half depths may differ."""
    x, y, z = fr.local(P)
    ry = np.where(y >= 0, ry_d, ry_p)
    px, py, pz = x / rx, y / ry, z / rz
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def capsule(P, A, B, r):
    A, B = np.asarray(A, float), np.asarray(B, float)
    ab = B - A
    h = np.clip(((P - A) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - A - h[:, None] * ab, axis=1) - r


def cylinder_z(P, R, z0, z1, cx=0.0, cy=0.0):
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    return np.maximum(r - R, np.maximum(z0 - P[:, 2], P[:, 2] - z1))


def hole_along(P, origin, t, r, depth, above=2.0):
    fr = Frame(origin, t, np.cross(t, [1, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1, 0]))
    x, y, z = fr.local(P)
    return np.maximum(np.hypot(x, y) - r, np.maximum(-depth - z, z - above))


# ---------------------------------------------------------------- plug

def plug_sdf(P, cfg):
    """Cylinder that pushes into the cup, three crush ribs, optional wax collar."""
    R = cfg["cup"] / 2 - cfg["clearance"]
    Lp = cfg["plug_len"]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.hypot(x, y)
    d = np.maximum(r - R, np.maximum(-Lp - z, z - 0.0))
    ch = 1.0  # bottom chamfer
    d = np.maximum(d, ((r - R) - (z + Lp) + ch) / math.sqrt(2))
    # crush ribs, ramping in from the bottom so the plug starts easily
    h = cfg["rib"] * smoothstep(-Lp + 1.5, -Lp + 6.0, z)
    ang = np.arctan2(y, x)
    for k in range(3):
        th = k * 2 * math.pi / 3 + math.pi / 6
        da = np.angle(np.exp(1j * (ang - th)))
        tang = r * np.abs(np.sin(da))
        rib = np.maximum(np.maximum(tang - 0.7, r * np.cos(da) - (R + h)),
                         np.maximum(-Lp + 1.0 - z, z + 1.5))
        rib = np.where(np.cos(da) > 0, rib, 1e3)
        d = np.minimum(d, rib)
    if cfg.get("collar"):
        Rc = cfg["collar_d"] / 2
        tcol = cfg["collar_t"]
        # puddle of wax on the rim: a disc with a fully rounded edge, wobbly outline
        wob = 0.45 * np.sin(5 * ang + 0.7) + 0.25 * np.sin(9 * ang + 2.1)
        dx = r - (Rc + wob) + tcol / 2
        dy = np.abs(z - tcol / 2)
        disc = np.minimum(np.maximum(dx, dy), 0.0) + np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) - tcol / 2
        d = smin(d, disc, 0.6)
    return d


# ---------------------------------------------------------------- kinematics

def chain(lengths, flex_deg, abd_deg, start=(0, 0, 0)):
    """Frames of a chain of segments. flex bends towards the palm, abd sideways."""
    t = np.array([0.0, 0.0, 1.0])
    n = np.array([0.0, 1.0, 0.0])
    P = np.asarray(start, float)
    frames = []
    for L, fl, ab in zip(lengths, flex_deg, abd_deg):
        b = np.cross(n, t)
        R = rot(b, math.radians(fl))
        t, n = R @ t, R @ n
        R = rot(n, math.radians(ab))
        t, n = R @ t, R @ n
        frames.append(Frame(P, t, n))
        P = P + t * L
    return frames, P


# ---------------------------------------------------------------- candle finger

class Noise3:
    """Smooth value noise in [-1, 1] with features about `cell` mm across."""

    def __init__(self, seed, cell):
        self.g = np.random.default_rng(seed).uniform(-1, 1, (48, 48, 48)).astype(np.float32)
        self.cell = cell

    def __call__(self, P):
        q = P / self.cell + 1000.0
        i = np.floor(q).astype(np.int64)
        f = q - i
        f = f * f * (3 - 2 * f)
        i0, i1 = i % 48, (i + 1) % 48
        g = self.g
        x0, y0, z0 = i0.T
        x1, y1, z1 = i1.T
        fx, fy, fz = f.T

        def lerp(a, b, t):
            return a + (b - a) * t
        c00 = lerp(g[x0, y0, z0], g[x1, y0, z0], fx)
        c10 = lerp(g[x0, y1, z0], g[x1, y1, z0], fx)
        c01 = lerp(g[x0, y0, z1], g[x1, y0, z1], fx)
        c11 = lerp(g[x0, y1, z1], g[x1, y1, z1], fx)
        return lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)


class CandleFinger:
    """A human finger built as one continuous tube along a smoothly bending centre line.

    Cross-sections are a little boxy on the back and round on the palm side. Along
    the finger the palm side swells into a pad on every segment and dips into a
    crease at every joint; the back carries a bony bump with oval wrinkles at each
    knuckle; the tip has a fleshy pulp and a nail set into skin folds. A slight
    unevenness covers the skin. Proportions follow an adult index finger 20 mm
    wide; `width` scales the girth and `scale` everything."""

    def __init__(self, cfg):
        self.cfg = cfg
        s = cfg["scale"]
        self.fem = cfg.get("form") == "female"
        self.detail = cfg.get("detail", 1.0)
        self.L = [cfg["prox_len"] * s, (29.0 if self.fem else 26.0) * s, (26.0 if self.fem else 23.0) * s]
        w = self.w = cfg["width"] * s / 20.0
        L0, L1, L2 = self.L
        S = self.S = L0 + L1 + L2
        self.s_pip, self.s_dip = L0, L0 + L1
        # centre line: start inside the plug, bend gradually over each joint
        ds = 0.25
        bends = [(0.0, 6.0, cfg["mcp"], cfg["tilt"]), (L0, 7.0 * s, cfg["pip"], cfg["pip_side"]),
                 (L0 + L1, 6.0 * s, cfg["dip"], cfg["dip_side"])]
        p = np.array([0.0, 0.0, -4.0])
        t = np.array([0.0, 0.0, 1.0])
        n = np.array([0.0, 1.0, 0.0])
        C, T, N, B, Sv = [], [], [], [], []
        k = 0
        while True:
            sc = k * ds
            b = np.cross(n, t)
            C.append(p.copy()); T.append(t.copy()); N.append(n.copy()); B.append(b.copy()); Sv.append(sc)
            if sc >= S:
                break
            for s0, lb, fl, ab in bends:
                lo_, hi_ = (s0, s0 + lb) if s0 == 0.0 else (s0 - lb / 2, s0 + lb / 2)
                if lo_ <= sc < hi_:
                    R = rot(b, math.radians(fl) * ds / lb)
                    t, n = R @ t, R @ n
                    R = rot(n, math.radians(ab) * ds / lb)
                    t, n = R @ t, R @ n
            p = p + t * ds
            k += 1
        self.C, self.T, self.N, self.B = map(np.array, (C, T, N, B))
        self.Sv = np.array(Sv)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.C)

        # profiles along the length (mm): half width, back and palm half depths
        sp, sd = self.s_pip, self.s_dip

        def table(pts, cap_from=None):
            u, v = zip(*sorted(pts))
            f = PchipInterpolator(np.array(u), np.array(v) * w)
            xs = np.arange(0.0, S + 1e-6, 0.1)
            ys = f(np.clip(xs, u[0], u[-1]))
            if cap_from is not None:
                q = np.clip((xs - cap_from) / (S - cap_from), 0, 1)
                ys = np.where(xs > cap_from, f(cap_from) * np.sqrt(np.maximum(1 - q * q, 0.0)), ys)
            return xs, np.maximum(ys, 0.25)

        if self.fem:
            # slender and evenly tapering: small joints, gentle pads, and a tip shaped by a long
            # oval nail: the back stays flat to the end, the pad rounds off earlier underneath
            cap_a = sd + 0.62 * L2
            self.a = table([(0, 9.3), (0.3 * L0, 8.7), (0.7 * L0, 8.2), (sp, 8.2), (sp + 0.4 * L1, 7.35),
                            (sd, 7.15), (sd + 0.25 * L2, 6.7), (sd + 0.45 * L2, 6.7), (cap_a, 6.5)], cap_a)
            cap_d = sd + 0.74 * L2
            self.cd = table([(0, 7.0), (0.6 * L0, 6.4), (sp, 6.35), (sp + 0.5 * L1, 5.55), (sd, 5.25),
                             (sd + 0.3 * L2, 4.6), (cap_d, 4.3)], cap_d)
            cap_p = sd + 0.55 * L2
            self.cp = table([(0, 8.3), (0.3 * L0, 8.6), (0.62 * L0, 8.35), (sp - 3.5 * s, 7.2), (sp - 0.4 * s, 6.7),
                             (sp + 3.0 * s, 6.8), (sp + 0.45 * L1, 7.0), (sd - 2.5 * s, 6.15), (sd, 5.7),
                             (sd + 0.2 * L2, 6.1), (sd + 0.4 * L2, 6.4), (cap_p, 6.3)], cap_p)
        else:
            cap_a = sd + 0.66 * L2
            self.a = table([(0, 10.2), (0.3 * L0, 9.7), (0.7 * L0, 9.25), (sp, 9.4), (sp + 0.4 * L1, 8.4),
                            (sd, 8.35), (sd + 0.25 * L2, 7.85), (sd + 0.45 * L2, 8.05), (cap_a, 8.0)], cap_a)
            self.cd = table([(0, 7.8), (0.6 * L0, 7.15), (sp, 7.25), (sp + 0.5 * L1, 6.3), (sd, 6.0),
                             (sd + 0.3 * L2, 5.3), (cap_a, 5.0)], cap_a)
            cap_p = sd + 0.7 * L2
            self.cp = table([(0, 9.3), (0.3 * L0, 9.9), (0.62 * L0, 9.45), (sp - 3.5 * s, 7.9), (sp - 0.4 * s, 7.0),
                             (sp + 3.0 * s, 7.4), (sp + 0.45 * L1, 8.0), (sd - 2.5 * s, 6.8), (sd, 6.0),
                             (sd + 0.2 * L2, 6.9), (sd + 0.45 * L2, 7.6), (cap_p, 7.45)], cap_p)
        u = np.arange(0.0, S + 1e-6, 0.1)
        self.yo = (u, w * (3.2 if self.fem else 2.4) * smoothstep(sd + 0.45 * L2, S, u) ** 1.5)

        # frames at the segment starts (used by the drips, the wick and the bounds)
        def at(sv):
            i = int(round(sv / ds))
            i = min(i, len(self.C) - 1)
            return i
        self.frames = [Frame(self.C[at(v)], self.T[at(v)], self.N[at(v)]) for v in (0.0, sp, sd)]
        self.joint = {name: at(v) for name, v in (("pip", sp), ("dip", sd))}
        self.tip = self.C[-1]
        self.tip_point = self.C[-1] + self.N[-1] * np.interp(S, *self.yo)
        self.frames[2] = Frame(self.frames[2].o, self.T[-1], self.N[-1])
        self.spine_path = list(self.C[::-8][:])
        self.noise = (Noise3(7, 6.0 * w), Noise3(11, 2.2 * w))
        self.drips = Drips(self, cfg) if cfg.get("drips") else None

    def bounds(self):
        lo = self.C.min(0) - 14 * self.w
        hi = self.C.max(0) + 14 * self.w
        R = max(self.cfg["cup"], self.cfg.get("collar_d", 0) if self.cfg.get("collar") else 0) / 2 + 1.5
        lo = np.minimum(lo, [-R, -R, -self.cfg["plug_len"] - 1.5])
        hi = np.maximum(hi, [R, R, 2.0])
        return lo, hi

    def local(self, P):
        """Sideways, backwards and along-the-finger coordinates of each point."""
        _, i = self.tree.query(P, workers=-1)
        v = P - self.C[i]
        x = np.einsum("ij,ij->i", v, self.B[i])
        y = np.einsum("ij,ij->i", v, self.N[i])
        z = np.einsum("ij,ij->i", v, self.T[i])
        return x, y, self.Sv[i] + z, i

    def flesh(self, P):
        w = self.w
        x, y, s, i = self.local(P)
        sc = np.clip(s, 0, self.S)
        a = np.interp(sc, *self.a)
        yy = y - np.interp(sc, *self.yo)
        c = np.where(yy >= 0, np.interp(sc, *self.cd), np.interp(sc, *self.cp))
        m = np.where(yy >= 0, 2.3, 2.0)
        ax, ay = np.abs(x) / a, np.abs(yy) / c
        q = (ax ** m + ay ** m) ** (1.0 / m)
        qs = np.maximum(q, 1e-6)
        g = np.sqrt((ax ** (m - 1) / a) ** 2 + (ay ** (m - 1) / c) ** 2) / qs ** (m - 1)
        d = (q - 1.0) / np.maximum(g, 1e-6)
        d = np.maximum(d, -np.minimum(a, c))
        d = np.maximum(d, np.maximum(-s, s - self.S))
        # bony bumps on the back of each knuckle
        kn = (("pip", (4.8, 1.1, 6.5, 5.0)), ("dip", (3.9, 0.9, 4.8, 4.0))) if self.fem else \
            (("pip", (5.8, 2.6, 5.0, 4.9)), ("dip", (4.8, 2.1, 3.8, 3.9)))
        for name, (lat, up, along, lift) in (() if self.fem else kn):   # female knuckles: only the faint swell in the outline
            j = self.joint[name]
            cen = self.C[j] + self.N[j] * lift * w
            bump = ellipsoid(P, Frame(cen, self.T[j], self.N[j]), lat * w, up * w, up * w, along * w)
            d = smin(d, bump, 2.6 if self.fem else 2.2)
        d = d - self.nail(x, yy, s)
        k = self.detail * (0.45 if self.fem else 1.0)       # finer, softer skin on the female fingers
        d = d + k * self.creases(P, x, y, s)
        n1, n2 = self.noise
        d = d + (0.11 * n1(P) + 0.035 * n2(P)) * max(self.detail, 0.3)
        return d

    def sdf(self, P):
        cfg = self.cfg
        d = self.flesh(P)
        if self.drips is not None:
            d = self.drips.apply(P, d)
        # slim fingers flare softly into the plug so no ledge shows at the cup rim
        d = smin(d, plug_sdf(P, cfg), 3.2 if self.fem else 1.2)
        if cfg["wick_hole"] > 0:
            h = hole_along(P, self.tip_point, self.T[-1], cfg["wick_hole"] / 2, cfg["wick_depth"])
            d = smax(d, -h, 0.25)
        return d

    def nail(self, x, yy, s):
        """Raise (mm) for the nail plate and the skin folds around it."""
        u = (s - self.s_dip) / self.L[2]
        phi = np.abs(np.arctan2(x, yy))
        nw = 0.74 if self.fem else 0.8                         # a narrower, longer, oval nail on the female fingers
        width = nw - 0.1 * smoothstep(0.82, 1.0, u)            # nail narrows towards the free edge
        u0 = (0.38 if self.fem else 0.5) + 0.08 * np.clip(phi / nw, 0, 1.3) ** 2   # cuticle arc
        on = smoothstep(u0, u0 + 0.02, u) * smoothstep(-0.04, 0.02, width - phi)
        plate = on * (0.42 - 0.22 * (phi / width) ** 2 + (0.5 if self.fem else 0.35) * smoothstep(0.88, 1.0, u))
        groove_side = -0.32 * np.exp(-((phi - width - 0.03) / 0.04) ** 2) * smoothstep(u0 - 0.02, u0 + 0.06, u)
        fold_side = 0.3 * np.exp(-((phi - width - 0.14) / 0.08) ** 2) * smoothstep(u0 - 0.06, u0 + 0.08, u) \
            * smoothstep(1.0, 0.85, u)
        near = smoothstep(-0.14, 0.0, width + 0.12 - phi)
        groove_cut = -0.3 * np.exp(-((u - u0 + 0.004) / 0.012) ** 2) * near
        fold_cut = 0.36 * np.exp(-((u - (u0 - 0.04)) / 0.03) ** 2) * near
        out = plate + groove_side + fold_side + groove_cut + fold_cut
        return np.where(u > -0.1, out, 0.0)

    def creases(self, P, x, y, s):
        """Palm creases and knuckle wrinkles, as depth (mm) added to the distance."""
        w = self.w
        n2 = self.noise[1]
        add = np.zeros(len(P))
        phi = np.arctan2(x, y)
        palm = smoothstep(2.0, 2.55, np.abs(phi))
        back = smoothstep(1.15, 0.55, np.abs(phi)) * (y > 0)
        wob = 0.25 * n2(P * 1.3)
        specs = [(self.s_pip, ((-1.2 * w, 0.55), (1.3 * w, 0.0 if self.fem else 0.35)), 0.24, 1.35 * w, (5.6 * w, 4.8 * w), 1.0 * w),
                 (self.s_dip, ((0.3 * w, 0.5),), 0.16, 1.1 * w, (4.6 * w, 3.2 * w), 0.8 * w)]
        for sj, creases, amp, lam, (hx, hz), shift in specs:
            z = s - sj
            for o, deep in creases:
                zc = o + 0.45 * w * (np.pi - np.abs(phi)) ** 2 / 2 + wob
                add += deep * np.exp(-((z - zc) / 0.26) ** 2) * palm
            zk = z - shift                 # the wrinkle field sits a little past the joint line
            env = np.exp(-((x / hx) ** 2 + (zk / hz) ** 2) * 1.6)
            zz = zk - 0.08 * x * x / w + 0.5 * n2(P)
            add += amp * (0.5 + 0.5 * np.cos(2 * np.pi * zz / lam)) * env * back * (0.7 + 0.5 * n2(P * 0.7))
        return add


class Drips:
    """Melted wax: a thin layer over the fingertip and drips running down the skin.

    Each drip is traced downhill over the finger's surface from the edge of the
    melted layer, as a flat ribbon of wax (about three times wider than high) that
    thickens towards a teardrop at its end. Lengths vary, so drips stop at
    different heights."""

    def __init__(self, finger, cfg):
        self.f = finger
        seed = cfg.get("drip_seed", 3)
        rng = np.random.default_rng(seed)
        w = finger.w
        tip, t = finger.tip_point, finger.frames[2].t
        self.tip, self.t = tip, t
        self.cap_r = 6.0 * w
        self.cap_t = 0.5
        self.lobes = Noise3(seed + 5, 2.2)
        samples = []
        n = cfg.get("drips", 6)
        fr = finger.frames[2]
        lo, hi = cfg.get("drip_len", (8.0, 38.0))
        for k in range(n):
            ang = 2 * np.pi * (k + rng.uniform(-0.35, 0.35)) / n
            radial = fr.b * np.cos(ang) + fr.n * np.sin(ang)
            p = tip - t * rng.uniform(2.0, 3.5) * w + radial * (6.5 * w)
            length = lo + (hi - lo) * rng.uniform(0, 1) ** 1.2
            samples += self.trace(p, length, rng.uniform(1.25, 1.75), rng)
        S = np.array(samples)
        self.c, self.n, self.r, self.k = S[:, :3], S[:, 3:6], S[:, 6], S[:, 7]
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.c)
        self.lo = self.c.min(0) - 4.0
        self.hi = self.c.max(0) + 4.0

    def grad(self, p, h=0.05):
        P = np.array([p + e for e in np.eye(3) * h] + [p - e for e in np.eye(3) * h])
        d = self.f.flesh(P)
        g = (d[:3] - d[3:]) / (2 * h)
        return g / max(np.linalg.norm(g), 1e-9)

    def project(self, p):
        for _ in range(3):
            p = p - self.f.flesh(p[None])[0] * self.grad(p)
        return p

    def trace(self, p, length, r0, rng):
        out = []
        p = self.project(p)
        s, step = 0.0, 0.3
        phase = rng.uniform(0, 6.28)
        while s < length and p[2] > 6.0:
            n = self.grad(p)
            flow = self.grad(p, h=0.8)          # the drip follows the overall slope, not small creases
            g = np.array([0.0, 0.0, -1.0])
            d = g - (g @ flow) * flow
            if np.linalg.norm(d) < 0.2:         # nearly level: the wax pools and stops
                break
            d /= np.linalg.norm(d)
            r = r0 * (0.85 + 0.3 * s / max(length, 1)) * (1 + 0.12 * np.sin(0.45 * s + phase))
            out.append([*(p + n * 0.15 * r), *n, r, 2.1])
            p = self.project(p + d * step)
            s += step
        if out:   # a small teardrop at the end: a little fuller and rounder, hanging down
            n = self.grad(p)
            r = out[-1][6]
            for dz, f, k in ((0.3, 1.08, 1.9), (0.8, 1.16, 1.7), (1.3, 1.08, 1.7)):
                q = self.project(p + np.array([0, 0, -dz]))
                out.append([*(q + n * 0.2 * r), *n, r * f, k])
        return out

    def apply(self, P, d):
        # the melted layer over the tip, with a lobed edge
        dist = np.linalg.norm(P - self.tip, axis=1)
        edge = self.cap_r + 1.3 * self.lobes(P)
        film = self.cap_t * smoothstep(edge + 0.5, edge - 0.5, dist)
        d = d - film
        inbox = np.all((P > self.lo) & (P < self.hi), axis=1)
        if inbox.any():
            q = P[inbox]
            _, ii = self.tree.query(q, k=8)
            v = q[:, None, :] - self.c[ii]
            vn = np.einsum("ijk,ijk->ij", v, self.n[ii])
            vt2 = np.einsum("ijk,ijk->ij", v, v) - vn * vn
            # flattened along the surface normal: wide and low like a run of wax
            dr = np.sqrt(np.maximum(vt2, 0) + (self.k[ii] * vn) ** 2) - self.r[ii]
            dr = np.min(dr / np.sqrt(self.k[ii]), axis=1)
            d = d.copy()
            d[inbox] = smin(d[inbox], dr, 0.6)
        return d


# ---------------------------------------------------------------- bone finger

class BoneFinger:
    """Skeletal finger: metacarpal head, proximal, middle and distal phalanx.

    Approximate adult index-finger proportions (real size at scale 1.0); the
    default scale is larger so the joints are wide enough to hold rings."""

    # length, base width, base depth, shaft width, shaft depth, head width, head depth
    BONES = [
        (40.0, 15.0, 12.0, 8.6, 6.8, 11.6, 9.0),
        (24.0, 12.4, 9.6, 7.0, 5.6, 9.4, 7.4),
        (18.0, 10.0, 7.6, 5.0, 4.2, 7.4, 3.6),
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        s = self.s = cfg["scale"]
        self.gap = cfg["joint_gap"]
        flex = [cfg["mcp"], cfg["pip"], cfg["dip"]]
        abd = [cfg["tilt"], cfg["pip_side"], cfg["dip_side"]]
        # metacarpal head: a knob half sunk into the wax at the cup
        self.mc_r = 0.42 * self.BONES[0][1] * s
        self.mc_c = np.array([0.0, 0.0, cfg["mc_height"]])
        self.mc_axes = (self.mc_r * 1.1, self.mc_r * 0.95, self.mc_r * 1.0, self.mc_r)
        # each bone pivots about the condyle centre of the bone below it
        pivot = self.mc_c.copy()
        pivot_r = self.mc_r
        sock = (Frame(pivot, [0, 0, 1], [0, 1, 0]), self.mc_axes)
        t = np.array([0.0, 0.0, 1.0])
        n = np.array([0.0, 1.0, 0.0])
        self.bones = []
        for i, (L, bw, bd, sw, sd, hw, hd) in enumerate(self.BONES):
            b = np.cross(n, t)
            R = rot(b, math.radians(flex[i]))
            t, n = R @ t, R @ n
            R = rot(n, math.radians(abd[i]))
            t, n = R @ t, R @ n
            L, bw, bd, sw, sd, hw, hd = (v * s for v in (L, bw, bd, sw, sd, hw, hd))
            origin = pivot + t * (pivot_r + self.gap)
            fr = Frame(origin, t, n)
            rc = 0.30 * hw  # condyle radius
            self.bones.append(dict(fr=fr, L=L, bw=bw, bd=bd, sw=sw, sd=sd, hw=hw, hd=hd, rc=rc,
                                   socket_c=pivot.copy(), socket=sock, last=(i == 2)))
            pivot = origin + t * (L - rc)
            pivot_r = rc
            sock = (Frame(pivot, t, n), (0.23 * hw + rc, rc * 1.25, rc * 1.3, rc))
        self.tip = self.bones[-1]["fr"].o + self.bones[-1]["fr"].t * self.bones[-1]["L"]

    def bounds(self):
        pts = [b["fr"].o for b in self.bones] + [self.tip, self.mc_c]
        lo = np.min(pts, axis=0) - 12 * self.s
        hi = np.max(pts, axis=0) + 12 * self.s
        R = max(self.cfg["cup"], self.cfg.get("collar_d", 0)) / 2 + 1.5
        lo = np.minimum(lo, [-R, -R, -self.cfg["plug_len"] - 1.5])
        hi = np.maximum(hi, [R, R, 2.0])
        return lo, hi

    def bone(self, P, B):
        fr, L = B["fr"], B["L"]
        bw, bd, sw, sd, hw, hd, rc = B["bw"], B["bd"], B["sw"], B["sd"], B["hw"], B["hd"], B["rc"]
        if not B["last"]:
            fa = profile([(0, bw * 0.47), (0.12, bw * 0.44), (0.3, sw * 0.62), (0.55, sw * 0.5),
                          (0.8, hw * 0.42), (1, hw * 0.45)])
            fcd = profile([(0, bd * 0.5), (0.15, bd * 0.42), (0.4, sd * 0.55), (0.7, sd * 0.52),
                           (1, hd * 0.42)])
            fcp = profile([(0, bd * 0.5), (0.15, bd * 0.4), (0.4, sd * 0.42), (0.7, sd * 0.45),
                           (1, hd * 0.5)])
            d = tube(P, fr, L - rc * 0.8, fa, fcd, fcp, m_pal=2.8)
            # two condyles at the head, slightly towards the palm
            for sgn in (-1, 1):
                c = fr.o + fr.t * (L - rc) + fr.b * (sgn * 0.23 * hw) - fr.n * (0.08 * hd)
                cf = Frame(c, fr.t, fr.n)
                d = smin(d, ellipsoid(P, cf, rc * 1.0, rc * 1.25, rc * 1.3, rc), 1.0)
            # collateral tubercles at the sides of the base
            for sgn in (-1, 1):
                c = fr.o + fr.t * (0.1 * L) + fr.b * (sgn * 0.4 * bw)
                d = smin(d, np.linalg.norm(P - c, axis=1) - 0.13 * bw, 1.2)
        else:
            tw = 0.62 * bw  # tuft width
            fa = profile([(0, bw * 0.47), (0.15, bw * 0.4), (0.4, sw * 0.55), (0.62, sw * 0.5),
                          (0.82, tw * 0.45), (0.92, tw * 0.47), (1, tw * 0.3)])
            fcd = profile([(0, bd * 0.5), (0.2, bd * 0.4), (0.45, sd * 0.5), (0.8, sd * 0.4),
                           (1, sd * 0.3)])
            fcp = profile([(0, bd * 0.5), (0.2, bd * 0.38), (0.45, sd * 0.45), (0.8, sd * 0.42),
                           (1, sd * 0.3)])
            d = tube(P, fr, L, fa, fcd, fcp, m_pal=2.6)
            # the tuft: a flat spade at the tip
            c = fr.o + fr.t * (0.9 * L)
            tf = Frame(c, fr.t, fr.n)
            d = smin(d, ellipsoid(P, tf, tw * 0.5, sd * 0.42, sd * 0.38, 0.12 * L), 1.0)
        # base: wider block with a socket around the condyles below
        bc = fr.o + fr.t * (0.06 * L)
        bf = Frame(bc, fr.t, fr.n)
        d = smin(d, ellipsoid(P, bf, bw * 0.5, bd * 0.48, bd * 0.52, 0.11 * L + 1.0), 1.5)
        sf, (rx, ryd, ryp, rz) = B["socket"]
        g = self.gap
        sock = ellipsoid(P, sf, rx + g, ryd + g, ryp + g, rz + g)
        d = smax(d, -sock, 0.4)
        return d

    def sdf(self, P):
        cfg, s = self.cfg, self.s
        d = np.full(len(P), 1e3)
        # metacarpal head: condyle knob with a short neck going down into the wax
        mc = ellipsoid(P, Frame(self.mc_c, [0, 0, 1], [0, 1, 0]), *self.mc_axes)
        neck = capsule(P, self.mc_c, self.mc_c - np.array([0, 0, 8.0]), self.mc_r * 0.72)
        d = smin(mc, neck, 2.0)
        for B in self.bones:
            d = np.minimum(d, self.bone(P, B))
            # hidden ligament core across each joint so the finger prints as one piece
            core_r = 0.3 * min(B["sw"], B["sd"]) + 0.4
            d = smin(d, capsule(P, B["socket_c"], B["fr"].o + B["fr"].t * (0.12 * B["L"]), core_r), 0.6)
        d = smin(d, plug_sdf(P, cfg), 1.6 if cfg.get("collar") else 1.0)
        return d


# ---------------------------------------------------------------- meshing

def mesh_sdf(obj, res, faces):
    lo, hi = obj.bounds()
    # offset the grid by an odd fraction of a voxel so flat faces (plug top
    # and bottom at z = 0 and z = -plug_len) never fall exactly on grid planes
    lo = lo - 2 * res - 0.3719 * res
    hi = hi + 2 * res
    nx, ny, nz = (np.ceil((hi - lo) / res).astype(int) + 1)
    xs = lo[0] + np.arange(nx) * res
    ys = lo[1] + np.arange(ny) * res
    zs = lo[2] + np.arange(nz) * res
    vol = np.empty((nx, ny, nz), np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    step = max(1, int(4e6 // (nx * ny)))
    for k0 in range(0, nz, step):
        k1 = min(nz, k0 + step)
        Z = np.repeat(zs[k0:k1], len(X))
        P = np.column_stack([np.tile(X, k1 - k0), np.tile(Y, k1 - k0), Z])
        vol[:, :, k0:k1] = obj.sdf(P).reshape(k1 - k0, nx, ny).transpose(1, 2, 0)
    vol[np.abs(vol) < 1e-6] = 1e-6
    vol[0], vol[-1], vol[:, 0], vol[:, -1], vol[:, :, 0], vol[:, :, -1] = (1.0,) * 6
    v, f, _, _ = marching_cubes(vol, level=0.0, spacing=(res, res, res))
    v = v + lo
    full = trimesh.Trimesh(v, f[:, ::-1], process=True)
    if not full.is_watertight:
        # marching cubes can leave an edge shared by three triangles where a fine
        # groove meets a tight bend: repair it (local, a fraction of a millimetre)
        import pymeshfix
        mf = pymeshfix.MeshFix(np.asarray(full.vertices), np.asarray(full.faces))
        mf.repair()
        full = trimesh.Trimesh(mf.points, mf.faces, process=True)
    m = full
    if faces and len(full.faces) > faces:
        import fast_simplification
        # simplification now and then leaves an edge shared by three triangles:
        # try a slightly different target, and keep the full mesh if none is clean
        for target in (faces, int(faces * 0.97), int(faces * 1.03), int(faces * 0.9)):
            v2, f2 = fast_simplification.simplify(full.vertices, full.faces, 1 - target / len(full.faces))
            cand = trimesh.Trimesh(v2, f2, process=True)
            parts = cand.split(only_watertight=False)
            cand = max(parts, key=lambda p: len(p.faces))   # drop stray single triangles
            if not cand.is_watertight:
                import pymeshfix
                mf = pymeshfix.MeshFix(np.asarray(cand.vertices), np.asarray(cand.faces))
                mf.repair()
                cand = trimesh.Trimesh(mf.points, mf.faces, process=True)
            if cand.is_watertight:
                m = cand
                break
    else:
        parts = m.split(only_watertight=False)
        m = max(parts, key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


# ---------------------------------------------------------------- ring fit

def ring_id_mm(us):
    """Inner diameter for a US ring size (standard 0.8128 mm per size)."""
    return 11.63 + 0.8128 * us


def spine_points(obj):
    """Polyline from the tip down to the cup rim (z = 0)."""
    if isinstance(obj, CandleFinger):
        # start on the centre line at the fingertip, so the ring begins square to the finger
        pts = list(obj.spine_path)
    else:
        pts = [obj.tip] + [B["fr"].o for B in reversed(obj.bones)] + [obj.mc_c]
    pts = [np.asarray(p, float) for p in pts]
    # end the path at the rim
    if pts[-1][2] < 0:
        a, b = pts[-2], pts[-1]
        s = a[2] / (a[2] - b[2])
        pts[-1] = a + (b - a) * s
    else:
        pts.append(np.array([0.0, 0.0, 0.0]))
    return pts


def ring_map(mesh, obj, band=3.0, step=0.25, path=None):
    """Slide a rigid ring (band width `band`) down the finger from the tip.

    At each position along the spine the ring axis follows the spine (blended
    over joints), the finger section inside the band is projected onto the
    ring plane and its smallest enclosing circle is measured. A ring rests at
    the first position from the tip where that circle is wider than its inner
    diameter. The ring is not allowed to tilt, so real rings may get a little
    further over a bent joint."""
    import shapely
    from scipy.spatial import ConvexHull
    pts = path if path is not None else spine_points(obj)
    segs = list(zip(pts[:-1], pts[1:]))
    lens = [np.linalg.norm(b - a) for a, b in segs]
    total = sum(lens)
    S = np.arange(0.0, total, step)
    cum = np.concatenate([[0], np.cumsum(lens)])
    V = mesh.vertices
    out = []
    for s in S:
        i = min(np.searchsorted(cum, s, side="right") - 1, len(segs) - 1)
        a, b = segs[i]
        C = a + (b - a) * ((s - cum[i]) / lens[i])
        # blend direction near joints (within 3 mm)
        t = (b - a) / lens[i]
        if i > 0 and s - cum[i] < 3.0:
            tp = (segs[i - 1][1] - segs[i - 1][0]) / lens[i - 1]
            w = 0.5 * (1 - (s - cum[i]) / 3.0)
            t = (1 - w) * t + w * tp
        if i < len(segs) - 1 and cum[i + 1] - s < 3.0:
            tn = (segs[i + 1][1] - segs[i + 1][0]) / lens[i + 1]
            w = 0.5 * (1 - (cum[i + 1] - s) / 3.0)
            t = (1 - w) * t + w * tn
        t = t / np.linalg.norm(t)
        rel = V - C
        ax = rel @ t
        # only material near the ring counts: other parts of a curled finger
        # can cross the ring's plane further away without touching it
        sel = (np.abs(ax) < band / 2) & (np.einsum("ij,ij->i", rel, rel) < 15.0 ** 2)
        if sel.sum() < 3:
            out.append((s, C[2], 0.0, *C, *t))
            continue
        q = rel[sel] - np.outer(ax[sel], t)
        e1 = np.cross(t, [1, 0, 0] if abs(t[0]) < 0.9 else [0, 1, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        q2 = np.column_stack([q @ e1, q @ e2])
        try:
            hull = q2[ConvexHull(q2).vertices]
        except Exception:
            hull = q2
        rad = shapely.minimum_bounding_radius(shapely.MultiPoint(hull))
        out.append((s, C[2], 2 * rad, *C, *t))
    prof = np.array(out)
    rows = []
    for us in np.arange(3, 13.5, 1.0):
        D = ring_id_mm(us)
        hit = np.nonzero(prof[:, 2] > D)[0]
        if len(hit) == 0:
            rows.append(dict(us=float(us), id_mm=round(D, 1), rests="falls to the cup rim", z_mm=0.0))
            continue
        j = hit[0]
        if j == 0:
            rows.append(dict(us=float(us), id_mm=round(D, 1), rests="does not go over the tip", z_mm=None))
            continue
        # the ring sits just above the first position where it touches
        k = max(j - 1, 0)
        rows.append(dict(us=float(us), id_mm=round(D, 1), rests="rests on the finger",
                         z_mm=round(float(prof[k, 1]), 1), along_mm=round(float(prof[k, 0]), 1),
                         centre=[round(float(v), 2) for v in prof[k, 3:6]],
                         axis=[round(float(v), 3) for v in prof[k, 6:9]]))
    return rows, prof


# ---------------------------------------------------------------- checks

def check(mesh, cfg):
    info = dict(watertight=bool(mesh.is_watertight), faces=int(len(mesh.faces)),
                volume_cm3=round(float(mesh.volume) / 1000, 2),
                height_mm=round(float(mesh.bounds[1][2] - mesh.bounds[0][2]), 1),
                above_cup_mm=round(float(mesh.bounds[1][2]), 1))
    # plug diameter measured on the mesh at mid plug, about the plug axis:
    # the smallest radius is the plain cylinder, the largest is over a rib
    z = -cfg["plug_len"] / 2
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is not None:
        pts = np.vstack(sec.discrete)
        r = np.hypot(pts[:, 0], pts[:, 1])
        info["plug_d_mm"] = round(2 * float(r.min()), 2)
        info["plug_d_ribs_mm"] = round(2 * float(r.max()), 2)
    # overhangs steeper than 50 degrees above the plug (need supports)
    n = mesh.face_normals
    c = mesh.triangles_center
    steep = (n[:, 2] < -math.cos(math.radians(40))) & (c[:, 2] > 0.5)
    info["steep_overhang_mm2"] = round(float(mesh.area_faces[steep].sum()), 0)
    # centre of mass offset (for stability with rings on)
    com = mesh.center_mass
    info["com_offset_mm"] = round(float(np.hypot(com[0], com[1])), 1)
    return info


# ---------------------------------------------------------------- presets

BASE = dict(cup=22.0, clearance=0.0, plug_len=18.0, rib=0.3, collar=False, collar_d=26.5,
            collar_t=1.6, scale=1.0)

PRESETS = {
    "candle-straight": dict(kind="candle", prox_len=42.0, width=20.0, mcp=-3.0, pip=14.0, dip=10.0,
                            tilt=0.0, pip_side=-3.0, dip_side=2.5, wick_hole=2.2, wick_depth=10.0),
    "candle-crooked": dict(kind="candle", prox_len=40.0, width=19.5, mcp=4.0, pip=22.0, dip=30.0,
                           tilt=5.0, pip_side=-6.0, dip_side=4.0, wick_hole=2.2, wick_depth=10.0),
    "candle-slim": dict(kind="candle", prox_len=36.0, width=17.0, mcp=-3.0, pip=12.0, dip=14.0,
                        tilt=-4.0, pip_side=3.0, dip_side=-2.0, wick_hole=2.2, wick_depth=9.0,
                        scale=0.92),
    "bone-straight": dict(kind="bone", scale=1.3, joint_gap=0.7, mc_height=1.5, mcp=4.0, pip=10.0,
                          dip=14.0, tilt=0.0, pip_side=-3.0, dip_side=3.0, collar=True),
    "bone-curled": dict(kind="bone", scale=1.3, joint_gap=0.7, mc_height=1.5, mcp=8.0, pip=40.0,
                        dip=55.0, tilt=3.0, pip_side=-4.0, dip_side=2.0, collar=True),
    "bone-hook": dict(kind="bone", scale=1.3, joint_gap=0.7, mc_height=1.5, mcp=-6.0, pip=25.0,
                      dip=75.0, tilt=-6.0, pip_side=5.0, dip_side=-3.0, collar=True),
}
FEMALE = {
    "fem-straight": dict(kind="candle", form="female", prox_len=48.0, width=17.5, mcp=-2.0, pip=12.0, dip=8.0,
                         tilt=0.0, pip_side=-3.0, dip_side=3.0, wick_hole=2.0, wick_depth=10.0),
    "fem-curved": dict(kind="candle", form="female", prox_len=47.0, width=17.0, mcp=3.0, pip=24.0, dip=18.0,
                       tilt=6.0, pip_side=-5.0, dip_side=3.0, wick_hole=2.0, wick_depth=10.0),
    "fem-pinky": dict(kind="candle", form="female", prox_len=38.0, width=15.5, mcp=-4.0, pip=10.0, dip=14.0,
                      tilt=-6.0, pip_side=4.0, dip_side=-2.0, wick_hole=2.0, wick_depth=9.0, scale=0.9),
}
PRESETS.update(FEMALE)
# every wax finger also comes with melted wax: a film over the tip, drips down the
# sides and a puddle on the cup rim
for _n in [k for k, v in PRESETS.items() if v["kind"] == "candle"]:
    PRESETS[_n + "-drips"] = dict(PRESETS[_n], drips=7, drip_seed=len(_n), collar=True)
for _n in FEMALE:   # the smooth, idealised version for a porcelain finish
    PRESETS[_n + "-smooth"] = dict(PRESETS[_n], detail=0.12)


def build(name, over, res, faces, outdir):
    cfg = dict(BASE)
    cfg.update(PRESETS[name])
    cfg.update({k: v for k, v in over.items() if v is not None})
    obj = CandleFinger(cfg) if cfg["kind"] == "candle" else BoneFinger(cfg)
    t0 = time.time()
    mesh = mesh_sdf(obj, res, faces)
    info = check(mesh, cfg)
    rows, prof = ring_map(mesh, obj)
    path = os.path.join(outdir, f"{name}.stl")
    mesh.export(path)
    info.update(name=name, file=os.path.basename(path), seconds=round(time.time() - t0, 1),
                config=cfg, rings=rows)
    if cfg["kind"] == "candle" and cfg.get("wick_hole", 0) > 0:
        info["wick"] = dict(at=[round(float(v), 3) for v in obj.tip_point],
                            dir=[round(float(v), 4) for v in obj.frames[2].t])
    with open(os.path.join(outdir, f"{name}.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    np.savetxt(os.path.join(outdir, f"{name}.profile.csv"), prof[:, :3], delimiter=",",
               header="along_spine_mm,height_above_rim_mm,section_circle_d_mm", fmt="%.2f")
    return info


def rerings(name, outdir):
    """Recompute where rings rest for an STL already built (after a change to the ring check)."""
    js = os.path.join(outdir, f"{name}.json")
    info = json.load(open(js))
    cfg = info["config"]
    obj = CandleFinger(cfg) if cfg["kind"] == "candle" else BoneFinger(cfg)
    mesh = trimesh.load(os.path.join(outdir, f"{name}.stl"))
    rows, prof = ring_map(mesh, obj)
    info["rings"] = rows
    json.dump(info, open(js, "w"), indent=1)
    np.savetxt(os.path.join(outdir, f"{name}.profile.csv"), prof[:, :3], delimiter=",",
               header="along_spine_mm,height_above_rim_mm,section_circle_d_mm", fmt="%.2f")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(PRESETS), help="presets to build")
    ap.add_argument("--cup", type=float, help="cup bore in mm (plug diameter before ribs)")
    ap.add_argument("--rib", type=float, help="crush rib height in mm (0 for none)")
    ap.add_argument("--plug-len", type=float, dest="plug_len", help="plug length in mm")
    ap.add_argument("--scale", type=float, help="overall size factor")
    ap.add_argument("--collar", choices=("on", "off"), help="wax collar on the cup rim")
    ap.add_argument("--res", type=float, default=0.2, help="voxel size in mm (smaller = finer)")
    ap.add_argument("--faces", type=int, default=300000, help="triangle budget per STL")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--rings-only", action="store_true", help="only redo the ring check on built STLs")
    a = ap.parse_args(argv)
    if a.rings_only:
        for name in a.only or PRESETS:
            rows = rerings(name, a.out)
            print(name, ", ".join(f"US{int(r['us'])}:{r['z_mm']}" for r in rows))
        return
    os.makedirs(a.out, exist_ok=True)
    over = dict(cup=a.cup, rib=a.rib, plug_len=a.plug_len, scale=a.scale,
                collar=None if a.collar is None else a.collar == "on")
    for name in a.only or PRESETS:
        info = build(name, over, a.res, a.faces, a.out)
        rings = ", ".join(f"US{int(r['us'])}:{r['z_mm']}" for r in info["rings"] if r["z_mm"])
        print(f"{name}: {info['faces']} faces, watertight={info['watertight']}, "
              f"{info['volume_cm3']} cm3, {info['above_cup_mm']} mm above rim, "
              f"plug {info.get('plug_d_mm')} / ribs {info.get('plug_d_ribs_mm')} mm, "
              f"overhang {info['steep_overhang_mm2']} mm2, {info['seconds']} s\n  rings rest at (mm above rim): {rings}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
