#!/usr/bin/env python3
"""Curling wax fingers: rigid pieces with the joints hidden in the knuckle creases.

Each slender finger is printed straight in three pieces (base with the plug, middle,
fingertip) and joined at the two knuckles by short pins of 1.75 mm filament.

How a joint stays closed while it bends. The hinge axis runs sideways through the
finger, a little inside the palm skin. Everything that moves against something else
is round about that axis, so turning the joint only slides surfaces along each other:

  - On the back, the lower piece ends in a rounded knuckle (a surface of revolution
    about the axis, as wide as the finger) and the piece above cups over it. As the
    finger curls the knuckle comes into view, as a real one does.
  - On the palm side the two pieces meet in a deepened crease: a narrow V that closes
    as the finger curls, so the palm side never collides.
  - Inside, a tab on the upper piece sits between two knuckle rings of the lower piece,
    and the pin runs through all three. It goes in from one side and stops inside, so
    the only hole to fill is one pinhead on the side of each knuckle.

    python curl.py                         # all curling fingers into out/curl/
    python curl.py --only fem-straight --res 0.25
"""
import argparse
import json
import math
import os

import numpy as np
import trimesh

import fingers as F
from poseable import sector

HERE = os.path.dirname(os.path.abspath(__file__))

FINGERS = {   # built straight; the curl comes from the joints
    "fem-straight": dict(F.FEMALE["fem-straight"]),
    "fem-pinky": dict(F.FEMALE["fem-pinky"]),
}
for _v in FINGERS.values():
    _v.update(mcp=0.0, pip=0.0, dip=0.0, tilt=0.0, pip_side=0.0, dip_side=0.0)

POSES = {"straight": (0.0, 0.0), "relaxed": (25.0, 15.0), "curled": (60.0, 50.0), "hooked": (72.0, 70.0)}


class CurlJoint:
    """Hinge at one knuckle of a straight finger (the finger runs up +z)."""

    def __init__(self, finger, name, pin_hole, alpha=35.0, axis_depth=2.0, gap=0.3):
        j = finger.joint[name]
        sj = finger.Sv[j]
        self.a = float(np.interp(sj, *finger.a))          # half width at the joint
        self.cd = float(np.interp(sj, *finger.cd))        # back half depth
        self.cp = float(np.interp(sj, *finger.cp))        # palm half depth
        self.y0 = -(self.cp - axis_depth)                 # axis a little inside the palm skin
        self.o = finger.C[j] + np.array([0.0, self.y0, 0.0])
        self.alpha = alpha                                # half opening of the palm crease
        self.gap = gap
        self.pin = pin_hole
        self.Rk = 1.7                                     # knuckle rings and tab round the pin
        self.tab_w = 2.6
        self.slot_w = self.tab_w + 0.5
        # width of the finger at the axis height, for the pin's blind end
        self.x_side = self.a * (1 - (abs(self.y0) / self.cp) ** 2.0) ** 0.5

    def local(self, P):
        return P - self.o

    def r_cut(self, x):
        """Radius of the back knuckle at sideways position x: just outside the back skin."""
        q = np.clip(np.abs(x) / self.a, 0, 1)
        ytop = self.cd * (1 - q ** 2.3) ** (1 / 2.3)
        return ytop - self.y0 + 0.25

    def regions(self, P):
        """Signed distances (negative inside) of the lower piece's and the upper piece's regions."""
        L = self.local(P)
        x, y, z = L[:, 0], L[:, 1], L[:, 2]
        r = np.hypot(y, z)
        g = self.gap
        rc = self.r_cut(x)
        Q = np.column_stack([x, y, z])
        drum = np.maximum(r - rc, -y)                                  # back half, inside the knuckle
        rings = np.maximum(r - self.Rk, self.slot_w / 2 - np.abs(x))   # knuckle rings round the pin
        low_ang = sector(Q, 0.0, 90.0 + self.alpha, 270.0, 1e6)        # below the joint, palm crease face
        lower = np.minimum(np.minimum(drum, rings), low_ang)
        # room for the tab to turn
        tab_room = np.maximum(np.abs(x) - self.slot_w / 2, r - (self.Rk + g))
        lower = np.maximum(lower, -tab_room)
        up_ang = sector(Q, 0.0, -90.0, 90.0 - self.alpha, 1e6)
        upper = np.maximum(up_ang, -(np.maximum(r - (rc + g), -y - g)))   # clear of the back knuckle
        upper = np.maximum(upper, -(np.maximum(r - (self.Rk + g), self.slot_w / 2 - g - np.abs(x))))
        tab = np.maximum(r - self.Rk, np.abs(x) - self.tab_w / 2)
        upper = np.minimum(upper, tab)
        return lower, upper

    def pin_hole(self, P):
        L = self.local(P)
        r = np.hypot(L[:, 1], L[:, 2])
        return np.maximum(r - self.pin / 2, -(self.x_side - 1.2) - L[:, 0])   # open on +x, blind on -x


class Piece:
    def __init__(self, finger, cfg, below, above):
        self.f, self.cfg, self.below, self.above = finger, cfg, below, above

    def bounds(self):
        return self.f.bounds()

    def sdf(self, P):
        d = self.f.flesh(P)
        # soft edges along the seams, so they read as creases, not cuts
        if self.below is not None:
            d = F.smax(d, self.below.regions(P)[1], 0.5)
        if self.above is not None:
            d = F.smax(d, self.above.regions(P)[0], 0.5)
        if self.below is None:
            d = F.smin(d, F.plug_sdf(P, self.cfg), 3.2)
        if self.above is None and self.cfg.get("wick_hole", 0) > 0:
            h = F.hole_along(P, self.f.tip_point, self.f.T[-1], self.cfg["wick_hole"] / 2, self.cfg["wick_depth"])
            d = F.smax(d, -h, 0.25)
        for j in (self.below, self.above):
            if j is not None:
                d = np.maximum(d, -j.pin_hole(P))
        return d


def rotation_about(o, deg):
    """4x4: turn by deg about the sideways axis through o, bending +z towards -y (the palm)."""
    a = math.radians(deg)
    R = np.eye(4)
    R[:3, :3] = [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    T1, T2 = np.eye(4), np.eye(4)
    T1[:3, 3] = -o
    T2[:3, 3] = o
    return T2 @ R @ T1


def range_of_motion(lower, upper, upper_mesh, joint, tol=0.1):
    V = upper_mesh.vertices
    near = np.linalg.norm(V - joint.o, axis=1) < 14
    V = V[near]
    good = {}
    for a in np.arange(-10, 100.01, 2.5):
        M = rotation_about(joint.o, a)
        Pw = V @ M[:3, :3].T + M[:3, 3]
        good[float(a)] = bool(lower.sdf(Pw).min() > -tol)
    hi = 0.0
    while good.get(hi + 2.5):
        hi += 2.5
    lo = 0.0
    while good.get(lo - 2.5):
        lo -= 2.5
    return (lo, hi) if good.get(0.0) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(FINGERS))
    ap.add_argument("--pin-hole", type=float, default=1.8)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--faces", type=int, default=200000)
    ap.add_argument("--smooth", action="store_true", help="smooth skin for a porcelain finish")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "curl"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.only or FINGERS:
        cfg = dict(F.BASE)
        cfg.update(FINGERS[name])
        if a.smooth:
            cfg["detail"] = 0.12
        tag = name + ("-smooth" if a.smooth else "")
        finger = F.CandleFinger(cfg)
        pip, dip = CurlJoint(finger, "pip", a.pin_hole), CurlJoint(finger, "dip", a.pin_hole)
        pieces = {"base": Piece(finger, cfg, None, pip), "middle": Piece(finger, cfg, pip, dip),
                  "tip": Piece(finger, cfg, dip, None)}
        meshes, report = {}, dict(finger=tag, pin_hole_mm=a.pin_hole, parts={}, joints={}, poses={})
        for pn, pc in pieces.items():
            m = F.mesh_sdf(pc, a.res, a.faces)
            meshes[pn] = m
            m.export(os.path.join(a.out, f"{tag}-{pn}.stl"))
            report["parts"][pn] = dict(file=f"{tag}-{pn}.stl", watertight=bool(m.is_watertight),
                                       volume_cm3=round(float(m.volume) / 1000, 2))
            print(tag, pn, report["parts"][pn])
        for jn, j, lo, up in (("middle knuckle", pip, "base", "middle"), ("end knuckle", dip, "middle", "tip")):
            rom = range_of_motion(pieces[lo], pieces[up], meshes[up], j)
            report["joints"][jn] = dict(bends_deg=rom, pin_length_mm=round(2 * j.x_side - 1.2 - 0.6, 1),
                                        axis=[round(float(v), 3) for v in j.o], crease_gap_mm=round(
                                            2.0 * math.radians(2 * j.alpha), 1))
            print(tag, jn, report["joints"][jn])
        for pose, (p, d) in POSES.items():
            Mp = rotation_about(pip.o, p)
            Md = Mp @ rotation_about(dip.o, d)
            parts = [meshes["base"].copy(), meshes["middle"].copy(), meshes["tip"].copy()]
            parts[1].apply_transform(Mp)
            parts[2].apply_transform(Md)
            whole = trimesh.util.concatenate(parts)
            tip = (Md @ [*finger.C[-1], 1])[:3]
            path = [tip, (Md @ [*dip.o, 1])[:3], (Mp @ [*pip.o, 1])[:3], np.zeros(3)]
            rows, _ = F.ring_map(whole, None, path=path)
            within = all(r["bends_deg"] and r["bends_deg"][0] <= ang <= r["bends_deg"][1]
                         for r, ang in zip(report["joints"].values(), (p, d)))
            report["poses"][pose] = dict(angles_deg=[p, d], reachable=within,
                                         above_cup_mm=round(float(whole.bounds[1][2]), 1), rings=rows)
            print(tag, pose, (p, d), "reachable", within)
        report["pivots"] = [[round(float(v), 3) for v in pip.o], [round(float(v), 3) for v in dip.o]]
        with open(os.path.join(a.out, f"{tag}.json"), "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
