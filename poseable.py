#!/usr/bin/env python3
"""Poseable bone finger: four printed parts joined by 1.75 mm filament pins.

Parts (each in its own frame: lower joint axis on the x axis through the origin,
the bone running up +z, the back of the finger towards +y):

  base      the plug for the candle cup, the wax collar and the knuckle head of
            the hand bone (metacarpal), standing upright as printed
  proximal  first finger bone
  middle    second finger bone
  distal    fingertip bone

Every joint is a hinge hidden inside the bones. The bone below ends in two rounded
knobs (condyles) with a thin valley between them; inside, the valley widens into a
slot for a tab on the base of the bone above. A pin of 1.75 mm filament goes in from
one side through the first knob and the tab and stops inside the second knob, so no
hardware shows once the single entry hole is filled and painted. The holes are sized
for a snug pin, and that friction holds the pose.

    python poseable.py                 # print parts, posed previews and a report into out/poseable/
    python poseable.py --pin-hole 1.85 # looser joints
"""
import argparse
import json
import math
import os

import numpy as np
import trimesh

import fingers as F

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1.3                       # same enlargement as the one-piece bone fingers
G = 0.88                      # slimmer, more delicate bones: widths and depths only, not lengths
# length, base width, base depth, shaft width, shaft depth, head width, head depth (real size)
BONES = {"proximal": (40.0, 15.0, 12.0, 8.6, 6.8, 11.6, 9.0),
         "middle": (24.0, 12.4, 9.6, 7.0, 5.6, 9.4, 7.4),
         "distal": (18.0, 10.0, 7.6, 5.0, 4.2, 7.4, 3.6)}
ORDER = ["base", "proximal", "middle", "distal"]

POSES = {  # knuckle, middle joint, end joint, in degrees towards the palm
    "straight": (4.0, 10.0, 14.0),
    "curled": (8.0, 40.0, 55.0),
    "hook": (0.0, 25.0, 75.0),
}


def wheel(P, centre_x, R, half_w, round_r, z0):
    """Rounded disc on the x axis (a condyle): radius R, half width half_w, centre (centre_x, 0, z0)."""
    r = np.hypot(P[:, 1], P[:, 2] - z0)
    dx = r - R + round_r
    dy = np.abs(P[:, 0] - centre_x) - half_w + round_r
    return np.minimum(np.maximum(dx, dy), 0.0) + np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) - round_r


def sector(P, z0, a_lo, a_hi, r_max):
    """Inside (negative) for points around the x axis at height z0 whose angle, measured from +z
    towards -y (the palm), is between a_lo and a_hi degrees, out to radius r_max."""
    y, z = P[:, 1], P[:, 2] - z0
    ang = np.degrees(np.arctan2(-y, z))
    r = np.hypot(y, z)
    mid, half = (a_lo + a_hi) / 2, (a_hi - a_lo) / 2
    da = np.abs((ang - mid + 180) % 360 - 180) - half
    return np.maximum(da * np.pi / 180 * np.maximum(r, 1.0), r - r_max)


class Joint:
    """The hinge between a part below and the part above it."""

    def __init__(self, R, head_w, tab_w, pin_hole, side_clear=0.25, radial_clear=0.45, gap=0.5):
        self.R = R                    # condyle radius
        self.head_w = head_w          # width across both condyles
        self.tab_w = tab_w
        self.slot_w = tab_w + 2 * side_clear
        self.stem_w = 1.6                 # the tab narrows where it leaves the joint ...
        self.outer_w = self.stem_w + 0.4  # ... so the groove that shows on the outside is a thin natural valley
        self.R_tab = 0.62 * R
        self.slot_r = self.R_tab + radial_clear
        self.gap = gap                # between the condyles and the socket of the bone above
        self.pin = pin_hole

    def head(self, P, z0):
        """Two condyles: bulbs that are round about the joint axis (so the bone above can turn
        around them), centred on the axis at height z0."""
        half = self.head_w / 2
        inner = self.slot_w / 2
        hw = (half - inner) / 2
        cx = inner + hw
        d = np.full(len(P), 1e3)
        for sgn in (-1, 1):
            fr = F.Frame([sgn * cx, 0, z0], [0, 0, 1], [0, 1, 0])
            d = F.smin(d, F.ellipsoid(P, fr, hw * 1.25, self.R, self.R, self.R), 0.8)
        return d

    @property
    def outer_x(self):
        inner = self.slot_w / 2
        hw = (self.head_w / 2 - inner) / 2
        return inner + hw + hw * 1.25

    @property
    def pin_length(self):
        # from the entry hole (recessed 0.6 mm for filler) to the blind end inside the far knob
        return (self.outer_x - 0.6) + (self.outer_x - 1.2)

    def cut_lower(self, P, z0, d):
        """Groove for the tab (room to bend from -40 to +115 degrees) and the pin hole.

        Nothing mechanical shows: around the joint axis the groove is as wide as the tab, but
        where it reaches the surface it is only a thin valley between the two knobs, and the
        pin hole opens on one side only (filled and painted over after assembly) and stops
        inside the other knob."""
        r = np.hypot(P[:, 1], P[:, 2] - z0)
        inner = np.maximum(np.abs(P[:, 0]) - self.slot_w / 2, r - self.slot_r)
        swing = sector(P, z0, -40.0, 115.0, self.R + 3.5)
        outer = np.maximum(np.abs(P[:, 0]) - self.outer_w / 2, swing)
        d = F.smax(d, -np.minimum(inner, outer), 0.7)
        hole = np.maximum(r - self.pin / 2, -(self.outer_x - 1.2) - P[:, 0])   # blind on the -x side
        return np.maximum(d, -hole)

    def tab(self, P, top):
        """Tab on the base of the bone above (its own frame: axis through the origin), up to height `top`."""
        half = self.tab_w / 2
        disc = np.maximum(np.hypot(P[:, 1], P[:, 2]) - self.R_tab, np.abs(P[:, 0]) - half)
        stem = np.maximum(np.maximum(np.abs(P[:, 1]) - 0.6 * self.R_tab, np.abs(P[:, 0]) - self.stem_w / 2),
                          np.maximum(-P[:, 2], P[:, 2] - top))
        d = F.smin(disc, stem, 0.4)
        return np.maximum(d, -(np.hypot(P[:, 1], P[:, 2]) - self.pin / 2))

    def socket(self, P):
        """Room for the condyles below, in the frame of the bone above."""
        return np.maximum(np.hypot(P[:, 1], P[:, 2]) - (self.R + self.gap), np.abs(P[:, 0]) - self.head_w / 2 - 3)


class Part:
    def mesh(self, res, faces):
        return F.mesh_sdf(self, res, faces)


class Base(Part):
    """Plug, wax collar and the knuckle head of the hand bone."""

    def __init__(self, cfg, joint):
        self.cfg, self.j = cfg, joint
        self.pivot = cfg["collar_t"] + 0.72 * joint.R          # the knob sits partly sunk in the wax
        self.top = self.pivot

    def bounds(self):
        R = max(self.cfg["cup"], self.cfg["collar_d"]) / 2 + 1.5
        return np.array([-R, -R, -self.cfg["plug_len"] - 1.5]), np.array([R, R, self.pivot + self.j.R + 2])

    def sdf(self, P):
        j, z0 = self.j, self.pivot
        d = j.head(P, z0)
        neck = F.capsule(P, [0, 0, z0 - 1.0], [0, 0, -6.0], 0.62 * j.R)
        d = F.smin(d, neck, 2.0)
        d = j.cut_lower(P, z0, d)
        return F.smin(d, F.plug_sdf(P, self.cfg), 1.6)


class Bone(Part):
    """One finger bone. Its lower joint axis is the x axis; `below` is that joint, `above` the next."""

    def __init__(self, name, below, above):
        self.name = name
        L, bw, bd, sw, sd, hw, hd = (v * S * (1 if k == 0 else G) for k, v in enumerate(BONES[name]))
        self.L, self.bw, self.bd, self.sw, self.sd, self.hw, self.hd = L, bw, bd, sw, sd, hw, hd
        self.below, self.above = below, above
        self.s0 = below.R + below.gap                    # where the base of the bone starts
        self.pivot = self.s0 + L - above.R if above else None
        self.fr = F.Frame([0, 0, self.s0], [0, 0, 1], [0, 1, 0])

    def bounds(self):
        top = self.s0 + self.L + 3
        return np.array([-self.bw / 2 - 3, -self.bd / 2 - 3, -self.below.R - 2]), \
            np.array([self.bw / 2 + 3, self.bd / 2 + 3, top])

    def sdf(self, P):
        fr, L = self.fr, self.L
        bw, bd, sw, sd, hw, hd = self.bw, self.bd, self.sw, self.sd, self.hw, self.hd
        if self.above is not None:
            j = self.above
            run = self.pivot - self.s0
            fa = F.profile([(0, bw * 0.47), (0.12, bw * 0.44), (0.3, sw * 0.62), (0.55, sw * 0.5),
                            (0.85, hw * 0.4), (1, hw * 0.4)])
            fcd = F.profile([(0, bd * 0.5), (0.15, bd * 0.42), (0.4, sd * 0.55), (0.7, sd * 0.52),
                             (1, min(hd * 0.42, j.R * 0.95))])
            fcp = F.profile([(0, bd * 0.5), (0.15, bd * 0.4), (0.4, sd * 0.42), (0.7, sd * 0.45),
                             (1, min(hd * 0.5, j.R * 0.95))])
            d = F.tube(P, fr, run, fa, fcd, fcp, m_pal=2.8)
            d = F.smin(d, j.head(P, self.pivot), 1.2)
            for sgn in (-1, 1):   # tubercles at the sides of the base
                c = fr.o + fr.t * (0.1 * L) + fr.b * (sgn * 0.4 * bw)
                d = F.smin(d, np.linalg.norm(P - c, axis=1) - 0.13 * bw, 1.2)
        else:
            tw = 0.62 * bw
            fa = F.profile([(0, bw * 0.47), (0.15, bw * 0.4), (0.4, sw * 0.55), (0.62, sw * 0.5),
                            (0.82, tw * 0.45), (0.92, tw * 0.47), (1, tw * 0.3)])
            fcd = F.profile([(0, bd * 0.5), (0.2, bd * 0.4), (0.45, sd * 0.5), (0.8, sd * 0.4), (1, sd * 0.3)])
            fcp = F.profile([(0, bd * 0.5), (0.2, bd * 0.38), (0.45, sd * 0.45), (0.8, sd * 0.42), (1, sd * 0.3)])
            d = F.tube(P, fr, L, fa, fcd, fcp, m_pal=2.6)
            tf = F.Frame(fr.o + fr.t * (0.9 * L), fr.t, fr.n)
            d = F.smin(d, F.ellipsoid(P, tf, tw * 0.5, sd * 0.42, sd * 0.38, 0.12 * L), 1.0)
        # base: a wider block that cups the condyles below
        bf = F.Frame(fr.o + fr.t * (0.06 * L), fr.t, fr.n)
        d = F.smin(d, F.ellipsoid(P, bf, bw * 0.5, bd * 0.48, bd * 0.52, 0.11 * L + 1.0), 1.5)
        # a rounded bump on the back of the base (where the extensor tendon attaches on a
        # real bone) that hangs down over the joint and hides it from behind
        j = self.below
        tf = F.Frame([0, j.R + j.gap + 0.4, 0.25 * j.R], [0, 0, 1], [0, 1, 0])
        lip = F.ellipsoid(P, tf, 0.3 * j.head_w, 2.0, 2.0, 0.95 * j.R)
        d = F.smin(d, lip, 2.2)
        d = F.smax(d, -self.below.socket(P), 0.3)
        # nothing below the joint axis but the tab and the back lip
        d = np.maximum(d, np.minimum(-P[:, 2], np.maximum(j.R * 0.5 - P[:, 1], -P[:, 2] - 0.7 * j.R)))
        d = F.smin(d, self.below.tab(P, self.s0 + 1.5), 0.5)
        if self.above is not None:
            d = self.above.cut_lower(P, self.pivot, d)
        return d


def build_parts(cfg, pin_hole):
    s = S
    head_w = {"base": 0.85 * 15.0 * s * G, "proximal": 11.6 * s * G, "middle": 9.4 * s * G}
    R = {"base": 0.32 * 15.0 * s * G, "proximal": 0.42 * 9.0 * s * G, "middle": 0.42 * 7.4 * s * G}
    joints = {k: Joint(R[k], head_w[k], max(2.6, 0.2 * head_w[k]), pin_hole) for k in head_w}
    parts = {"base": Base(cfg, joints["base"]),
             "proximal": Bone("proximal", joints["base"], joints["proximal"]),
             "middle": Bone("middle", joints["proximal"], joints["middle"]),
             "distal": Bone("distal", joints["middle"], None)}
    return parts, joints


def rot_x(deg):
    a = math.radians(deg)
    # positive angles bend +z towards -y (the palm)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def pose_transforms(parts, angles):
    """World transform (4x4) of each part for knuckle, middle and end joint angles."""
    T = {"base": np.eye(4)}
    M = np.eye(4)
    pivots = [parts["base"].pivot, parts["proximal"].pivot, parts["middle"].pivot]
    for name, z0, a in zip(ORDER[1:], pivots, angles):
        step = np.eye(4)
        step[:3, 3] = [0, 0, z0]
        R = np.eye(4)
        R[:3, :3] = rot_x(a)
        M = M @ step @ R
        T[name] = M.copy()
    return T


def range_of_motion(lower, upper, upper_mesh, lower_mesh, z0, tol=0.08):
    """Angles (degrees) at which the two parts of a joint do not overlap, checked every 2.5 degrees."""
    ok = []
    Vu = upper_mesh.vertices
    Vl = lower_mesh.vertices
    near = Vl[np.hypot(Vl[:, 1], Vl[:, 2] - z0) < 25]
    for a in np.arange(-30, 120.01, 2.5):
        Rm = rot_x(a)
        Pu = Vu @ Rm.T + [0, 0, z0]                      # upper part placed in the lower part's frame
        d1 = lower.sdf(Pu).min()
        Pl = (near - [0, 0, z0]) @ Rm                    # lower part in the upper part's frame
        d2 = upper.sdf(Pl).min()
        ok.append((float(a), bool(min(d1, d2) > -tol)))
    # the free range around zero
    angles = [a for a, good in ok]
    good = dict(ok)
    lo = hi = 0.0
    while lo - 2.5 in good and good[lo - 2.5]:
        lo -= 2.5
    while hi + 2.5 in good and good[hi + 2.5]:
        hi += 2.5
    return (lo, hi) if good.get(0.0) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin-hole", type=float, default=1.8, help="pin hole diameter in mm for 1.75 mm filament")
    ap.add_argument("--cup", type=float, default=22.0)
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--faces", type=int, default=200000)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "poseable"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = dict(F.BASE, cup=a.cup, collar=True)
    parts, joints = build_parts(cfg, a.pin_hole)
    meshes = {}
    report = dict(pin_hole_mm=a.pin_hole, parts={}, joints={}, poses={})
    for name in ORDER:
        m = parts[name].mesh(a.res, a.faces)
        meshes[name] = m
        # print orientation: the base stands on its plug; bones lie on their side so the
        # pin holes are vertical and come out round
        pm = m.copy()
        if name != "base":
            pm.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        pm.apply_translation([0, 0, -pm.bounds[0][2]])
        pm.export(os.path.join(a.out, f"bone-{name}.stl"))
        report["parts"][name] = dict(file=f"bone-{name}.stl", watertight=bool(m.is_watertight),
                                     volume_cm3=round(float(m.volume) / 1000, 2),
                                     size_mm=[round(float(v), 1) for v in m.extents])
        print(name, report["parts"][name])
    names = list(zip(ORDER[:-1], ORDER[1:]))
    titles = ["knuckle", "middle joint", "end joint"]
    for (lo_name, up_name), title in zip(names, titles):
        z0 = parts[lo_name].pivot
        rom = range_of_motion(parts[lo_name], parts[up_name], meshes[up_name], meshes[lo_name], z0)
        j = parts[up_name].below
        report["joints"][title] = dict(between=[lo_name, up_name], bends_deg=rom,
                                       pin_length_mm=round(j.pin_length, 1))
        print(title, report["joints"][title])
    # posed assemblies: for the preview and the ring check (not for printing)
    for pose, angles in POSES.items():
        T = pose_transforms(parts, angles)
        posed = []
        for name in ORDER:
            m = meshes[name].copy()
            m.apply_transform(T[name])
            posed.append(m)
        whole = trimesh.util.concatenate(posed)
        whole.export(os.path.join(a.out, f"pose-{pose}.stl"))
        # centre line for the ring check: fingertip, each joint, the rim
        tipT = T["distal"]
        tip = (tipT @ [0, 0, parts["distal"].s0 + parts["distal"].L, 1])[:3]
        path = [tip] + [T[n][:3, 3] for n in ("distal", "middle", "proximal")] + [np.zeros(3)]
        rows, _ = F.ring_map(whole, None, path=path)
        within = {t: (r["bends_deg"] is not None and r["bends_deg"][0] <= ang <= r["bends_deg"][1])
                  for (t, r), ang in zip(report["joints"].items(), angles)}
        report["poses"][pose] = dict(angles_deg=list(angles), file=f"pose-{pose}.stl", reachable=within,
                                     above_cup_mm=round(float(whole.bounds[1][2]), 1), rings=rows)
        print(pose, angles, "reachable:", within, "rings:",
              ", ".join(f"US{int(r['us'])}:{r['z_mm']}" for r in rows if r["z_mm"]))
    # what the preview needs to pose the finger itself
    report["pivots_mm"] = [parts["base"].pivot, parts["proximal"].pivot, parts["middle"].pivot]
    with open(os.path.join(a.out, "poseable.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    for name in ORDER:   # unrotated parts for the preview
        meshes[name].export(os.path.join(a.out, f"part-{name}.stl"))


if __name__ == "__main__":
    main()
