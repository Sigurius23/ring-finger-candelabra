#!/usr/bin/env python3
"""Build an interactive 3D preview page: fingers standing in candlesticks with rings on.

    python preview.py                      # writes preview.html next to this file

The candelabra is a generic three-arm stand drawn in the page itself, with
22.6 mm cups, only there to show the fingers at scale.
"""
import argparse
import base64
import glob
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))

SCENES = {
    "curl": dict(title="Curling", finish="porcelain", slots=[
        dict(kit="fem-straight-smooth", pose="curled", cup=0, rot=25, rings=[(6, "gold", True)]),
        dict(kit="fem-straight-smooth", pose="relaxed", cup=1, rot=-5, rings=[(5, "gold", True), (7, "gold", False)], sliders=True),
        dict(kit="fem-pinky-smooth", pose="hooked", cup=2, rot=-30, rings=[(4, "gold", True)]),
    ]),
    "fem": dict(title="Female", finish="gloss", slots=[
        dict(finger="fem-curved", cup=0, rot=25, rings=[(5, "gold", True), (7, "silver", False)]),
        dict(finger="fem-straight", cup=1, rot=-5, rings=[(4, "silver", True), (6, "gold", False)]),
        dict(finger="fem-pinky", cup=2, rot=-30, rings=[(3, "gold", True)]),
    ]),
    "fem-drips": dict(title="Female, drips", finish="gloss", slots=[
        dict(finger="fem-curved-drips", cup=0, rot=25, rings=[(6, "gold", True)]),
        dict(finger="fem-straight-drips", cup=1, rot=-5, rings=[(5, "silver", True), (7, "gold", False)]),
        dict(finger="fem-pinky-drips", cup=2, rot=-30, rings=[(4, "gold", False)]),
    ]),
    "porcelain": dict(title="Porcelain", finish="porcelain", slots=[
        dict(finger="fem-curved-smooth", cup=0, rot=25, rings=[(5, "gold", True), (7, "gold", False)]),
        dict(finger="fem-straight-smooth", cup=1, rot=-5, rings=[(4, "gold", True), (6, "gold", False)]),
        dict(finger="fem-pinky-smooth", cup=2, rot=-30, rings=[(3, "gold", True)]),
    ]),
    "wax": dict(title="Male, drips", slots=[
        dict(finger="candle-crooked-drips", cup=0, rot=25, rings=[(7, "gold", True), (9, "silver", False)]),
        dict(finger="candle-straight-drips", cup=1, rot=-5, rings=[(8, "silver", True), (10, "gold", False)]),
        dict(finger="candle-slim-drips", cup=2, rot=-30, rings=[(3, "gold", False), (6, "silver", True)]),
    ]),
    "plain": dict(title="Male", slots=[
        dict(finger="candle-crooked", cup=0, rot=25, rings=[(6, "gold", True), (9, "silver", False)]),
        dict(finger="candle-straight", cup=1, rot=-5, rings=[(5, "silver", True), (8, "gold", False)]),
        dict(finger="candle-slim", cup=2, rot=-30, rings=[(3, "gold", False), (4, "silver", True)]),
    ]),
    "bone": dict(title="Poseable bones", slots=[
        dict(kit="bone", pose="curled", cup=0, rot=30, rings=[(8, "gold", True), (5, "silver", False)]),
        dict(kit="bone", pose="straight", cup=1, rot=0, rings=[(6, "gold", True), (9, "silver", False)], sliders=True),
        dict(kit="bone", pose="hook", cup=2, rot=-35, rings=[(4, "silver", True), (10, "gold", False)]),
    ]),
}


def pack(mesh, faces=None):
    """Indexed mesh: positions quantised to uint16 over the bounds, uint32 indices."""
    if faces and len(mesh.faces) > faces:
        import fast_simplification
        v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, 1 - faces / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=True)
    lo, hi = mesh.bounds
    span = np.maximum(hi - lo, 1e-6)
    q = np.round((mesh.vertices - lo) / span * 65535).astype("<u2")
    return dict(lo=lo.round(4).tolist(), span=span.round(4).tolist(),
                pos=base64.b64encode(q.tobytes()).decode(),
                idx=base64.b64encode(mesh.faces.astype("<u4").tobytes()).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "preview.html"))
    ap.add_argument("--stl", default=os.path.join(HERE, "out"))
    ap.add_argument("--faces", type=int, default=70000)
    ap.add_argument("--poseable", default=os.path.join(HERE, "out", "poseable"), help="output of poseable.py")
    ap.add_argument("--curl", default=os.path.join(HERE, "out", "curl"), help="output of curl.py")
    ap.add_argument("--local-three", action="store_true", help="use ./three for offline screenshots")
    ap.add_argument("--split", action="store_true",
                    help="write the meshes to preview-data.json beside the page instead of inline")
    a = ap.parse_args()

    data = dict(fingers={}, scenes=SCENES)
    data["kits"] = {}
    # the bone kit: parts are stored in their own frames, so place them at rest (straight, stacked)
    rep = json.load(open(os.path.join(a.poseable, "poseable.json")))
    z, pivots, parts = 0.0, [], []
    for i, n in enumerate(("base", "proximal", "middle", "distal")):
        m = trimesh.load(os.path.join(a.poseable, f"part-{n}.stl"))
        m.apply_translation([0, 0, z])
        parts.append(dict(name=n, file=f"bone-{n}.stl", mesh=pack(m, min(a.faces, 30000))))
        if i < 3:
            z += rep["pivots_mm"][i]
            pivots.append([0.0, 0.0, round(z, 3)])
    data["kits"]["bone"] = dict(
        title="Poseable bones", kind="bone", parts=parts, pivots=pivots, pin_hole=rep["pin_hole_mm"],
        joints=[dict(name=k, **v) for k, v in rep["joints"].items()],
        poses={k: dict(angles=v["angles_deg"], rings=v["rings"], above=v["above_cup_mm"]) for k, v in rep["poses"].items()},
        volume=round(sum(v["volume_cm3"] for v in rep["parts"].values()), 2),
        print="Supports under each bone, lying on its side")
    # curling wax fingers: parts are already in their rest position
    for js in sorted(glob.glob(os.path.join(a.curl, "*.json"))):
        rep = json.load(open(js))
        tag = rep["finger"]
        names = ("base", "middle", "tip")
        cfgw = json.load(open(os.path.join(a.stl, tag.replace("-smooth", "") + ".json"))).get("wick") \
            if os.path.exists(os.path.join(a.stl, tag.replace("-smooth", "") + ".json")) else None
        data["kits"][tag] = dict(
            title="Curling " + ("smooth " if "smooth" in tag else "") + ("little finger" if "pinky" in tag else "finger"),
            kind="candle", pin_hole=rep["pin_hole_mm"], pivots=rep["pivots"],
            parts=[dict(name=n, file=f"{tag}-{n}.stl", mesh=pack(trimesh.load(os.path.join(a.curl, f"{tag}-{n}.stl")), min(a.faces, 40000)))
                   for n in names],
            joints=[dict(name=k, **v) for k, v in rep["joints"].items()],
            poses={k: dict(angles=v["angles_deg"], rings=v["rings"], above=v["above_cup_mm"]) for k, v in rep["poses"].items()},
            volume=round(sum(v["volume_cm3"] for v in rep["parts"].values()), 2),
            print="Each piece upright, no supports", wick=None)
    for scene in SCENES.values():
        for slot in scene["slots"]:
            if "finger" not in slot:
                continue
            name = slot["finger"]
            if name in data["fingers"]:
                continue
            m = trimesh.load(os.path.join(a.stl, f"{name}.stl"))
            info = json.load(open(os.path.join(a.stl, f"{name}.json")))
            data["fingers"][name] = dict(mesh=pack(m, a.faces), kind=info["config"]["kind"],
                                         rings=info["rings"], above=info["above_cup_mm"],
                                         volume=info["volume_cm3"], wick=info.get("wick"),
                                         drips=bool(info["config"].get("drips")))
    tpl = open(os.path.join(HERE, "preview_template.html")).read()
    three = ("three/three.min.js", "three/OrbitControls.js") if a.local_three else (
        "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
        "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js")
    blob = json.dumps(data, separators=(",", ":"))
    if a.split:
        with open(os.path.join(os.path.dirname(a.out), "preview-data.json"), "w") as fh:
            fh.write(blob)
        blob = "null"
    html = tpl.replace("__THREE__", three[0]).replace("__ORBIT__", three[1]).replace("__DATA__", blob)
    open(a.out, "w").write(html)
    print(a.out, round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
