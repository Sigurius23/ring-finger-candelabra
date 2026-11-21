#!/usr/bin/env python3
"""Turn any finger STL into a ring-holder candle finger.

The mesh is filled solid (hollow sleeves, hinge slots and open bases are
closed), stood on a plug for a candle cup, and given a wick hole at the tip.

    python adapt.py finger.stl out/my-finger.stl --cut 2 --cup 22

The input must stand upright with its base at the bottom (as most finger
models are printed). --cut removes that many mm from the bottom first.
"""
import argparse
import json
import os

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

import fingers as F


def to_manifold(tm):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                               tri_verts=np.asarray(tm.faces, np.uint32)))


def to_trimesh(M):
    m = M.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def mc_mesh(field, lo, pitch):
    field = field.copy()
    field[np.abs(field) < 1e-6] = 1e-6
    field[0], field[-1], field[:, 0], field[:, -1], field[:, :, 0], field[:, :, -1] = (1.0,) * 6
    v, f, _, _ = marching_cubes(field, 0.0, spacing=(pitch,) * 3)
    m = trimesh.Trimesh(v + lo, f[:, ::-1], process=True)
    trimesh.repair.fix_normals(m)
    return m


def sdf_mesh(fn, lo, hi, pitch):
    """Mesh the zero set of fn(P) over a box."""
    lo = np.asarray(lo, float) - 0.3719 * pitch
    n = np.ceil((np.asarray(hi) - lo) / pitch).astype(int) + 1
    g = [lo[i] + np.arange(n[i]) * pitch for i in range(3)]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    d = fn(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])).reshape(n).astype(np.float32)
    return mc_mesh(d, lo, pitch)


def solid_occupancy(mesh, pitch, close_mm):
    """Voxels inside the filled-in mesh (hollows, slots and open base closed)."""
    lo = mesh.bounds[0] - 6 * pitch
    hi = mesh.bounds[1] + 6 * pitch
    shape = np.ceil((hi - lo) / pitch).astype(int) + 1
    occ = np.zeros(shape, bool)
    # rasterise the surface densely: sample points on every triangle
    pts, _ = trimesh.sample.sample_surface_even(mesh, int(mesh.area / (pitch * pitch) * 6) + 1000)
    pts = np.vstack([pts, mesh.vertices])
    ijk = np.round((pts - lo) / pitch).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    # The surface raster is a thin shell. Fill it in three ways and combine:
    #  sharp  - whatever the shell already encloses (keeps every surface detail)
    #  closed - the shell closed over gaps of about 2 * close_mm (seals hinge
    #           slots and the open base) and then filled
    #  deep   - only the part of `closed` more than close_mm inside its surface,
    #           so grooves on the outside (nail edges, creases) stay open
    r = max(1.0, close_mm / pitch)
    shell = ndimage.binary_dilation(occ)
    sharp = ndimage.binary_fill_holes(shell)
    dil = ndimage.distance_transform_edt(~shell) <= r
    dil[:, :, :2] = False
    closed = ndimage.binary_fill_holes(dil)
    closed = ndimage.distance_transform_edt(closed) > r
    deep = ndimage.distance_transform_edt(closed) > r
    occ = sharp | (closed & deep)
    return occ, lo


def voxel_route(m, a, tip, t, cfg):
    """For models without a closed surface: rebuild the surface from voxels."""
    occ, lo = solid_occupancy(m, a.pitch, a.close)
    inside = ndimage.distance_transform_edt(occ) * a.pitch
    outside = ndimage.distance_transform_edt(~occ) * a.pitch
    field = ndimage.gaussian_filter((outside - inside).astype(np.float32), 1.2)
    nx, ny, nz = field.shape
    extra = int(np.ceil((a.plug_len + 2.0 + lo[2]) / a.pitch)) if lo[2] > -a.plug_len - 2 else 0
    if extra > 0:
        field = np.concatenate([np.full((nx, ny, extra), 5.0, np.float32), field], axis=2)
        lo = lo - np.array([0, 0, extra * a.pitch])
        nz += extra
    xs = lo[0] + np.arange(nx) * a.pitch
    ys = lo[1] + np.arange(ny) * a.pitch
    zs = lo[2] + np.arange(nz) * a.pitch
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    for k in range(nz):
        P = np.column_stack([X, Y, np.full(len(X), zs[k])])
        d = field[:, :, k].ravel()
        d = np.where(P[:, 2] < 0, np.maximum(d, -P[:, 2]), d)
        d = F.smin(d, F.plug_sdf(P, cfg), 1.2)
        if a.wick > 0:
            d = F.smax(d, -F.hole_along(P, tip - t * 0.6, t, a.wick / 2, 10.0), 0.25)
        field[:, :, k] = d.reshape(nx, ny)
    mesh = mc_mesh(field, lo, a.pitch)
    if len(mesh.faces) > a.faces:
        import fast_simplification
        v2, f2 = fast_simplification.simplify(mesh.vertices, mesh.faces, 1 - a.faces / len(mesh.faces))
        mesh = trimesh.Trimesh(v2, f2, process=True)
    return mesh


def tip_axis(mesh, span=12.0):
    """Tip point and the direction of the last `span` mm of the finger."""
    V = mesh.vertices
    top = V[:, 2].max()
    zs = np.linspace(top - span, top - 1.0, 6)
    cs = []
    for z in zs:
        sel = np.abs(V[:, 2] - z) < 0.6
        cs.append(V[sel].mean(axis=0) if sel.any() else None)
    cs = np.array([c for c in cs if c is not None])
    t = cs[-1] - cs[0]
    t /= np.linalg.norm(t)
    tip = V[np.argmax(V @ t)]
    return tip, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--cut", type=float, default=0.0, help="mm to remove from the bottom")
    ap.add_argument("--cup", type=float, default=22.0)
    ap.add_argument("--rib", type=float, default=0.3)
    ap.add_argument("--plug-len", type=float, default=18.0)
    ap.add_argument("--wick", type=float, default=2.2, help="wick hole diameter, 0 for none")
    ap.add_argument("--close", type=float, default=1.0, help="gaps narrower than about twice this are closed")
    ap.add_argument("--pitch", type=float, default=0.2)
    ap.add_argument("--faces", type=int, default=300000)
    ap.add_argument("--credit", default="")
    a = ap.parse_args()

    m = trimesh.load(a.src, force="mesh")
    # centre the base on the cup axis, base at z = 0, optional cut
    z0 = m.bounds[0][2] + a.cut
    if a.cut > 0:
        m = m.slice_plane([0, 0, z0], [0, 0, 1], cap=False)
    base = m.vertices[m.vertices[:, 2] < z0 + 3.0]
    cx, cy = (base[:, 0].min() + base[:, 0].max()) / 2, (base[:, 1].min() + base[:, 1].max()) / 2
    m.apply_translation([-cx, -cy, -z0])
    tip, t = tip_axis(m)

    cfg = dict(F.BASE, cup=a.cup, rib=a.rib, plug_len=a.plug_len)
    bodies = [p for p in m.split(only_watertight=False) if p.is_watertight and abs(p.volume) > 500]
    if bodies:
        # exact route: keep the model's own surface, fill only what is hidden inside
        shell = to_manifold(bodies[0])
        for p in bodies[1:]:
            shell = shell + to_manifold(p)
        occ, lo = solid_occupancy(m, a.pitch, a.close)
        inside = ndimage.distance_transform_edt(occ) * a.pitch
        core = ndimage.gaussian_filter((0.45 - inside).astype(np.float32), 1.0)   # 0.45 mm inside the skin
        core_m = to_manifold(mc_mesh(core, lo, a.pitch))
        solid = shell + core_m
        R = a.cup / 2 + 1.0
        # the plug reaches 1 mm up into the finger so the two join
        pcfg = dict(cfg, plug_len=cfg["plug_len"] + 1.0)
        plug = sdf_mesh(lambda P: F.plug_sdf(P - np.array([0, 0, 1.0]), pcfg),
                        [-R, -R, -a.plug_len - 1], [R, R, 2.2], 0.15)
        solid = solid + to_manifold(plug)
        if a.wick > 0:
            import manifold3d as mf
            cyl = trimesh.creation.cylinder(radius=a.wick / 2, height=12.0, sections=48)
            T = trimesh.geometry.align_vectors([0, 0, 1], t)
            cyl.apply_transform(T)
            cyl.apply_translation(tip - t * 4.0)
            solid = solid - to_manifold(cyl)
        mesh = to_trimesh(solid)
    else:
        mesh = voxel_route(m, a, tip, t, cfg)
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda p: len(p.faces))
    trimesh.repair.fix_normals(mesh)
    mesh.export(a.out)

    info = F.check(mesh, cfg)
    # ring check along the finger's own centre line: tip, section centres, rim
    V = mesh.vertices
    path = [tip]
    for z in np.linspace(tip[2] - 12, 3.0, 8):
        sel = np.abs(V[:, 2] - z) < 0.5
        path.append(V[sel].mean(axis=0))
    path.append(np.array([0.0, 0.0, 0.0]))
    rows, prof = F.ring_map(mesh, None, path=path)
    info.update(name=os.path.splitext(os.path.basename(a.out))[0], file=os.path.basename(a.out),
                config=dict(cfg, kind="candle", source=os.path.basename(a.src), cut=a.cut, credit=a.credit),
                rings=rows)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("rings", "config")}))
    print("rings:", ", ".join(f"US{int(r['us'])}:{r['z_mm']}" for r in rows if r["z_mm"]))


if __name__ == "__main__":
    main()
