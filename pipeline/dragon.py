"""Stanford dragon -> closed, decimated, scaled mesh shared by Mitsuba and three.js.
python -m pipeline.dragon   (downloads into pipeline/out/dragon/, writes web/data/dragon.ply + config.json)"""
import json
import tarfile
import urllib.request

import pymeshlab
import numpy as np
import trimesh

from pipeline.config import DRAGON, OUT, WEB_DATA, to_json


def load_dragon(ply_path) -> trimesh.Trimesh:
    return trimesh.load(ply_path, process=False)


def _repair(m: trimesh.Trimesh) -> trimesh.Trimesh:
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)  # merges duplicate vertices
    m = max(m.split(only_watertight=False), key=lambda p: len(p.faces))  # drop scan debris
    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)  # consistent outward winding
    return m


def prepare(mesh: trimesh.Trimesh, target_faces: int, height: float):
    m = _repair(mesh)
    if len(m.faces) > target_faces:
        # topology-preserving quadric collapse: plain QEM leaves non-manifold edges on the dragon
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservetopology=True,
                                                    preservenormal=True, qualitythr=0.5, optimalplacement=True)
        o = ms.current_mesh()
        m = _repair(trimesh.Trimesh(o.vertex_matrix(), o.face_matrix(), process=False))
    if not m.is_watertight:
        raise ValueError("mesh is not watertight after repair")
    V = m.vertices.copy()
    lo, hi = V.min(0), V.max(0)
    V = (V - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * (height / (hi[1] - lo[1]))
    m = trimesh.Trimesh(V, m.faces, process=False)
    return V.astype(np.float32), m.faces.astype(np.int32), m.vertex_normals.astype(np.float32)


def write_ply(path, V, F, N):
    vert = np.zeros(len(V), dtype=[(c, "<f4") for c in ("x", "y", "z", "nx", "ny", "nz")])
    for k, c in enumerate("xyz"):
        vert[c], vert["n" + c] = V[:, k], N[:, k]
    face = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"], face["i"] = 3, F
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(V)}\n" + "".join(f"property float {c}\n" for c in ("x", "y", "z", "nx", "ny", "nz"))
              + f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vert.tobytes())
        f.write(face.tobytes())


def main():
    d = OUT / "dragon"
    d.mkdir(parents=True, exist_ok=True)
    tgz = d / "dragon_recon.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(DRAGON["url"], tgz)
    src = d / DRAGON["member"]
    if not src.exists():
        with tarfile.open(tgz) as t:
            t.extract(DRAGON["member"], d, filter="data")
    V, F, N = prepare(load_dragon(src), DRAGON["target_faces"], DRAGON["height"])
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    write_ply(WEB_DATA / "dragon.ply", V, F, N)
    (WEB_DATA / "config.json").write_text(json.dumps(to_json(), indent=1))
    print(f"dragon: {len(V)} vertices, {len(F)} faces")


if __name__ == "__main__":
    main()
