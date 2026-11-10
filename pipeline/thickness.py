"""Per-vertex local thickness for the classic baseline: mean distance to the far surface along
cosine-distributed directions around the inward normal (Mitsuba ray casting).
python -m pipeline.thickness   -> web/data/thickness.bin (uint8) + thickness_max in config.json"""
import json
import tempfile
from pathlib import Path

import numpy as np
import mitsuba as mi
import trimesh

if mi.variant() is None:
    mi.set_variant("llvm_ad_rgb")

from pipeline.config import WEB_DATA
from pipeline.dragon import write_ply


def tangent_frame(n):
    a = np.where(np.abs(n[:, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t = np.cross(n, a); t /= np.linalg.norm(t, axis=1, keepdims=True)
    return t, np.cross(n, t)


def vertex_thickness(V, F, N, rays=32, seed=0) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.ply"
        write_ply(p, V, F, N)
        scene = mi.load_dict({"type": "scene", "m": dict(type="ply", filename=str(p), face_normals=True)})
    n = -np.asarray(N, np.float64)  # inward
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t, b = tangent_frame(n)
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random((2, len(V), rays))
    r, phi = np.sqrt(u1), 2 * np.pi * u2  # cosine-weighted hemisphere
    z = np.sqrt(1 - u1)
    d = (r * np.cos(phi))[..., None] * t[:, None] + (r * np.sin(phi))[..., None] * b[:, None] + z[..., None] * n[:, None]
    o = np.repeat(np.asarray(V, np.float64)[:, None] + 1e-5 * n[:, None], rays, 1)
    ray = mi.Ray3f(mi.Point3f(*o.reshape(-1, 3).T.astype(np.float32)), mi.Vector3f(*d.reshape(-1, 3).T.astype(np.float32)))
    dist = np.array(scene.ray_intersect(ray).t, dtype=np.float64).reshape(len(V), rays)
    hit = np.isfinite(dist) & (dist < 1e3)
    return np.where(hit.any(1), np.where(hit, dist, 0).sum(1) / np.maximum(hit.sum(1), 1), 0).astype(np.float32)


def main():
    m = trimesh.load(WEB_DATA / "dragon.ply", process=False)
    t = vertex_thickness(m.vertices, m.faces, m.vertex_normals)
    tmax = float(t.max())
    (WEB_DATA / "thickness.bin").write_bytes(np.round(t / tmax * 255).astype(np.uint8).tobytes())
    cfg = json.loads((WEB_DATA / "config.json").read_text())
    cfg["thickness_max"] = tmax
    (WEB_DATA / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"thickness: median {np.median(t) * 1000:.1f} mm, max {tmax * 1000:.1f} mm")


if __name__ == "__main__":
    main()
