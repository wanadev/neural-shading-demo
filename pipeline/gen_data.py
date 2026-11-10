"""Random Mitsuba renders (camera + lamp) -> training samples. One shard per view (resumable), merged at the end.
python -m pipeline.gen_data --views 400 --res 192 --spp 512 --out pipeline/out/dragon_train.npz"""
import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from pipeline.config import CAMERA, LAMP, WEB_DATA
from pipeline.features import barycentric
from pipeline.scene import camera_origin, lamp_position, render


def save_atomic(path, **arrays):
    """Write to a temp file then rename: a job killed mid-write never leaves a torn shard."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    try:
        np.savez(tmp, **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def shard_dir(out: Path, res: int, spp: int) -> Path:
    return out.parent / f"{out.stem}_r{res}_s{spp}_shards"  # new settings never reuse old shards


@lru_cache(maxsize=1)
def dragon_mesh():
    m = trimesh.load(WEB_DATA / "dragon.ply", process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def random_view(i: int):
    rng = np.random.default_rng([1234, i])  # per-view seed: same views whatever the resume point
    view = dict(az_deg=rng.uniform(0, 360), el_deg=rng.uniform(*CAMERA["elevation_deg"]),
                radius=rng.uniform(*CAMERA["radius"]))
    return view, (rng.uniform(0, 360), rng.uniform(*LAMP["elevation_deg"]))


def render_view(i, res, spp) -> dict:
    view, lamp = random_view(i)
    r = render(view, lamp, res, spp, passes=("env", "lamp"))
    m = r["mask"]
    V, F = dragon_mesh()
    pos, tri = r["pos"][m], r["tri"][m]
    n = len(pos)
    return dict(tri=tri.astype(np.int32), bary=barycentric(pos, V, F, tri), nrm=r["nrm"][m], pos=pos,
                cam=np.tile(camera_origin(view), (n, 1)).astype(np.float32),
                lamp=np.tile(lamp_position(*lamp), (n, 1)).astype(np.float32),
                Y_env=r["env"][m].astype(np.float32), Y_lamp=r["lamp"][m].astype(np.float32),
                view_id=np.full(n, i, np.int32))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, default=400)
    p.add_argument("--res", type=int, default=192)
    p.add_argument("--spp", type=int, default=512)
    p.add_argument("--out", default="pipeline/out/dragon_train.npz")
    a = p.parse_args(argv)
    out = Path(a.out)
    shards = shard_dir(out, a.res, a.spp)
    shards.mkdir(parents=True, exist_ok=True)
    for i in range(a.views):
        f = shards / f"view_{i:04d}.npz"
        if f.exists():
            continue
        t = time.time()
        save_atomic(f, **render_view(i, a.res, a.spp))
        print(f"view {i + 1}/{a.views} {time.time() - t:.1f}s", flush=True)
    parts = [dict(np.load(shards / f"view_{i:04d}.npz")) for i in range(a.views)]  # dict(): loads and closes
    save_atomic(out, **{k: np.concatenate([q[k] for q in parts]) for k in parts[0]})
    print("samples", sum(len(q["tri"]) for q in parts), "->", out)


if __name__ == "__main__":
    main()
