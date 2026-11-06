"""Random Mitsuba renders -> training samples. One shard per view (resumable), merged at the end.
python -m pipeline.gen_data --views 200 --res 128 --spp 1024 --out pipeline/out/train.npz"""
import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.config import CAMERA, FLAME
from pipeline.features import features
from pipeline.scene import camera_origin, render


def random_view(i: int):
    rng = np.random.default_rng([1234, i])  # per-view seed: same views whatever the resume point
    view = dict(az_deg=rng.uniform(0, 360), el_deg=rng.uniform(*CAMERA["elevation_deg"]),
                radius=rng.uniform(*CAMERA["radius"]))
    a, b = FLAME["flicker_xz"], FLAME["flicker_y"]
    return view, rng.uniform([-a, -b, -a], [a, b, a])


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


def render_view(i, res, spp) -> dict:
    view, off = random_view(i)
    r = render(view, off, res, spp)
    m = r["mask"]
    pos, nrm, cam = r["pos"][m], r["nrm"][m], camera_origin(view)
    n = len(pos)
    return dict(X=features(pos, nrm, cam, off), Y=(r["full"] - r["spec"])[m].astype(np.float32),
                view_id=np.full(n, i, np.int32), pos=pos, nrm=nrm,
                cam=np.tile(cam, (n, 1)).astype(np.float32), offset=np.tile(off, (n, 1)).astype(np.float32))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=int, default=200)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--out", default="pipeline/out/train.npz")
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
    parts = [np.load(shards / f"view_{i:04d}.npz") for i in range(a.views)]
    np.savez(out, **{k: np.concatenate([q[k] for q in parts]) for k in parts[0].files})
    print("samples", sum(len(q["X"]) for q in parts), "->", out)


if __name__ == "__main__":
    main()
