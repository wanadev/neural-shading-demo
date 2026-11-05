"""Candle + wick meshes shared by Mitsuba and three.js. `python -m pipeline.assets` writes web/data/."""
import json
from pathlib import Path

import numpy as np

from pipeline.config import CANDLE, FLAME, to_json

WEB_DATA = Path(__file__).resolve().parent.parent / "web" / "data"


def _arc(cx, cy, rad, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rad * np.cos(a), cy + rad * np.sin(a)], 1)


def _revolve(profile, segments, wobble_fn=None):
    """Revolve a (r, y) profile running from a bottom pole (r=0) to a top pole (r=0) around +Y.
    Returns a closed mesh with outward (CCW) winding and area-weighted vertex normals."""
    rings = profile[1:-1]
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    r = rings[:, 0:1] * np.ones((1, segments))
    y = rings[:, 1:2] * np.ones((1, segments))
    if wobble_fn is not None:
        r = r + wobble_fn(th[None, :], y) * (r / CANDLE["radius"])
    ring_v = np.stack([r * np.cos(th), y, r * np.sin(th)], -1).reshape(-1, 3)
    V = np.vstack([[0, profile[0, 1], 0], ring_v, [0, profile[-1, 1], 0]]).astype(np.float64)
    nr, top = len(rings), len(V) - 1
    idx = lambda t, i: 1 + t * segments + (i % segments)
    F = []
    for i in range(segments):  # winding: d(profile) x d(theta) points outward
        F.append([0, idx(0, i + 1), idx(0, i)])
        F.append([top, idx(nr - 1, i), idx(nr - 1, i + 1)])
        for t in range(nr - 1):
            a, b, c, d = idx(t, i), idx(t + 1, i), idx(t + 1, i + 1), idx(t, i + 1)
            F += [[a, b, c], [a, c, d]]
    F = np.array(F, dtype=np.int32)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area-weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return V.astype(np.float32), F, N.astype(np.float32)


def _profile(cfg):
    R, H, cup, rr = cfg["radius"], cfg["height"], cfg["cup_depth"], cfg["rim_radius"]
    bev, n = 0.001, cfg["rings"]
    bottom = np.stack([np.linspace(0, R - bev, 8), np.zeros(8)], 1)
    bevel = _arc(R - bev, bev, bev, -np.pi / 2, 0, 6)[1:]
    side = np.stack([np.full(n // 2, R), np.linspace(bev, H - rr, n // 2)], 1)[1:]
    rim = _arc(R - rr, H - rr, rr, 0, np.pi / 2, 10)[1:]
    rc = R - rr
    u = np.linspace(1, 0, n // 3)[1:]
    cupc = np.stack([u * rc, H - cup * (1 - u**2) ** 2], 1)
    return np.vstack([bottom, bevel, side, rim, cupc])


def candle_mesh(cfg=CANDLE):
    rng = np.random.default_rng(cfg["seed"])
    p = rng.uniform(0, 2 * np.pi, 3)
    w = cfg["wobble"]
    # weights sum to 1 -> |offset| <= wobble
    wob = lambda th, y: w * (0.5 * np.sin(3 * th + p[0] + 20 * y)
                             + 0.3 * np.sin(7 * th + p[1]) + 0.2 * np.sin(60 * y + p[2]))
    return _revolve(_profile(cfg), cfg["segments"], wob)


def wick_mesh(cfg=CANDLE, flame=FLAME):
    rw, y0, y1 = 0.0008, cfg["height"] - cfg["cup_depth"] + 0.0002, flame["base"][1] - 0.004
    prof = np.array([[0, y0], [rw, y0], [rw, y0 + 1e-4], [rw, y1 - 1e-4], [rw, y1], [0, y1]])
    return _revolve(prof, 16)


def write_obj(path, V, F, N):
    with open(path, "w") as f:
        f.writelines(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in V)
        f.writelines(f"vn {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in N)
        f.writelines("f {0}//{0} {1}//{1} {2}//{2}\n".format(*(t + 1)) for t in F)


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    write_obj(WEB_DATA / "candle.obj", *candle_mesh())
    write_obj(WEB_DATA / "wick.obj", *wick_mesh())
    (WEB_DATA / "config.json").write_text(json.dumps(to_json(), indent=1))


if __name__ == "__main__":
    main()
