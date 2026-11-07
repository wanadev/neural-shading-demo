"""SPIKE (throwaway): can a tiny MLP do screen-space ambient occlusion better than classic SSAO?

Ground truth = ray-traced AO (cosine hemisphere, radius R) at every pixel of random views of the
dragon on its ground plane. Both screen-space methods only see what an SSAO pass sees: the
G-buffer (positions + normals) at 16 screen offsets around the pixel.
  classic : Crytek/SAO-style formula, its 2 parameters fitted on the same data (fair baseline)
  neural  : MLP 35 -> 32 -> 32 -> 1
python -m spikes.ao.ao_spike   (from the repo root)"""
import time
from pathlib import Path

import numpy as np
import mitsuba as mi
import torch

from pipeline.scene import _sensor, camera_origin  # sets the llvm variant
from pipeline.config import CAMERA, GROUND, WEB_DATA
from pipeline.tonemap import to_srgb8

OUT = Path(__file__).parent / "out"
R = 0.04          # AO radius, metres
K = 16            # screen-space samples
GT_RAYS = 128
RES, N_TRAIN, N_TEST = 192, 120, 6

# fixed sample pattern: Vogel disk, radius in [0.1, 1]
_i = np.arange(K)
PATTERN = np.stack([np.cos(_i * 2.39996), np.sin(_i * 2.39996)], 1) * np.sqrt((_i + 0.5) / K)[:, None]


def scene_for(view, res):
    T = mi.ScalarTransform4f
    half = GROUND["size"] / 2
    return mi.load_dict({"type": "scene", "sensor": _sensor(view, res),
                         "dragon": dict(type="ply", filename=str(WEB_DATA / "dragon.ply"), face_normals=False),
                         "ground": dict(type="rectangle", to_world=T().rotate([1, 0, 0], -90).scale([half, half, 1]))})


def gbuffer(sc):
    aov = mi.load_dict(dict(type="aov", aovs="pos:position,nn:sh_normal,dd:depth"))
    a = np.array(mi.render(sc, integrator=aov, spp=1), dtype=np.float32)
    return a[..., 0:3], a[..., 3:6], a[..., 6] > 0


def gt_ao(sc, pos, nrm, hit, rng):
    p, n = pos[hit].astype(np.float64), nrm[hit].astype(np.float64)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    a = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t = np.cross(n, a); t /= np.linalg.norm(t, axis=1, keepdims=True); b = np.cross(n, t)
    occ = np.zeros(len(p))
    for _ in range(GT_RAYS // 32):
        u1, u2 = rng.random((2, len(p), 32))
        r, ph = np.sqrt(u1), 2 * np.pi * u2
        d = (r * np.cos(ph))[..., None] * t[:, None] + (r * np.sin(ph))[..., None] * b[:, None] + np.sqrt(1 - u1)[..., None] * n[:, None]
        o = np.repeat(p[:, None] + 1e-4 * n[:, None], 32, 1).reshape(-1, 3)
        ray = mi.Ray3f(mi.Point3f(*o.T.astype(np.float32)), mi.Vector3f(*d.reshape(-1, 3).T.astype(np.float32)))
        ray.maxt = R
        occ += np.array(sc.ray_test(ray), dtype=bool).reshape(len(p), 32).sum(1)
    ao = np.zeros(hit.shape, np.float32)
    ao[hit] = 1 - occ / GT_RAYS
    return ao


def ss_features(pos, nrm, hit, eye, fwd, res):
    """What an SSAO pass sees: per pixel, (n.d/R, |d|/R) at K screen offsets scaled by projected R."""
    H, W = hit.shape
    f = (res / 2) / np.tan(np.radians(CAMERA["fov_y"]) / 2)
    ys, xs = np.nonzero(hit)
    p = pos[ys, xs]; n = nrm[ys, xs] / np.linalg.norm(nrm[ys, xs], axis=1, keepdims=True)
    z = (p - eye) @ fwd
    rad_px = R * f / np.maximum(z, 1e-3)
    feats = []
    for ox, oy in PATTERN:
        sx = np.clip(np.round(xs + ox * rad_px), 0, W - 1).astype(int)
        sy = np.clip(np.round(ys + oy * rad_px), 0, H - 1).astype(int)
        valid = hit[sy, sx]
        d = pos[sy, sx] - p
        dist = np.linalg.norm(d, axis=1)
        nd = np.where(valid, (n * d).sum(1) / R, 0.0)
        feats += [np.clip(nd, -4, 4), np.where(valid, np.clip(dist / R, 0, 4), 4.0)]
    return np.concatenate([np.stack(feats, 1), n], 1).astype(np.float32), ys, xs


def make_view(i, rng):
    v = dict(az_deg=rng.uniform(0, 360), el_deg=rng.uniform(*CAMERA["elevation_deg"]), radius=rng.uniform(*CAMERA["radius"]))
    eye = camera_origin(v); fwd = np.array(CAMERA["target"]) - eye; fwd /= np.linalg.norm(fwd)
    return v, eye, fwd


def dataset(n_views, seed, res):
    rng = np.random.default_rng(seed)
    X, Y, imgs = [], [], []
    for i in range(n_views):
        v, eye, fwd = make_view(i, rng)
        sc = scene_for(v, res)
        pos, nrm, hit = gbuffer(sc)
        ao = gt_ao(sc, pos, nrm, hit, rng)
        x, ys, xs = ss_features(pos, nrm, hit, eye, fwd, res)
        X.append(x); Y.append(ao[ys, xs])
        imgs.append((hit, ys, xs, ao))
    return np.concatenate(X), np.concatenate(Y), imgs


class Classic(torch.nn.Module):
    """AO = 1 - s * mean_k max(0, n.d/|d| - bias) * max(0, 1 - (|d|/R)^2)"""
    def __init__(self):
        super().__init__()
        self.s, self.bias = torch.nn.Parameter(torch.tensor(1.0)), torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        nd, dist = x[:, 0:2 * K:2] * R, x[:, 1:2 * K:2] * R
        cosang = nd / torch.clamp(dist, min=1e-5)
        fall = torch.clamp(1 - (dist / R) ** 2, min=0)
        return torch.clamp(1 - self.s * (torch.clamp(cosang - self.bias, min=0) * fall).mean(1), 0, 1)


def fit(model, X, Y, steps, lr, batch=8192):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        b = torch.randint(0, len(X), (batch,), generator=g)
        loss = ((model(Xt[b]).squeeze(-1) - Yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    Xtr, Ytr, _ = dataset(N_TRAIN, 1, RES)
    Xte, Yte, test_imgs = dataset(N_TEST, 2, 384)  # features scale with resolution
    print(f"data {time.time() - t:.0f}s: train {len(Xtr)} px, test {len(Xte)} px", flush=True)
    torch.manual_seed(0)
    classic = fit(Classic(), Xtr, Ytr, 1500, 0.02)
    mlp = torch.nn.Sequential(torch.nn.Linear(2 * K + 3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1), torch.nn.Sigmoid())
    mlp = fit(mlp, Xtr, Ytr, 6000, 3e-3)
    with torch.no_grad():
        pc, pn = classic(torch.from_numpy(Xte)).numpy(), mlp(torch.from_numpy(Xte)).squeeze(-1).numpy()
    mae = lambda p: float(np.abs(p - Yte).mean())
    occluded = Yte < 0.9
    print(f"MAE all px   classic {mae(pc):.4f}  neural {mae(pn):.4f}")
    print(f"MAE occluded classic {np.abs(pc - Yte)[occluded].mean():.4f}  neural {np.abs(pn - Yte)[occluded].mean():.4f}  ({occluded.mean():.1%} of px)")
    print(f"classic params s={classic.s.item():.3f} bias={classic.bias.item():.3f}")
    # comparison sheet: rows = test views, columns = ground truth | classic | neural
    rows, off = [], 0
    for hit, ys, xs, ao in test_imgs[:3]:
        n = len(ys)
        tiles = []
        for vals in (ao[ys, xs], pc[off:off + n], pn[off:off + n]):
            img = np.ones(hit.shape, np.float32)
            img[ys, xs] = vals
            tiles.append(np.repeat((np.clip(img, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2))  # AO is a factor, not a colour
        rows.append(np.concatenate(tiles, 1))
        off += n
    mi.Bitmap(np.concatenate(rows, 0)).write(str(OUT / "ao_compare.png"))
    print("wrote", OUT / "ao_compare.png")


if __name__ == "__main__":
    main()
