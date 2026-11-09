"""SPIKE (throwaway): neural refraction through a thick glass dragon.

Real-time engines refract once, at the entry surface, and look the environment up in that
direction. Real glass refracts again on the way out, often after internal reflections, which is
what makes thick glass look like glass. Here a network learns, per surface point and view
direction, where the light finally leaves the object and how much of it gets through.

  classic : one refraction at entry + Fresnel reflection (what three.js-style env refraction does)
  neural  : per-vertex latents (8) + MLP -> exit direction + transmittance, then one env lookup
  traced  : the exact dominant path (upper bound of the neural approach)
  mitsuba : full path-traced reference (all paths, internal reflections split)
python -m spikes.glass.glass_spike   (from the repo root)"""
import time
from pathlib import Path

import numpy as np
import mitsuba as mi
import torch

from pipeline.scene import ENV_ROT_Y_DEG, _sensor  # sets the llvm variant
from pipeline.config import CAMERA, ENV, WEB_DATA
from pipeline.env import read_hdr
from pipeline.features import barycentric
from pipeline.gen_data import dragon_mesh
from pipeline.tonemap import aces_filmic, to_srgb8

OUT = Path(__file__).parent / "out"
IOR = 1.5
RES_TRAIN, N_TRAIN, RES_TEST, N_TEST = 160, 150, 512, 4
LABO = WEB_DATA / "labo" / "glass"  # images + metrics for web/labo.html
EXPOSURE = 1.0
ENV_FILE = WEB_DATA / "env" / f"{ENV['name']}_1k.hdr"


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fresnel(cos_i, eta):
    """Exact dielectric Fresnel, eta = n_incident / n_transmitted. Returns (F, cos_t, tir)."""
    sin2_t = eta ** 2 * np.maximum(0, 1 - cos_i ** 2)
    tir = sin2_t >= 1
    cos_t = np.sqrt(np.maximum(0, 1 - sin2_t))
    rs = ((eta * cos_i - cos_t) / (eta * cos_i + cos_t + 1e-12)) ** 2
    rp = ((eta * cos_t - cos_i) / (eta * cos_t + cos_i + 1e-12)) ** 2
    return np.where(tir, 1.0, (rs + rp) / 2), cos_t, tir


def refract(d, n, eta, cos_i, cos_t):
    """d incoming (unit), n facing against d."""
    return unit(eta * d + (eta * cos_i - cos_t)[:, None] * n)


def reflect(d, n):
    return d - 2 * (d * n).sum(1, keepdims=True) * n


def mesh_scene():
    return mi.load_dict({"type": "scene", "m": dict(type="ply", filename=str(WEB_DATA / "dragon.ply"), face_normals=False)})


def trace_exit(sc, p, n, d, max_bounces=8):
    """Follow the dominant path through the glass. Returns exit direction, transmittance, valid."""
    cos_i = -(d * n).sum(1)
    F0, cos_t, _ = fresnel(cos_i, 1 / IOR)
    t = refract(d, n, 1 / IOR, cos_i, cos_t)
    T = 1 - F0
    pos = p - 1e-4 * n
    out_dir, done = np.zeros_like(d), np.zeros(len(p), bool)
    for _ in range(max_bounces):
        act = ~done
        if not act.any():
            break
        ray = mi.Ray3f(mi.Point3f(*pos[act].T.astype(np.float32)), mi.Vector3f(*t[act].T.astype(np.float32)))
        si = sc.ray_intersect(ray)
        ok = np.array(si.is_valid(), bool)
        q = np.stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)], 1)
        m = np.stack([np.array(si.sh_frame.n.x), np.array(si.sh_frame.n.y), np.array(si.sh_frame.n.z)], 1)
        idx = np.flatnonzero(act)
        lost = idx[~ok]
        done[lost] = True
        T[lost] = 0
        idx, q, m, tt = idx[ok], q[ok], m[ok], t[idx[ok]]
        m = np.where(((tt * m).sum(1) < 0)[:, None], -m, m)  # outward, along the inside ray
        c = (tt * m).sum(1)
        F, cos_t2, tir = fresnel(c, IOR)
        exit_ = ~tir
        e = idx[exit_]
        out_dir[e] = refract(tt[exit_], -m[exit_], IOR, c[exit_], cos_t2[exit_])
        T[e] *= 1 - F[exit_]
        done[e] = True
        r = idx[tir]  # total internal reflection: lossless, keep going
        t[r] = reflect(tt[tir], m[tir])
        pos[r] = q[tir] - 1e-4 * m[tir]
    T[~done] = 0
    return out_dir, T, done


class EnvMap:
    def __init__(self):
        self.img = read_hdr(ENV_FILE)[..., :3] * ENV["scale"] * 4  # brighter room: glass needs light to read
        self.h, self.w = self.img.shape[:2]

    def __call__(self, d):  # three.js equirectUv convention, file row 0 = top
        u = np.arctan2(d[:, 2], d[:, 0]) / (2 * np.pi) + 0.5
        v = np.arcsin(np.clip(d[:, 1], -1, 1)) / np.pi + 0.5
        x, y = u * self.w - 0.5, (1 - v) * self.h - 0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        g = lambda yy, xx: self.img[np.clip(yy, 0, self.h - 1), xx % self.w]
        return (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy)
                + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy)


def view_buffers(view, res):
    """G-buffer of the dragon + the exact camera ray of every pixel."""
    sc = mi.load_dict({"type": "scene", "sensor": _sensor(view, res),
                       "m": dict(type="ply", filename=str(WEB_DATA / "dragon.ply"), face_normals=False)})
    aov = mi.load_dict(dict(type="aov", aovs="pos:position,nn:sh_normal,pi:prim_index,dd:depth"))
    a = np.array(mi.render(sc, integrator=aov, spp=1), dtype=np.float32)
    hit = a[..., 7] > 0
    sensor = sc.sensors()[0]
    ys, xs = np.mgrid[0:res, 0:res]
    pp = mi.Point2f(((xs.ravel() + 0.5) / res).astype(np.float32), ((ys.ravel() + 0.5) / res).astype(np.float32))
    ray, _ = sensor.sample_ray(0.0, 0.5, pp, mi.Point2f(0.5, 0.5))
    dirs = np.stack([np.array(ray.d.x), np.array(ray.d.y), np.array(ray.d.z)], 1).reshape(res, res, 3)
    return a[..., 0:3], unit(a[..., 3:6]), np.round(a[..., 6]).astype(np.int64), hit, dirs


def random_view(rng):
    return dict(az_deg=rng.uniform(0, 360), el_deg=rng.uniform(*CAMERA["elevation_deg"]), radius=rng.uniform(*CAMERA["radius"]))


def samples(view, res, msc, V, F):
    pos, nrm, tri, hit, dirs = view_buffers(view, res)
    p, n, d, tr = pos[hit].astype(np.float64), nrm[hit].astype(np.float64), dirs[hit].astype(np.float64), tri[hit]
    n = np.where(((d * n).sum(1) > 0)[:, None], -n, n)
    e, T, ok = trace_exit(msc, p, n, d)
    return dict(p=p, n=n, d=d, tri=tr, bary=barycentric(p, V, F, tr), e=e, T=T, ok=ok, hit=hit, dirs=dirs)


class NeuralGlass(torch.nn.Module):
    def __init__(self, n_vertices, width=64):
        super().__init__()
        self.latents = torch.nn.Parameter(torch.randn(n_vertices, 8) * 0.01)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(14, width), torch.nn.ReLU(), torch.nn.Linear(width, width),
                                       torch.nn.ReLU(), torch.nn.Linear(width, 4))

    def forward(self, tv, bary, n, d):
        lat = (bary[..., None] * self.latents[tv]).sum(1)
        o = self.mlp(torch.cat([lat, n, d], 1))
        return torch.nn.functional.normalize(o[:, :3], dim=1), torch.sigmoid(o[:, 3])


def mitsuba_reference(view, res, spp=256):
    T = mi.ScalarTransform4f
    sc = mi.load_dict({"type": "scene", "sensor": _sensor(view, res),
                       "m": dict(type="ply", filename=str(WEB_DATA / "dragon.ply"), face_normals=False,
                                 bsdf=dict(type="dielectric", int_ior=IOR, ext_ior=1.0)),
                       "env": dict(type="envmap", filename=str(ENV_FILE), scale=ENV["scale"] * 4,
                                   to_world=T().rotate([0, 1, 0], ENV_ROT_Y_DEG))})
    return np.array(mi.render(sc, integrator=mi.load_dict(dict(type="path", max_depth=64)), spp=spp), dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V, F = dragon_mesh()
    msc, env = mesh_scene(), EnvMap()
    rng = np.random.default_rng(3)
    t0 = time.time()
    train = [samples(random_view(rng), RES_TRAIN, msc, V, F) for _ in range(N_TRAIN)]
    cat = lambda k: np.concatenate([s[k] for s in train])
    ok = cat("ok")
    print(f"traced {N_TRAIN} views in {time.time() - t0:.0f}s, {len(ok)} px, {ok.mean():.1%} exit within 8 bounces", flush=True)
    tv = torch.from_numpy(F[cat("tri")]); bary = torch.from_numpy(cat("bary").astype(np.float32))
    n_, d_ = torch.from_numpy(cat("n").astype(np.float32)), torch.from_numpy(cat("d").astype(np.float32))
    e_, T_ = torch.from_numpy(cat("e").astype(np.float32)), torch.from_numpy(cat("T").astype(np.float32))
    okt = torch.from_numpy(ok)
    torch.manual_seed(0)
    model = NeuralGlass(len(V))
    opt = torch.optim.Adam([{"params": model.mlp.parameters(), "lr": 3e-3}, {"params": [model.latents], "lr": 1e-2}])
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(8000):
        b = torch.randint(0, len(ok), (8192,), generator=g)
        pe, pT = model(tv[b], bary[b], n_[b], d_[b])
        dir_loss = ((1 - (pe * e_[b]).sum(1)) * okt[b]).mean()
        loss = dir_loss + ((pT - T_[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"trained in {time.time() - t0:.0f}s, final loss {loss.item():.4f}", flush=True)

    rows, errs = [], {"classic": [], "neural": [], "traced": []}
    LABO.mkdir(parents=True, exist_ok=True)
    save = lambda img, key, i: mi.Bitmap(np.ascontiguousarray(to_srgb8(aces_filmic(img, EXPOSURE)))).write(str(LABO / f"view{i}-{key}.jpg"))
    rng = np.random.default_rng(11)
    for i in range(N_TEST):
        view = random_view(rng)
        s = samples(view, RES_TEST, msc, V, F)
        ref = mitsuba_reference(view, RES_TEST)
        hit, dirs = s["hit"], s["dirs"]
        n, d = s["n"], s["d"]
        cos_i = -(d * n).sum(1)
        F0, cos_t, _ = fresnel(cos_i, 1 / IOR)
        refl = F0[:, None] * env(reflect(d, n))
        with torch.no_grad():
            pe, pT = model(torch.from_numpy(F[s["tri"]]), torch.from_numpy(s["bary"].astype(np.float32)),
                           torch.from_numpy(n.astype(np.float32)), torch.from_numpy(d.astype(np.float32)))
        shade = {
            "classic": refl + (1 - F0)[:, None] * env(refract(d, n, 1 / IOR, cos_i, cos_t)),
            "neural": refl + pT.numpy()[:, None] * env(pe.numpy().astype(np.float64)),
            "traced": refl + s["T"][:, None] * env(s["e"]),
        }
        bg = env(dirs.reshape(-1, 3).astype(np.float64)).reshape(RES_TEST, RES_TEST, 3)
        tiles = []
        for name in ("classic", "neural", "traced"):
            img = bg.copy()
            img[hit] = shade[name]
            errs[name].append(np.abs(img[hit] - ref[hit]).mean() / ref[hit].mean())
            tiles.append(to_srgb8(aces_filmic(img, EXPOSURE)))
            save(img, name, i)
        tiles.append(to_srgb8(aces_filmic(ref, EXPOSURE)))
        save(ref, "ref", i)
        rows.append(np.concatenate(tiles, 1))
        print(f"test view {i}: " + "  ".join(f"{k} {v[-1]:.3f}" for k, v in errs.items()), flush=True)
    print("mean relative error vs Mitsuba on glass pixels: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in errs.items()))
    mi.Bitmap(np.concatenate(rows, 0)).write(str(OUT / "glass_compare.png"))
    print("wrote", OUT / "glass_compare.png", "(columns: classic | neural | traced | Mitsuba)")
    import json
    rng = np.random.default_rng(11)  # the test views above, for the real-time page
    (LABO.parent / "glass_views.json").write_text(json.dumps([{k: float(v) for k, v in random_view(rng).items()} for _ in range(N_TEST)]))
    (LABO.parent / "glass.json").write_text(json.dumps(dict(
        views=[dict(errors={k: round(float(v[i]), 4) for k, v in errs.items()}) for i in range(N_TEST)],
        mean={k: round(float(np.mean(v)), 4) for k, v in errs.items()}), indent=1))
    print("labo export ->", LABO)


if __name__ == "__main__":
    main()
