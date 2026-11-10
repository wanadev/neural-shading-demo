"""Auto-decoder: a learned 8-value latent per dragon vertex + a tiny MLP, trained together.
Output = 6 values: SSS due to the room (RGB) | SSS due to the lamp (RGB).
python -m pipeline.train  -> web/data/weights_{w}.json, latents_{w}.bin, selftest.json"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch import nn

from pipeline.config import LATENT_DIM, WEB_DATA, WIDTHS
from pipeline.features import raw_features

N_RAW = 11


class NeuralSSS(nn.Module):
    """[latent(8) interpolated by barycentrics | standardized raw(11)] -> [Linear -> relu^2] x depth -> Linear(6) -> exp"""

    def __init__(self, n_vertices, width, latent_dim=LATENT_DIM, depth=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_vertices, latent_dim) * 0.01)
        dims = [latent_dim + N_RAW] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(width, 6)
        self.register_buffer("log_cap", torch.full((6,), 80.0))  # ~float32 exp() limit until set_output_cap

    def set_output_cap(self, cap):
        """Bound exp(): without it a few unseen input combinations produced values 60x the
        brightest target and dominated the error (and would show as fireflies)."""
        self.log_cap.copy_(torch.log(torch.as_tensor(np.asarray(cap, np.float32))))

    def mlp_parameters(self):
        return [p for n, p in self.named_parameters() if n != "latents"]

    def forward(self, tri_verts, bary, raw_std, latents=None):
        L = self.latents if latents is None else latents
        x = torch.cat([(bary[..., None] * L[tri_verts]).sum(1), raw_std], 1)
        for l in self.hidden:
            x = torch.relu(l(x)) ** 2
        return torch.exp(torch.minimum(self.out(x), self.log_cap))


def split_views(view_id, frac=0.1, seed=0):
    views = np.unique(view_id)
    val = np.random.default_rng(seed).choice(views, max(1, int(len(views) * frac)), replace=False)
    is_val = np.isin(view_id, val)
    return ~is_val, is_val


def rel_err(pred, y):
    return float(np.abs(pred - y).mean() / np.abs(y).mean())


def clamp_fireflies(Y, pct=99.99, factor=1.5):
    """Clip isolated Monte Carlo spikes: per channel, at factor x the pct-th percentile."""
    return np.minimum(Y, factor * np.percentile(Y, pct, axis=0)).astype(np.float32)


@lru_cache(maxsize=1)
def _load_samples(npz_path: str):
    """Millions of samples: loaded once per run (train + finetune for each width reuse them)."""
    d = dict(np.load(npz_path))
    raw = raw_features(d["nrm"], d["pos"], d["cam"], d["lamp"])
    Y = clamp_fireflies(np.concatenate([d["Y_env"], d["Y_lamp"]], 1))
    return d, raw, Y


def _load(npz_path, F):
    d, raw, Y = _load_samples(str(npz_path))
    return d, raw, Y, np.asarray(F)[d["tri"]]


def _fit(model, params, TV, B, R, Y, idx, epochs, batch, lr, seed, extra_groups=()):
    opt = torch.optim.Adam([{"params": params, "lr": lr}, *extra_groups])
    steps = epochs * max(1, len(idx) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b = torch.from_numpy(idx[rng.integers(0, len(idx), batch)])  # with replacement: O(batch)
        pred = model(TV[b], B[b], R[b])
        # relative MSE (unbiased on noisy targets, unlike fitting a transformed target)
        loss = ((pred - Y[b]) ** 2 / (pred.detach() + 0.01) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def _tensors(raw, mean, std, Y, tv, d):
    return (torch.from_numpy(tv.astype(np.int64)), torch.from_numpy(d["bary"].astype(np.float32)),
            torch.from_numpy(((raw - mean) / std).astype(np.float32)), torch.from_numpy(Y))


def _val_err(model, TV, B, R, Y, va):
    with torch.no_grad():
        v = torch.from_numpy(np.flatnonzero(va))
        return rel_err(model(TV[v], B[v], R[v]).numpy(), Y[v].numpy())


def train(npz_path, F, n_vertices, width, epochs=30, batch=8192, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    d, raw, Y, tv = _load(npz_path, F)
    tr, va = split_views(d["view_id"])
    mean, std = raw[tr].mean(0), raw[tr].std(0) + 1e-6
    TV, B, R, Yt = _tensors(raw, mean, std, Y, tv, d)
    model = NeuralSSS(n_vertices, width)
    model.set_output_cap(1.2 * Y[tr].max(0))  # a little headroom over the brightest target
    with torch.no_grad():
        model.out.bias.copy_(torch.log(torch.from_numpy(np.maximum(Y[tr].mean(0), 1e-4))))
    _fit(model, model.mlp_parameters(), TV, B, R, Yt, np.flatnonzero(tr), epochs, batch, lr, seed,
         extra_groups=[{"params": [model.latents], "lr": lr * 3}])  # each latent sees few samples: faster lr
    return model, mean, std, _val_err(model, TV, B, R, Yt, va)


def quantize(latents):
    x = np.asarray(latents.detach() if torch.is_tensor(latents) else latents, np.float32)
    lo, hi = x.min(0), x.max(0)
    hi = np.where(hi - lo < 1e-8, lo + 1e-8, hi)
    q = np.round((x - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    return q, lo.astype(np.float32), hi.astype(np.float32)


def finetune_quantized(model, npz_path, F, mean, std, epochs=5, batch=8192, lr=1e-3, seed=1):
    """Freeze the latents at their 8-bit values and refit the MLP to them."""
    q, lo, hi = quantize(model.latents)
    with torch.no_grad():
        model.latents.copy_(torch.from_numpy(lo + (hi - lo) * q.astype(np.float32) / 255))
    model.latents.requires_grad_(False)
    d, raw, Y, tv = _load(npz_path, F)
    tr, va = split_views(d["view_id"])
    TV, B, R, Yt = _tensors(raw, mean, std, Y, tv, d)
    _fit(model, model.mlp_parameters(), TV, B, R, Yt, np.flatnonzero(tr), epochs, batch, lr, seed)
    return q, lo, hi, _val_err(model, TV, B, R, Yt, va)


def export(model, in_mean, in_std, lo, hi, json_path, bin_path):
    q = np.round((model.latents.detach().numpy() - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    layers = list(model.hidden) + [model.out]
    w = dict(width=model.out.in_features, latent_dim=int(q.shape[1]),
             latent_lo=np.asarray(lo).tolist(), latent_hi=np.asarray(hi).tolist(), out_log_cap=model.log_cap.tolist(),
             in_mean=np.asarray(in_mean).tolist(), in_std=np.asarray(in_std).tolist(),
             layers=[dict(W=l.weight.detach().numpy().tolist(), b=l.bias.detach().numpy().tolist()) for l in layers])
    Path(json_path).write_text(json.dumps(w))
    Path(bin_path).write_bytes(q.tobytes())


def forward_numpy(weights: dict, q_latents, tri_verts, bary, raw) -> np.ndarray:
    lo, hi = np.array(weights["latent_lo"]), np.array(weights["latent_hi"])
    L = lo + (hi - lo) * np.asarray(q_latents, np.float64) / 255
    lat = (np.asarray(bary, np.float64)[..., None] * L[np.asarray(tri_verts)]).sum(1)
    h = np.concatenate([lat, (np.asarray(raw, np.float64) - weights["in_mean"]) / np.asarray(weights["in_std"])], 1)
    *hidden, out = weights["layers"]
    for l in hidden:
        h = np.maximum(h @ np.array(l["W"]).T + l["b"], 0) ** 2
    return np.exp(np.minimum(h @ np.array(out["W"]).T + out["b"], weights.get("out_log_cap", 80.0)))


def selftest_samples(d, n, seed=1):
    """Validation samples whose third barycentric is >= 0.2: the GPU selftest draws one small
    triangle per sample and needs b2 away from 0."""
    _, va = split_views(d["view_id"])
    ok = np.flatnonzero(va & (d["bary"][:, 2] >= 0.2))
    return np.random.default_rng(seed).choice(ok, n, replace=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pipeline/out/dragon_train.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ft-epochs", type=int, default=5)
    a = p.parse_args(argv)
    m = trimesh.load(WEB_DATA / "dragon.ply", process=False)
    F, nv = np.asarray(m.faces), len(m.vertices)
    d, raw, _, tv = _load(a.data, F)
    pick = selftest_samples(d, 32)
    fx = dict(samples=[dict(tri_verts=tv[i].tolist(), bary=d["bary"][i].tolist(), nrm=d["nrm"][i].tolist(),
                            pos=d["pos"][i].tolist(), cam=d["cam"][i].tolist(), lamp=d["lamp"][i].tolist()) for i in pick],
              expected={})
    for w in WIDTHS:
        model, mean, std, err = train(a.data, F, nv, w, a.epochs)
        q, lo, hi, err_q = finetune_quantized(model, a.data, F, mean, std, a.ft_epochs)
        export(model, mean, std, lo, hi, WEB_DATA / f"weights_{w}.json", WEB_DATA / f"latents_{w}.bin")
        weights = json.loads((WEB_DATA / f"weights_{w}.json").read_text())
        fx["expected"][str(w)] = forward_numpy(weights, q, tv[pick], d["bary"][pick], raw[pick]).tolist()
        print(f"width {w}: val rel err {err:.4f} float, {err_q:.4f} 8-bit latents", flush=True)
    (WEB_DATA / "selftest.json").write_text(json.dumps(fx))


if __name__ == "__main__":
    main()
