"""Train the SSS MLPs and export them as JSON for the GLSL generator (web/src/mlp.js).
python -m pipeline.train --data pipeline/out/train.npz"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pipeline.assets import WEB_DATA
from pipeline.config import WIDTHS


class MLP(nn.Module):
    """Standardized inputs -> [Linear -> relu(x)^2] x depth -> Linear(3) -> exp.
    relu^2 is C1 (plain ReLU leaves visible creases); exp keeps the output positive."""

    def __init__(self, in_dim=17, width=16, depth=2):
        super().__init__()
        dims = [in_dim] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.out = nn.Linear(width, 3)

    def forward(self, x):
        for l in self.hidden:
            x = torch.relu(l(x)) ** 2
        return torch.exp(self.out(x))


def _split(view_id, frac=0.1, seed=0):
    views = np.unique(view_id)
    val = np.random.default_rng(seed).choice(views, max(1, int(len(views) * frac)), replace=False)
    is_val = np.isin(view_id, val)
    return ~is_val, is_val


def rel_err(pred, y):
    return float(np.abs(pred - y).mean() / np.abs(y).mean())


def train(npz_path, width, epochs=30, batch=4096, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    d = np.load(npz_path)
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    tr, va = _split(d["view_id"])
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = torch.from_numpy((X - mean) / std)
    Yt = torch.from_numpy(Y)
    model = MLP(X.shape[1], width)
    with torch.no_grad():
        model.out.bias.copy_(torch.log(torch.from_numpy(np.maximum(Y[tr].mean(0), 1e-4))))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = np.flatnonzero(tr)
    steps = epochs * max(1, len(idx) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b = torch.from_numpy(idx[rng.integers(0, len(idx), batch)])  # with replacement: O(batch), not O(n)
        pred = model(Xs[b])
        # relative MSE (unbiased on noisy targets, unlike fitting a transformed target)
        loss = ((pred - Yt[b]) ** 2 / (pred.detach() + 0.01) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        err = rel_err(model(Xs[va]).numpy(), Y[va])
    return model, mean, std, err


def export(model, in_mean, in_std, path):
    layers = list(model.hidden) + [model.out]
    w = dict(width=model.out.in_features, in_mean=np.asarray(in_mean).tolist(), in_std=np.asarray(in_std).tolist(),
             layers=[dict(W=l.weight.detach().numpy().tolist(), b=l.bias.detach().numpy().tolist()) for l in layers])
    Path(path).write_text(json.dumps(w))


def forward_numpy(weights: dict, x_raw) -> np.ndarray:
    h = (np.asarray(x_raw, np.float64) - weights["in_mean"]) / np.asarray(weights["in_std"])
    *hidden, out = weights["layers"]
    for l in hidden:
        h = np.maximum(h @ np.array(l["W"]).T + l["b"], 0) ** 2
    return np.exp(h @ np.array(out["W"]).T + out["b"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pipeline/out/train.npz")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args(argv)
    for w in WIDTHS:
        model, mean, std, err = train(a.data, w, a.epochs)
        export(model, mean, std, WEB_DATA / f"weights_{w}.json")
        print(f"width {w}: val rel err {err:.4f}", flush=True)
    # GLSL parity fixture: 64 validation samples through the width-16 network
    d = np.load(a.data)
    _, va = _split(d["view_id"])
    pick = np.random.default_rng(1).choice(np.flatnonzero(va), 64, replace=False)
    w16 = json.loads((WEB_DATA / "weights_16.json").read_text())
    exp = forward_numpy(w16, d["X"][pick])
    fx = [dict(pos=d["pos"][i].tolist(), nrm=d["nrm"][i].tolist(), cam=d["cam"][i].tolist(),
               flame_offset=d["offset"][i].tolist(), expected=e.tolist()) for i, e in zip(pick, exp)]
    (WEB_DATA / "selftest.json").write_text(json.dumps(fx))


if __name__ == "__main__":
    main()
