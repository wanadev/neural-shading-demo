"""Classic real-time translucency (wrap lighting + thickness transmittance), fitted on the same data
as the network so the baseline is the best version of itself. Mirrored in GLSL (web/src/materials.js).
python -m pipeline.classic --data pipeline/out/train.npz"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as Fn

from pipeline.assets import WEB_DATA
from pipeline.config import CANDLE, FLAME
from pipeline.features import chord_to_flame
from pipeline.train import _split, rel_err


def classic_sss(pos, nrm, cam, flame, p: dict) -> np.ndarray:
    """sss = (1/d^2) * tint * (a * max((n.L + w)/(1 + w), 0) + b * exp(-k * chord)). cam unused (view-independent)."""
    pos, nrm, flame = np.asarray(pos, np.float64), np.asarray(nrm, np.float64), np.asarray(flame, np.float64)
    to_l = flame - pos
    d = np.linalg.norm(to_l, axis=-1, keepdims=True)
    ndl = np.sum(nrm * to_l / d, axis=-1, keepdims=True)
    c = chord_to_flame(pos, flame, CANDLE["radius"], CANDLE["height"])[:, None]
    wrap = np.maximum((ndl + p["w"]) / (1 + p["w"]), 0)
    return np.asarray(p["tint"]) / d**2 * (p["a"] * wrap + p["b"] * np.exp(-p["k"] * c))


def fit_classic(npz_path, steps=3000, lr=0.05, seed=0):
    d = np.load(npz_path)
    pos, nrm = d["pos"].astype(np.float64), d["nrm"].astype(np.float64)
    flame = np.array(FLAME["base"]) + d["offset"]
    to_l = flame - pos
    dist = np.linalg.norm(to_l, axis=-1, keepdims=True)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    inv_d2, ndl = t(1 / dist**2), t(np.sum(nrm * to_l / dist, -1, keepdims=True))
    chord = t(chord_to_flame(pos, flame, CANDLE["radius"], CANDLE["height"])[:, None])
    Y = t(d["Y"])
    tr, va = _split(d["view_id"])
    torch.manual_seed(seed)
    raw = {k: torch.zeros(s, requires_grad=True) for k, s in (("w", 1), ("k", 1), ("a", 1), ("b", 1), ("tint", 3))}
    with torch.no_grad():  # sensible starting point: k ~ 1/(2 cm), tint ~ data scale
        raw["k"].fill_(50.0)
        raw["tint"].copy_(torch.log(torch.expm1(Y[tr].mean(0) / inv_d2[tr].mean() + 1e-6)))
    params = lambda: {k: Fn.softplus(v) for k, v in raw.items()}

    def model(m):
        p = params()
        wrap = torch.clamp((ndl[m] + p["w"]) / (1 + p["w"]), min=0)
        return p["tint"] * inv_d2[m] * (p["a"] * wrap + p["b"] * torch.exp(-p["k"] * chord[m]))

    opt = torch.optim.Adam(raw.values(), lr=lr)
    trt = torch.from_numpy(tr)
    for _ in range(steps):
        pred = model(trt)
        loss = ((pred - Y[trt]) ** 2 / (pred.detach() + 0.01) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        err = rel_err(model(torch.from_numpy(va)).numpy(), d["Y"][va])
        p = {k: (v.tolist() if k == "tint" else float(v)) for k, v in params().items()}
    return p, err


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline/out/train.npz")
    a = ap.parse_args(argv)
    p, err = fit_classic(a.data)
    (WEB_DATA / "classic.json").write_text(json.dumps(p, indent=1))
    print(f"classic: val rel err {err:.4f}", p)


if __name__ == "__main__":
    main()
