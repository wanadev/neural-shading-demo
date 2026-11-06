"""Classic real-time translucency ("local thickness", DICE/Frostbite style) + SH ambient, fitted on the
same data as the network so the baseline is the best version of itself. Mirrored by CLASSIC_GLSL.
python -m pipeline.classic"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh

from pipeline.config import WEB_DATA
from pipeline.env import sh_basis
from pipeline.train import _split, rel_err

KEYS3 = ("tint_d", "tint_t", "tint_e", "tint_eb")


def _terms(nrm, pos, cam, lamp, xp):
    to_l = lamp - pos
    d = xp.sqrt((to_l ** 2).sum(-1, keepdims=True))
    L = to_l / d
    v = cam - pos
    V = v / xp.sqrt((v ** 2).sum(-1, keepdims=True))
    return L, d, V


def classic_sss(nrm, pos, cam, lamp, th, sh, p) -> np.ndarray:
    """-> (N, 6): env RGB | lamp RGB.
    lamp = (tint_d * wrap + tint_t * exp(-k th) * max(V . -normalize(L + n dist), 0)^p) / d^2
    env  = tint_e * E(n) + tint_eb * E(-n) * exp(-k th)"""
    nrm, pos, cam, lamp = (np.asarray(a, np.float64) for a in (nrm, pos, cam, lamp))
    th = np.asarray(th, np.float64)[:, None]
    L, d, V = _terms(nrm, pos, cam, lamp, np)
    wrap = np.maximum(((nrm * L).sum(-1, keepdims=True) + p["w"]) / (1 + p["w"]), 0)
    h = L + nrm * p["dist"]
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    tr = np.exp(-p["k"] * th) * np.maximum((V * -h).sum(-1, keepdims=True), 0) ** p["p"]
    lamp_t = (np.asarray(p["tint_d"]) * wrap + np.asarray(p["tint_t"]) * tr) / d ** 2
    E = lambda n: sh_basis(n) @ np.asarray(sh, np.float64)
    env_t = np.asarray(p["tint_e"]) * E(nrm) + np.asarray(p["tint_eb"]) * E(-nrm) * np.exp(-p["k"] * th)
    return np.concatenate([env_t, lamp_t], 1)


def _load_defaults():
    m = trimesh.load(WEB_DATA / "dragon.ply", process=False)
    cfg = json.loads((WEB_DATA / "config.json").read_text())
    vert_th = np.frombuffer((WEB_DATA / "thickness.bin").read_bytes(), np.uint8) / 255 * cfg["thickness_max"]
    sh = np.array(json.loads((WEB_DATA / "env_sh.json").read_text())["sh"]) * cfg["env"]["scale"]
    return np.asarray(m.faces), vert_th, sh


def fit_classic(npz_path, F=None, vert_th=None, sh=None, steps=3000, lr=0.03, seed=0, max_train=300_000):
    if F is None:
        F, vert_th, sh = _load_defaults()
    d = np.load(npz_path)
    th = (d["bary"] * np.asarray(vert_th)[np.asarray(F)[d["tri"]]]).sum(1)
    Y = np.concatenate([d["Y_env"], d["Y_lamp"]], 1)
    tr, va = _split(d["view_id"])
    if tr.sum() > max_train:  # full-batch fit of ~15 parameters: a fixed random subset is plenty
        keep = np.random.default_rng(seed).choice(np.flatnonzero(tr), max_train, replace=False)
        tr = np.zeros_like(tr); tr[keep] = True
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    nrm, pos, cam, lamp, TH = t(d["nrm"]), t(d["pos"]), t(d["cam"]), t(d["lamp"]), t(th)[:, None]
    En, Em = t(sh_basis(d["nrm"].astype(np.float64)) @ sh), t(sh_basis(-d["nrm"].astype(np.float64)) @ sh)
    L, dist, V = _terms(nrm, pos, cam, lamp, torch)
    ndl = (nrm * L).sum(-1, keepdims=True)
    Yt = t(Y)
    torch.manual_seed(seed)
    inv_sp = lambda x: float(np.log(np.expm1(x)))
    raw = {"w": torch.tensor([inv_sp(0.5)]), "k": torch.tensor([inv_sp(30.0)]), "p": torch.tensor([inv_sp(3.0)]),
           "dist": torch.tensor([inv_sp(0.2)])}
    scale = Y[tr].mean(0)
    for i, k in enumerate(KEYS3):
        cols = slice(0, 3) if k in ("tint_e", "tint_eb") else slice(3, 6)
        raw[k] = torch.tensor([inv_sp(max(s, 1e-4) * (0.3 if k.startswith("tint_e") else 0.02)) for s in scale[cols]])
    raw = {k: v.double().requires_grad_() for k, v in raw.items()}

    def params():
        p = {k: Fn.softplus(v) for k, v in raw.items()}
        p["p"] = 1 + p["p"]  # exponent >= 1
        return p

    def model(m):
        p = params()
        wrap = torch.clamp((ndl[m] + p["w"]) / (1 + p["w"]), min=0)
        h = L[m] + nrm[m] * p["dist"]
        h = h / h.norm(dim=-1, keepdim=True)
        trn = torch.exp(-p["k"] * TH[m]) * torch.clamp((V[m] * -h).sum(-1, keepdim=True), min=0) ** p["p"]
        lamp_t = (p["tint_d"] * wrap + p["tint_t"] * trn) / dist[m] ** 2
        env_t = p["tint_e"] * En[m] + p["tint_eb"] * Em[m] * torch.exp(-p["k"] * TH[m])
        return torch.cat([env_t, lamp_t], 1)

    opt = torch.optim.Adam(raw.values(), lr=lr)
    trt = torch.from_numpy(tr)
    for _ in range(steps):
        pred = model(trt)
        loss = ((pred - Yt[trt]) ** 2 / (pred.detach() + 0.01) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        err = rel_err(model(torch.from_numpy(va)).numpy(), Y[va])
        p = {k: (v.tolist() if v.numel() > 1 else float(v)) for k, v in params().items()}
    return p, err


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pipeline/out/dragon_train.npz")
    a = ap.parse_args(argv)
    p, err = fit_classic(a.data)
    (WEB_DATA / "classic.json").write_text(json.dumps(p, indent=1))
    print(f"classic: val rel err {err:.4f}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})


if __name__ == "__main__":
    main()
