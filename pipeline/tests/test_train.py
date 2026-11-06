import json

import numpy as np
import torch

from pipeline.train import NeuralSSS, export, finetune_quantized, forward_numpy, quantize, selftest_samples, train


def _toy(tmp_path, n_vert=50, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.integers(0, n_vert, (80, 3))
    true = rng.normal(size=(n_vert, 8))
    tri = rng.integers(0, len(F), n)
    bary = rng.dirichlet([2, 2, 2], n).astype(np.float32)
    lat = (bary[:, :, None] * true[F[tri]]).sum(1)
    pos = rng.normal(0, 0.05, (n, 3)); nrm = rng.normal(size=(n, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    cam = np.tile([0.4, 0.1, 0.0], (n, 1)); lamp = rng.normal(0, 0.25, (n, 3))
    w = rng.normal(size=(8, 6))
    Y = 0.5 + 0.2 * np.tanh(lat @ w * 0.5 + nrm[:, :1])
    p = tmp_path / "toy.npz"
    np.savez(p, tri=tri, bary=bary, nrm=nrm, pos=pos, cam=cam, lamp=lamp, Y_env=Y[:, :3], Y_lamp=Y[:, 3:],
             view_id=np.arange(n) % 30)
    return p, F, n_vert


def test_quantization_roundtrip():
    x = torch.randn(100, 8)
    q, lo, hi = quantize(x)
    deq = lo + (hi - lo) * q.astype(np.float32) / 255
    assert q.dtype == np.uint8 and np.all(np.abs(deq - x.numpy()) <= (hi - lo) / 255 / 2 + 1e-6)


def test_forward_numpy_matches_torch(tmp_path):
    torch.manual_seed(0)
    m = NeuralSSS(20, 16)
    mean, std = np.linspace(-1, 1, 11), np.linspace(0.5, 2, 11)
    _, lo, hi = quantize(m.latents.detach())
    export(m, mean, std, lo, hi, tmp_path / "w.json", tmp_path / "l.bin")
    w = json.loads((tmp_path / "w.json").read_text())
    q = np.frombuffer((tmp_path / "l.bin").read_bytes(), np.uint8).reshape(20, 8)
    rng = np.random.default_rng(0)
    tv, bary, raw = rng.integers(0, 20, (10, 3)), rng.dirichlet([1, 1, 1], 10), rng.normal(size=(10, 11))
    deq = torch.tensor(lo + (hi - lo) * q.astype(np.float32) / 255)
    with torch.no_grad():
        ref = m(torch.tensor(tv), torch.tensor(bary, dtype=torch.float32),
                torch.tensor((raw - mean) / std, dtype=torch.float32), latents=deq).numpy()
    assert np.allclose(forward_numpy(w, q, tv, bary, raw), ref, rtol=1e-4)


def test_export_sizes_match_mesh(tmp_path):  # Review Focus #2
    m = NeuralSSS(37, 32)
    _, lo, hi = quantize(m.latents.detach())
    export(m, np.zeros(11), np.ones(11), lo, hi, tmp_path / "w.json", tmp_path / "l.bin")
    w = json.loads((tmp_path / "w.json").read_text())
    assert len((tmp_path / "l.bin").read_bytes()) == 37 * 8
    assert np.array(w["layers"][0]["W"]).shape == (32, 19) and np.array(w["layers"][-1]["W"]).shape == (6, 32)
    assert len(w["latent_lo"]) == len(w["latent_hi"]) == 8 and len(w["in_mean"]) == 11


def test_overfits_toy_and_survives_quantization(tmp_path):
    p, F, n_vert = _toy(tmp_path)
    model, mean, std, err = train(p, F, n_vert, 16, epochs=150, batch=512)
    assert err < 0.05
    q, lo, hi, err_q = finetune_quantized(model, p, F, mean, std, epochs=20, batch=512)
    assert err_q < 0.06


def test_selftest_samples_avoid_degenerate_triangles(tmp_path):
    p, F, _ = _toy(tmp_path)
    d = np.load(p)
    idx = selftest_samples(d, 32)
    assert len(idx) == 32 and (d["bary"][idx, 2] >= 0.2).all()
