import json

import numpy as np
import torch

from pipeline.train import MLP, export, forward_numpy, train


def test_forward_numpy_matches_torch(tmp_path):
    torch.manual_seed(0)
    m = MLP(17, 8)
    mean, std = np.linspace(-1, 1, 17), np.linspace(0.5, 2, 17)
    export(m, mean, std, tmp_path / "w.json")
    w = json.loads((tmp_path / "w.json").read_text())
    assert w["width"] == 8 and len(w["layers"]) == 3
    assert np.array(w["layers"][0]["W"]).shape == (8, 17)
    x = np.random.default_rng(0).normal(size=(10, 17)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy((x - mean) / std).float()).numpy()
    assert np.allclose(forward_numpy(w, x), ref, rtol=1e-5)


def test_overfits_toy(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 17)).astype(np.float32)
    Y = (0.5 + 0.1 * X[:, :3]).astype(np.float32)
    p = tmp_path / "toy.npz"
    np.savez(p, X=X, Y=Y, view_id=np.arange(4000) % 20)
    _, _, _, err = train(p, 16, epochs=60, batch=256)
    assert err < 0.05
