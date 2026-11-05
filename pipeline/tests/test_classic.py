import numpy as np

from pipeline.classic import classic_sss, fit_classic

P = dict(w=0.5, k=50.0, a=0.0, b=1.0, tint=[1.0, 0.6, 0.3])


def test_classic_decays_with_chord():
    flame = np.array([0, 0.11, 0])
    nrm = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    pos = np.array([[0.035, 0.09, 0], [0.035, 0.03, 0]])  # lower point: longer chord
    cam = np.array([0.3, 0.06, 0])
    s = classic_sss(pos, nrm, cam, flame, P)
    assert (s >= 0).all()
    assert (s[1] < s[0]).all()


def test_fit_recovers_params(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    th = rng.uniform(0, 2 * np.pi, n); y = rng.uniform(0.01, 0.1, n)
    pos = np.stack([0.035 * np.cos(th), y, 0.035 * np.sin(th)], 1)
    nrm = np.stack([np.cos(th), np.zeros(n), np.sin(th)], 1)
    off = rng.uniform(-0.003, 0.003, (n, 3)); cam = np.tile([0.3, 0.06, 0], (n, 1))
    flame = np.array([0, 0.11, 0]) + off
    true = dict(w=0.3, k=40.0, a=0.2, b=1.5, tint=[1.0, 0.5, 0.2])
    Y = np.stack([classic_sss(pos[i:i + 1], nrm[i:i + 1], cam[i], flame[i], true)[0] for i in range(n)])
    p = tmp_path / "d.npz"
    np.savez(p, pos=pos, nrm=nrm, cam=cam, offset=off, Y=Y, view_id=np.arange(n) % 30)
    params, err = fit_classic(p, steps=1500)
    assert err < 0.05
