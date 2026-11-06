import numpy as np

from pipeline.classic import classic_sss, fit_classic

SH = np.zeros((9, 3), np.float32); SH[0] = 3.0; SH[1] = 1.0  # bright ambient, a bit more from above
TRUE = dict(w=0.3, k=40.0, p=4.0, dist=0.2, tint_d=[0.01, 0.05, 0.02], tint_t=[0.02, 0.08, 0.04],
            tint_e=[0.02, 0.06, 0.03], tint_eb=[0.01, 0.04, 0.02])


def _inputs(n, rng):
    nrm = rng.normal(size=(n, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    pos = rng.normal(0, 0.03, (n, 3)) + [0, 0.07, 0]
    cam = np.tile([0.4, 0.1, 0.1], (n, 1)); lamp = rng.normal(size=(n, 3))
    lamp = lamp / np.linalg.norm(lamp, axis=1, keepdims=True) * 0.25 + [0, 0.07, 0]
    return nrm, pos, cam, lamp


def test_classic_positive_and_lamp_term_decays_with_thickness():
    nrm, pos, cam, lamp = _inputs(200, np.random.default_rng(0))
    p = dict(TRUE, tint_d=[0, 0, 0])
    thin = classic_sss(nrm, pos, cam, lamp, np.full(200, 0.002), SH, p)
    thick = classic_sss(nrm, pos, cam, lamp, np.full(200, 0.04), SH, p)
    assert (thin >= 0).all() and thin.shape == (200, 6)
    assert (thick[:, 3:] <= thin[:, 3:] + 1e-12).all() and thick[:, 3:].sum() < thin[:, 3:].sum()


def test_fit_recovers_synthetic_params(tmp_path):
    rng = np.random.default_rng(1)
    n, nv = 4000, 60
    F = rng.integers(0, nv, (100, 3)); vert_th = rng.uniform(0.001, 0.05, nv)
    tri = rng.integers(0, 100, n); bary = rng.dirichlet([2, 2, 2], n)
    th = (bary * vert_th[F[tri]]).sum(1)
    nrm, pos, cam, lamp = _inputs(n, rng)
    Y = classic_sss(nrm, pos, cam, lamp, th, SH, TRUE)
    f = tmp_path / "d.npz"
    np.savez(f, tri=tri, bary=bary, nrm=nrm, pos=pos, cam=cam, lamp=lamp, Y_env=Y[:, :3], Y_lamp=Y[:, 3:],
             view_id=np.arange(n) % 30)
    params, err = fit_classic(f, F=F, vert_th=vert_th, sh=SH, steps=2500)
    assert err < 0.05
    assert set(params) == set(TRUE)
