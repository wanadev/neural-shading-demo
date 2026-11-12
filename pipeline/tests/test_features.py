import numpy as np

from pipeline.features import barycentric, raw_features


def test_raw_feature_layout():
    pos = np.array([[0.0, 0.05, 0.0]]); nrm = np.array([[1.0, 0, 0]])
    cam = np.array([[0.3, 0.05, 0.0]]); lamp = np.array([[0.0, 0.05, 0.2]])
    x = raw_features(nrm, pos, cam, lamp)
    assert x.shape == (1, 11) and x.dtype == np.float32
    assert np.allclose(x[0, 0:3], [1, 0, 0])      # nrm
    assert np.allclose(x[0, 3:6], [1, 0, 0])      # view
    assert np.allclose(x[0, 6:9], [0, 0, 1])      # L
    assert np.isclose(x[0, 9], 0.2)               # dist
    assert np.isclose(x[0, 10], 0.0)              # n.L


def test_barycentric_recovers_vertices_and_centroid():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5.0]])
    F = np.array([[0, 1, 2], [1, 2, 3]])
    pts = np.array([V[0], V[1], V[2], V[:3].mean(0)])
    b = barycentric(pts, V, F, np.zeros(4, int))
    assert np.allclose(b[:3], np.eye(3), atol=1e-6)
    assert np.allclose(b[3], [1 / 3] * 3, atol=1e-6)
    assert np.allclose(b.sum(1), 1)


def test_smooth_normals_removes_bumps_and_matches_interp():
    from pipeline.features import interp_normal, mesh_edges, smooth_normals
    n = 12  # flat grid, noisy normals around +Y
    idx = np.arange(n * n).reshape(n, n)
    F = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3),
                        np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    rng = np.random.default_rng(0)
    N = np.array([0, 1, 0]) + rng.normal(0, 0.3, (n * n, 3))
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    E = mesh_edges(F)
    spread = lambda q: np.degrees(np.arccos(np.clip((q[E[:, 0]] * q[E[:, 1]]).sum(1), -1, 1))).mean()
    raw, smooth = smooth_normals(F, N, 0) / 127.0, smooth_normals(F, N, 20) / 127.0
    assert np.allclose(raw, N, atol=1 / 127)                     # 0 iterations = the input, quantized
    assert spread(smooth / np.linalg.norm(smooth, axis=1, keepdims=True)) < spread(N) / 4
    Nq = smooth_normals(F, N, 20)
    p = interp_normal(Nq, F, np.array([0]), np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(p[0], Nq[F[0, 0]] / np.linalg.norm(Nq[F[0, 0]]), atol=1e-6)
