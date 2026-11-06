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
