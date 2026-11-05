import numpy as np

from pipeline.config import FLAME
from pipeline.features import chord_to_flame, features


def test_feature_layout():
    pos = np.array([[0.035, 0.05, 0.0]]); nrm = np.array([[1.0, 0, 0]])
    x = features(pos, nrm, cam=np.array([0.3, 0.05, 0]), flame_offset=np.zeros(3))
    assert x.shape == (1, 17) and x.dtype == np.float32
    assert np.allclose(x[0, 0:3], pos[0]) and np.allclose(x[0, 3:6], nrm[0])
    assert np.allclose(x[0, 6:9], [1, 0, 0])                      # view
    assert np.allclose(x[0, 9:12], 0)                             # flame offset
    L = np.array(FLAME["base"]) - pos[0]; d = np.linalg.norm(L)
    assert np.allclose(x[0, 12:15], L / d) and np.isclose(x[0, 15], d)
    assert np.isclose(x[0, 16], (L / d)[0])


def test_chord_side_point_to_axis_flame():
    # side point at y=0.05, flame on the axis at y=0.05: chord = radius
    c = chord_to_flame(np.array([[0.035, 0.05, 0]]), np.array([0, 0.05, 0]), 0.035, 0.10)
    assert np.isclose(c[0], 0.035, atol=1e-6)


def test_chord_flame_above_top():
    # side point straight below the rim, flame above the top: path exits through the top cap
    c = chord_to_flame(np.array([[0.0, 0.05, 0]]), np.array([0, 0.11, 0]), 0.035, 0.10)
    assert np.isclose(c[0], 0.05, atol=1e-6)


def test_chord_outside_is_zero():
    c = chord_to_flame(np.array([[0.2, 0.0, 0]]), np.array([0, 0.5, 0]), 0.035, 0.10)
    assert c[0] == 0
