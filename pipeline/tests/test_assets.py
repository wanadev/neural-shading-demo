import numpy as np
import pytest

from pipeline.assets import candle_mesh, wick_mesh, write_obj
from pipeline.config import CANDLE


def test_candle_bounds():
    V, F, N = candle_mesh()
    assert V[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert V[:, 1].max() <= CANDLE["height"] + 1e-6
    r = np.hypot(V[:, 0], V[:, 2])
    assert r.max() <= CANDLE["radius"] + CANDLE["wobble"] + 1e-6


def _closed(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return (counts == 2).all()


def test_candle_closed():
    _, F, _ = candle_mesh()
    assert _closed(F)


def test_wick_closed():
    _, F, _ = wick_mesh()
    assert _closed(F)


def test_normals_outward_unit():
    V, _, N = candle_mesh()
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-4)
    side = (V[:, 1] > 0.01) & (V[:, 1] < 0.08)
    assert (np.einsum("ij,ij->i", N[side][:, [0, 2]], V[side][:, [0, 2]]) > 0).all()


def test_faces_wound_outward():
    # signed volume > 0 means CCW-outward winding (matches the normals)
    V, F, _ = candle_mesh()
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0


def test_obj_roundtrip(tmp_path):
    V, F, N = candle_mesh()
    p = tmp_path / "c.obj"
    write_obj(p, V, F, N)
    lines = p.read_text().splitlines()
    assert sum(l.startswith("v ") for l in lines) == len(V)
    assert sum(l.startswith("vn ") for l in lines) == len(V)
    faces = [l for l in lines if l.startswith("f ")]
    assert len(faces) == len(F)
    assert faces[0] == "f {0}//{0} {1}//{1} {2}//{2}".format(*(F[0] + 1))
