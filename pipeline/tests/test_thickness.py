import numpy as np
import trimesh

from pipeline.thickness import vertex_thickness


def _vfn(m):
    return (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
            np.asarray(m.vertex_normals, np.float32))


def test_sphere_thickness_is_mean_cosine_chord():
    r = 0.05  # cosine-weighted mean chord of a sphere = 2r * E[cos] = 4r/3
    t = vertex_thickness(*_vfn(trimesh.creation.icosphere(subdivisions=3, radius=r)), rays=256)
    assert (t > 1.2 * r).all() and (t < 1.47 * r).all()  # 4r/3 ± 3 sigma at 256 rays


def test_thin_cylinder_is_thin():
    t = vertex_thickness(*_vfn(trimesh.creation.cylinder(radius=0.003, height=0.1, sections=32)), rays=32)
    assert np.median(t) < 0.01
