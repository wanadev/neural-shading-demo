import numpy as np
import trimesh

from pipeline.dragon import prepare, write_ply


def holed_sphere():
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    keep = np.ones(len(m.faces), bool)
    keep[:6] = False  # punch a hole
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


def test_prepare_is_closed_scaled_grounded():
    V, F, N = prepare(holed_sphere(), target_faces=500, height=0.15)
    m = trimesh.Trimesh(V, F, process=False)
    assert m.is_watertight and m.volume > 0  # closed, outward-facing
    assert abs(V[:, 1].min()) < 1e-6 and abs(V[:, 1].max() - 0.15) < 1e-4
    assert abs(V[:, 0].mean()) < 0.01 and len(F) <= 550
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-4)


def test_ply_roundtrip_keeps_vertex_order(tmp_path):  # Review Focus #2
    V, F, N = prepare(holed_sphere(), 500, 0.15)
    write_ply(tmp_path / "d.ply", V, F, N)
    m = trimesh.load(tmp_path / "d.ply", process=False)
    assert np.allclose(m.vertices, V) and (m.faces == F).all()


def test_cli_never_overwrites_existing_mesh(tmp_path):
    # decimation is not bit-reproducible: regenerating would silently reorder vertices under
    # every per-vertex file (latents, thickness, training data)
    from pipeline.dragon import main
    f = tmp_path / "dragon.ply"
    f.write_bytes(b"keep me")
    main(["--out", str(tmp_path)])
    assert f.read_bytes() == b"keep me"
