import numpy as np

from pipeline.config import CAMERA, REF_VIEWS
from pipeline.scene import camera_origin, render


def test_camera_origin_distance():
    o = camera_origin(dict(az_deg=0, el_deg=0, radius=0.3))
    assert np.allclose(o, np.array(CAMERA["target"]) + [0, 0, 0.3], atol=1e-6)


def test_camera_origin_az90_is_plus_x():  # same convention as web/src/refs.js viewDir
    o = camera_origin(dict(az_deg=90, el_deg=0, radius=0.3))
    assert np.allclose(o, np.array(CAMERA["target"]) + [0.3, 0, 0], atol=1e-6)


def test_render_smoke():
    r = render(REF_VIEWS[0], np.zeros(3), res=32, spp=8)
    assert r["full"].shape == (32, 32, 3) and np.isfinite(r["full"]).all()
    assert r["mask"].sum() > 20
    m = r["mask"]
    assert r["full"][m].mean() > r["spec"][m].mean()  # the wax scatters light
    assert np.allclose(np.linalg.norm(r["nrm"][m], axis=1), 1, atol=1e-3)
    # wax pixels lie on the candle
    assert np.hypot(r["pos"][m][:, 0], r["pos"][m][:, 2]).max() < 0.037
