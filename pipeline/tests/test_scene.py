import numpy as np
import mitsuba as mi

from pipeline.config import CAMERA, LAMP, REF_POSES
from pipeline.scene import ENV_ROT_Y_DEG, camera_origin, lamp_position, render


def three_equirect_dir(u, v):  # inverse of three.js equirectUv
    phi, lat = (u - 0.5) * 2 * np.pi, (v - 0.5) * np.pi
    return np.array([np.cos(lat) * np.cos(phi), np.sin(lat), np.cos(lat) * np.sin(phi)])


def marker_hdr(u, v, h=64, w=128):  # bright block at three.js (u, v); file row 0 = top (v = 1)
    img = np.full((h, w, 3), 0.01, np.float32)
    r, c = int((1 - v) * h), int(u * w)
    img[r - 2:r + 2, c - 2:c + 2] = 100.0
    return img


def eval_env_direction(img, d):
    T = mi.ScalarTransform4f
    em = mi.load_dict(dict(type="envmap", bitmap=mi.Bitmap(img), to_world=T().rotate([0, 1, 0], ENV_ROT_Y_DEG)))
    si = mi.SurfaceInteraction3f()
    si.wi = mi.Vector3f(-d)  # Mitsuba envmap looks up -wi
    return float(np.array(em.eval(si)).ravel()[0])


def test_env_orientation_matches_three():
    for u, v in ((0.8, 0.7), (0.3, 0.4)):
        img, d = marker_hdr(u, v), three_equirect_dir(u, v)
        assert eval_env_direction(img, d) > 10 * eval_env_direction(img, -d)
        assert eval_env_direction(img, d) > 50  # hits the block itself


def test_camera_origin_distance():
    o = camera_origin(dict(az_deg=0, el_deg=0, radius=0.3))
    assert np.allclose(o, np.array(CAMERA["target"]) + [0, 0, 0.3], atol=1e-6)


def test_lamp_position_convention():
    p = lamp_position(90, 0)
    assert np.allclose(p, np.array(LAMP["center"]) + [LAMP["radius"], 0, 0], atol=1e-6)


def test_render_passes_smoke():
    pose = REF_POSES[0]
    r = render(pose, (pose["lamp_az_deg"], pose["lamp_el_deg"]), res=32, spp=8,
               passes=("env", "lamp", "both", "spec"))
    m = r["mask"]
    assert m.sum() > 30 and (r["tri"][m] >= 0).all()
    assert r["env"][m].mean() > 0 and r["lamp"][m].mean() > 0
    assert r["both"].shape == (32, 32, 4)
    assert (r["both"][..., 3][m] > 0.99).all()  # dragon pixels are opaque in alpha
    assert (r["both"][0, :, 3] < 0.01).any()  # sky above the horizon is transparent
    assert np.allclose(np.linalg.norm(r["nrm"][m], axis=1), 1, atol=1e-3)
