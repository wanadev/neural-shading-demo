import numpy as np

from pipeline.env import eval_sh_irradiance, sh9_irradiance


def test_sh_constant_env():  # uniform radiance 1 -> irradiance pi for any normal
    sh = sh9_irradiance(np.ones((64, 128, 3), np.float32))
    e = eval_sh_irradiance(sh, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]))
    assert np.allclose(e, np.pi, rtol=0.02)


def test_sh_direction_convention():  # bright rows at the top of the image -> max irradiance toward +Y
    img = np.zeros((64, 128, 3), np.float32)
    img[0:4] = 50.0
    sh = sh9_irradiance(img)
    up, down = eval_sh_irradiance(sh, np.array([[0, 1, 0], [0, -1, 0.0]]))[:, 0]
    assert up > 5 * down


def test_sh_azimuth_convention():  # bright column at u=0.75 -> three.js direction atan2(z,x)=pi/2 -> +Z
    img = np.zeros((64, 128, 3), np.float32)
    img[24:40, 94:98] = 50.0
    sh = sh9_irradiance(img)
    pz, nz = eval_sh_irradiance(sh, np.array([[0, 0, 1], [0, 0, -1.0]]))[:, 0]
    assert pz > 3 * nz
