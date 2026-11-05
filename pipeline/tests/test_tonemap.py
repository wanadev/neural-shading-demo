import numpy as np

from pipeline.tonemap import aces_filmic, to_srgb8


def test_zero_maps_to_zero():
    assert np.allclose(aces_filmic(np.zeros((1, 3)), 1.0), 0, atol=1e-3)


def test_monotonic_grey():
    x = np.linspace(0, 50, 200)[:, None].repeat(3, 1)
    y = aces_filmic(x, 1.0)[:, 0]
    assert (np.diff(y) >= -1e-9).all()


def test_saturates():
    assert (aces_filmic(np.full((1, 3), 1e4), 1.0) > 0.99).all()


def test_srgb8_endpoints():
    assert to_srgb8(np.array([0.0, 1.0])).tolist() == [0, 255]


def test_matches_three_mid_grey():
    # hand-evaluated three r186 ACESFilmicToneMapping(vec3(0.18)) with exposure 1
    v = 0.18 / 0.6
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    assert np.allclose(aces_filmic(np.full((1, 3), 0.18), 1.0), a / b, atol=2e-3)  # grey stays ~grey
