"""Verbatim port of three.js r186 (0.186.1) ACESFilmicToneMapping + sRGBTransferOETF
(src/renderers/shaders/ShaderChunk/{tonemapping,colorspace}_pars_fragment.glsl.js),
so reference PNGs and the WebGL render share one tone curve."""
import numpy as np

# rows = GLSL mat3 columns, so `color @ M` == GLSL `M * color`
_IN = np.array([[0.59719, 0.07600, 0.02840], [0.35458, 0.90834, 0.13383], [0.04823, 0.01566, 0.83777]])
_OUT = np.array([[1.60475, -0.10208, -0.00327], [-0.53108, 1.10813, -0.07276], [-0.07367, -0.00605, 1.07602]])


def _rrt_odt(v):
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b


def aces_filmic(rgb_linear, exposure):
    c = np.asarray(rgb_linear, np.float64) * (exposure / 0.6)
    return np.clip(_rrt_odt(c @ _IN) @ _OUT, 0, 1)


def to_srgb8(x):
    x = np.asarray(x, np.float64)
    s = np.where(x <= 0.0031308, x * 12.92, np.power(np.maximum(x, 0), 0.41666) * 1.055 - 0.055)
    return np.round(np.clip(s, 0, 1) * 255).astype(np.uint8)
