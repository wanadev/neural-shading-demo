"""Poly Haven HDRI + order-2 SH irradiance (Ramamoorthi & Hanrahan) for the classic baseline.
Direction convention = three.js equirectUv with flipY: u = atan2(z, x)/2pi + 0.5, v = asin(y)/pi + 0.5,
file row 0 = top (v = 1). python -m pipeline.env"""
import json
import urllib.request

import numpy as np

from pipeline.config import ENV, WEB_DATA

A = np.array([np.pi] + [2 * np.pi / 3] * 3 + [np.pi / 4] * 5)  # cosine-lobe convolution per band


def sh_basis(d) -> np.ndarray:
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
                     1.092548 * x * z, 0.546274 * (x * x - y * y)], -1)


def equirect_dirs(h, w):
    u = (np.arange(w) + 0.5) / w
    v = 1 - (np.arange(h) + 0.5) / h
    phi, lat = (u - 0.5) * 2 * np.pi, (v - 0.5) * np.pi
    lat, phi = np.meshgrid(lat, phi, indexing="ij")
    d = np.stack([np.cos(lat) * np.cos(phi), np.sin(lat), np.cos(lat) * np.sin(phi)], -1)
    return d, np.cos(lat) * (2 * np.pi / w) * (np.pi / h)  # directions, solid angle per texel


def sh9_irradiance(hdr) -> np.ndarray:
    """Irradiance SH coefficients (9, 3): E(n) = sum_k c_k Y_k(n)."""
    h, w, _ = hdr.shape
    d, dw = equirect_dirs(h, w)
    L = np.einsum("hwk,hwc,hw->kc", sh_basis(d), np.asarray(hdr, np.float64)[..., :3], dw)
    return (A[:, None] * L).astype(np.float32)


def eval_sh_irradiance(sh, normals) -> np.ndarray:
    return sh_basis(np.asarray(normals, np.float64)) @ np.asarray(sh, np.float64)


def read_hdr(path) -> np.ndarray:
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("scalar_rgb")
    return np.array(mi.Bitmap(str(path)), dtype=np.float32)


def main():
    d = WEB_DATA / "env"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{ENV['name']}_1k.hdr"
    if not f.exists():
        urllib.request.urlretrieve(f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{f.name}", f)
    sh = sh9_irradiance(read_hdr(f) * ENV["scale"])
    (WEB_DATA / "env_sh.json").write_text(json.dumps(dict(file=f"env/{f.name}", sh=sh.tolist())))
    print("env:", f.name, "mean irradiance up", eval_sh_irradiance(sh, [[0, 1, 0]])[0])


if __name__ == "__main__":
    main()
