"""Mitsuba scene matching the three.js demo. render() returns full radiance, a specular-only pass
(wax interior replaced by a perfect absorber) and per-pixel AOVs. SSS term = full - spec."""
import numpy as np
import mitsuba as mi

try:
    mi.set_variant("llvm_ad_rgb")
except Exception:  # ponytail: scalar fallback is ~10x slower, only for machines without LLVM
    mi.set_variant("scalar_rgb")

from pipeline.assets import WEB_DATA
from pipeline.config import CAMERA, CANDLE, FLAME, TABLE, WAX

MAX_DEPTH = 1024


def camera_origin(view: dict) -> np.ndarray:
    a, e = np.radians(view["az_deg"]), np.radians(view["el_deg"])
    d = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    return np.array(CAMERA["target"]) + view["radius"] * d


def _scene(view, flame_offset, res, absorb):
    """absorb=False: index-matched scattering wax (null boundary) -> shadow rays from inside the
    medium reach the flame, so SSS converges. absorb=True: rough dielectric over a perfect absorber
    -> surface reflection only. The wax is modelled as SSS + specular coat (same model in three.js)."""
    T = mi.ScalarTransform4f
    if absorb:
        medium = dict(type="homogeneous", sigma_t=1e5, albedo=0.0)
        bsdf = dict(type="roughdielectric", distribution="ggx", int_ior=WAX["ior"], ext_ior=1.0, alpha=WAX["roughness"])
    else:
        medium = dict(type="homogeneous", sigma_t=WAX["sigma_t"], albedo=dict(type="rgb", value=list(WAX["albedo"])))
        bsdf = dict(type="null")
    half = TABLE["size"] / 2
    return mi.load_dict({
        "type": "scene",
        "sensor": dict(
            type="perspective", fov=CAMERA["fov_y"], fov_axis="y",
            to_world=T().look_at(origin=camera_origin(view).tolist(), target=list(CAMERA["target"]), up=[0, 1, 0]),
            film=dict(type="hdrfilm", width=res, height=res, rfilter=dict(type="box"), pixel_format="rgb")),
        "wax": dict(type="obj", filename=str(WEB_DATA / "candle.obj"), face_normals=False,
                    bsdf=bsdf, interior=medium),
        "wick": dict(type="obj", filename=str(WEB_DATA / "wick.obj"),
                     bsdf=dict(type="diffuse", reflectance=dict(type="rgb", value=[0.02] * 3))),
        "flame": dict(type="sphere", center=(np.array(FLAME["base"]) + flame_offset).tolist(),
                      radius=FLAME["sphere_radius"],
                      emitter=dict(type="area", radiance=dict(type="rgb", value=list(FLAME["radiance"])))),
        "table": dict(type="rectangle", to_world=T().rotate([1, 0, 0], -90).scale([half, half, 1]),
                      bsdf=dict(type="diffuse", reflectance=dict(type="rgb", value=list(TABLE["albedo"])))),
    })


def _np(img):
    return np.array(img, dtype=np.float32)


def render(view: dict, flame_offset: np.ndarray, res: int, spp: int, max_pass_samples: int = 2**27) -> dict:
    flame_offset = np.asarray(flame_offset, dtype=np.float64)
    vol = mi.load_dict(dict(type="volpath", max_depth=MAX_DEPTH))
    full_scene = _scene(view, flame_offset, res, absorb=False)
    spec_scene = _scene(view, flame_offset, res, absorb=True)
    # split spp into passes with distinct seeds, spp-weighted average: bounded work per call
    per = max(1, max_pass_samples // (res * res))
    chunks = [min(per, spp - s) for s in range(0, spp, per)]
    avg = lambda sc: sum(_np(mi.render(sc, integrator=vol, spp=n, seed=k)) * n for k, n in enumerate(chunks)) / spp
    sss_pass, spec_pass = avg(full_scene), avg(spec_scene)
    # AOVs at 1 spp so positions are never averaged across a silhouette
    aov = mi.load_dict(dict(type="aov", aovs="pos:position,nn:sh_normal"))
    a = _np(mi.render(full_scene, integrator=aov, spp=1))
    pos, nrm = a[..., 0:3], a[..., 3:6]
    # wax mask by geometry: shape_index order does not follow scene.shapes() order
    r, y = np.hypot(pos[..., 0], pos[..., 2]), pos[..., 1]
    wick = (r < 0.001) & (y >= CANDLE["height"] - CANDLE["cup_depth"] + 1e-4)
    mask = ((r <= CANDLE["radius"] + CANDLE["wobble"] + 1e-4) & (y > 1e-4) & (y <= CANDLE["height"] + 1e-4)
            & ~wick & (np.linalg.norm(nrm, axis=-1) > 0.5))
    # drop silhouette pixels: radiance there mixes wax and background
    m = mask.copy()
    m[1:, :] &= mask[:-1, :]; m[:-1, :] &= mask[1:, :]
    m[:, 1:] &= mask[:, :-1]; m[:, :-1] &= mask[:, 1:]
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8)
    # wax pixels = SSS + coat reflection; elsewhere the SSS pass is the image
    spec = np.where(mask[..., None], spec_pass, 0.0).astype(np.float32)
    return dict(full=sss_pass + spec, spec=spec, pos=pos, nrm=nrm, mask=m)
