"""Mitsuba scene matching the three.js demo: jade dragon (index-matched scattering volume), dark glossy
ground, HDRI room, movable sphere lamp. Separate passes so the network learns env and lamp SSS apart:
  env  : HDRI only           -> SSS due to the room
  lamp : lamp only           -> SSS due to the lamp
  both : HDRI + lamp, RGBA   -> full reference image (no coat)
  spec : coat reflection only (rough GGX dielectric over an absorber), kept on dragon pixels
With a null boundary the radiance leaving the dragon IS the SSS term."""
import numpy as np
import mitsuba as mi

try:
    mi.set_variant("llvm_ad_rgb")
except Exception:  # ponytail: scalar fallback is ~10x slower, only for machines without LLVM
    mi.set_variant("scalar_rgb")

from pipeline.config import CAMERA, ENV, GROUND, JADE, LAMP, WEB_DATA

MAX_DEPTH = 1024
ENV_ROT_Y_DEG = 90.0  # aligns Mitsuba's envmap mapping with three.js equirectUv (pinned by a test)


def _dir(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])


def camera_origin(view: dict) -> np.ndarray:
    return np.array(CAMERA["target"]) + view["radius"] * _dir(view["az_deg"], view["el_deg"])


def lamp_position(az_deg, el_deg) -> np.ndarray:
    return np.array(LAMP["center"]) + LAMP["radius"] * _dir(az_deg, el_deg)


def _sensor(view, res, rgba=False):
    T = mi.ScalarTransform4f
    return dict(type="perspective", fov=CAMERA["fov_y"], fov_axis="y",
                to_world=T().look_at(origin=camera_origin(view).tolist(), target=list(CAMERA["target"]), up=[0, 1, 0]),
                film=dict(type="hdrfilm", width=res, height=res, rfilter=dict(type="box"),
                          pixel_format="rgba" if rgba else "rgb"))


def _scene(view, lamp, res, *, env, lamp_on, coat=False, rgba=False, ground=True, dragon_only=False):
    T = mi.ScalarTransform4f
    if coat:
        bsdf = dict(type="roughdielectric", distribution="ggx", int_ior=JADE["ior"], ext_ior=1.0, alpha=JADE["roughness"])
        medium = dict(type="homogeneous", sigma_t=1e5, albedo=0.0)
    else:
        bsdf = dict(type="null")
        medium = dict(type="homogeneous", sigma_t=JADE["sigma_t"], albedo=dict(type="rgb", value=list(JADE["albedo"])))
    d = {"type": "scene", "sensor": _sensor(view, res, rgba),
         "dragon": dict(type="ply", filename=str(WEB_DATA / "dragon.ply"), face_normals=False, bsdf=bsdf, interior=medium)}
    if dragon_only:
        return mi.load_dict(d)
    if ground:
        half = GROUND["size"] / 2
        d["ground"] = dict(type="rectangle", to_world=T().rotate([1, 0, 0], -90).scale([half, half, 1]),
                           bsdf=dict(type="roughplastic", distribution="ggx", alpha=GROUND["roughness"] ** 2,  # three alpha = r^2
                                     diffuse_reflectance=dict(type="rgb", value=list(GROUND["albedo"]))))
    if env:
        d["env"] = dict(type="envmap", filename=str(WEB_DATA / "env" / f"{ENV['name']}_1k.hdr"), scale=ENV["scale"],
                        to_world=T().rotate([0, 1, 0], ENV_ROT_Y_DEG))
    if lamp_on:
        d["lamp"] = dict(type="sphere", center=lamp_position(*lamp).tolist(), radius=LAMP["sphere_radius"],
                         emitter=dict(type="area", radiance=dict(type="rgb", value=list(LAMP["radiance"]))))
    return mi.load_dict(d)


def _np(img):
    return np.array(img, dtype=np.float32)


def render(view: dict, lamp, res: int, spp: int, passes=("env", "lamp"), max_pass_samples: int = 2**27) -> dict:
    vol = mi.load_dict(dict(type="volpath", max_depth=MAX_DEPTH))
    per = max(1, max_pass_samples // (res * res))
    chunks = [min(per, spp - s) for s in range(0, spp, per)]
    # spp split into seeded passes, spp-weighted average: bounded work per call
    avg = lambda sc: sum(_np(mi.render(sc, integrator=vol, spp=n, seed=k)) * n for k, n in enumerate(chunks)) / spp
    out = {}
    if "env" in passes:
        out["env"] = avg(_scene(view, lamp, res, env=True, lamp_on=False))
    if "lamp" in passes:
        out["lamp"] = avg(_scene(view, lamp, res, env=False, lamp_on=True))
    if "both" in passes:
        out["both"] = avg(_scene(view, lamp, res, env=True, lamp_on=True, rgba=True))
    # AOVs at 1 spp so positions are never averaged across a silhouette
    aov = mi.load_dict(dict(type="aov", aovs="pos:position,nn:sh_normal,pi:prim_index,dd:depth"))
    a = _np(mi.render(_scene(view, lamp, res, env=False, lamp_on=False, dragon_only=True), integrator=aov, spp=1))
    full = _np(mi.render(_scene(view, lamp, res, env=False, lamp_on=True), integrator=aov, spp=1))
    pos, nrm, tri, depth = a[..., 0:3], a[..., 3:6], a[..., 6], a[..., 7]
    hit = depth > 0
    visible = hit & (np.linalg.norm(full[..., 0:3] - pos, axis=-1) < 1e-5)  # nothing in front of the dragon
    m = visible.copy()  # drop silhouette pixels: their radiance mixes dragon and background
    m[1:, :] &= visible[:-1, :]; m[:-1, :] &= visible[1:, :]
    m[:, 1:] &= visible[:, :-1]; m[:, :-1] &= visible[:, 1:]
    out.update(pos=pos, nrm=nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8),
               tri=np.where(hit, np.round(tri), -1).astype(np.int32), mask=m)
    if "spec" in passes:
        coat = avg(_scene(view, lamp, res, env=True, lamp_on=True, coat=True))
        out["spec"] = np.where(m[..., None], coat, 0.0).astype(np.float32)
    return out
