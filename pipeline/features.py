"""Network inputs. The order is mirrored verbatim by FEATURES_GLSL in web/src/glsl.js."""
import numpy as np

from pipeline.config import FLAME


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def features(pos, nrm, cam, flame_offset) -> np.ndarray:
    """pos(3) | nrm(3) | view(3) | flame_offset(3) | L(3) | dist(1) | dot(nrm, L)(1) -> (N, 17)"""
    pos, nrm = np.asarray(pos, np.float64), np.asarray(nrm, np.float64)
    flame = np.array(FLAME["base"]) + np.asarray(flame_offset, np.float64)
    off = np.broadcast_to(np.asarray(flame_offset, np.float64), pos.shape)
    view = _unit(np.asarray(cam, np.float64) - pos)
    to_l = flame - pos
    dist = np.linalg.norm(to_l, axis=-1, keepdims=True)
    L = to_l / np.maximum(dist, 1e-12)
    ndl = np.sum(nrm * L, axis=-1, keepdims=True)
    return np.concatenate([pos, nrm, view, off, L, dist, ndl], -1).astype(np.float32)


def chord_to_flame(pos, flame, radius, height) -> np.ndarray:
    """Length of segment pos->flame inside the cylinder r<radius, 0<y<height (cup ignored)."""
    P = np.asarray(pos, np.float64)
    D = np.asarray(flame, np.float64) - P
    t0, t1 = np.zeros(len(P)), np.ones(len(P))
    # radial: a t^2 + b t + c < 0
    a = D[:, 0] ** 2 + D[:, 2] ** 2
    b = 2 * (P[:, 0] * D[:, 0] + P[:, 2] * D[:, 2])
    c = P[:, 0] ** 2 + P[:, 2] ** 2 - radius ** 2
    disc = b * b - 4 * a * c
    vert = a < 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(np.maximum(disc, 0))
        r0, r1 = (-b - s) / (2 * a), (-b + s) / (2 * a)
    r0 = np.where(vert, np.where(c < 0, -np.inf, np.inf), np.where(disc > 0, r0, np.inf))
    r1 = np.where(vert, np.where(c < 0, np.inf, -np.inf), np.where(disc > 0, r1, -np.inf))
    t0, t1 = np.maximum(t0, r0), np.minimum(t1, r1)
    # y slab
    with np.errstate(divide="ignore", invalid="ignore"):
        y0, y1 = (0 - P[:, 1]) / D[:, 1], (height - P[:, 1]) / D[:, 1]
    flat = np.abs(D[:, 1]) < 1e-14
    inside_y = (P[:, 1] > 0) & (P[:, 1] < height)
    lo = np.where(flat, np.where(inside_y, -np.inf, np.inf), np.minimum(y0, y1))
    hi = np.where(flat, np.where(inside_y, np.inf, -np.inf), np.maximum(y0, y1))
    t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
    return np.maximum(t1 - t0, 0) * np.linalg.norm(D, axis=-1)
