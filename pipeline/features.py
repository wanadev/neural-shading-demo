"""Network inputs. Full MLP input = latent(8) | raw(11); the raw order is mirrored verbatim by
FEATURES_GLSL in web/src/glsl.js."""
import numpy as np


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def raw_features(nrm, pos, cam, lamp) -> np.ndarray:
    """nrm(3) | view = normalize(cam - pos)(3) | L = normalize(lamp - pos)(3) | dist(1) | dot(nrm, L)(1)"""
    nrm, pos = np.asarray(nrm, np.float64), np.asarray(pos, np.float64)
    view = unit(np.asarray(cam, np.float64) - pos)
    to_l = np.asarray(lamp, np.float64) - pos
    dist = np.linalg.norm(to_l, axis=-1, keepdims=True)
    L = to_l / np.maximum(dist, 1e-12)
    return np.concatenate([nrm, view, L, dist, np.sum(nrm * L, -1, keepdims=True)], -1).astype(np.float32)


def mesh_edges(F) -> np.ndarray:
    """Unique undirected edges (E, 2) of a triangle mesh."""
    F = np.asarray(F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def smooth_normals(F, N, iters) -> np.ndarray:
    """Vertex normals averaged with their neighbours `iters` times (a blur of radius ~ edge * sqrt(iters)),
    as int8 (x 127), exactly what the web reads (web/data/normals_net.bin, a normalized Int8 attribute).
    Light diffuses a few millimetres inside jade, so the scan's small bumps should not reach the network."""
    E = mesh_edges(F)
    n = np.asarray(N, np.float64)
    for _ in range(iters):
        acc = n.copy()
        np.add.at(acc, E[:, 0], n[E[:, 1]])
        np.add.at(acc, E[:, 1], n[E[:, 0]])
        n = unit(acc)
    return np.round(unit(n) * 127).astype(np.int8)


def interp_normal(Nq, F, tri, bary) -> np.ndarray:
    """The network's normal at a surface point: int8 vertex normals interpolated, then normalized (as in GLSL)."""
    n = (np.asarray(bary, np.float64)[..., None] * (np.asarray(Nq)[np.asarray(F)[tri]] / 127.0)).sum(1)
    return unit(n).astype(np.float32)


def barycentric(pos, V, F, tri) -> np.ndarray:
    """Barycentric coordinates of each point in its triangle, clipped to [0, 1] and renormalised."""
    a, b, c = (np.asarray(V, np.float64)[F[tri, k]] for k in range(3))
    p = np.asarray(pos, np.float64)
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    w1, w2 = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, 1)
    return (bary / np.maximum(bary.sum(-1, keepdims=True), 1e-12)).astype(np.float32)
