"""Error vs the path-traced reference poses -> docs/results.md.
python -m pipeline.evaluate"""
import json
from pathlib import Path

import numpy as np
from pipeline.classic import classic_sss, load_classic_inputs
from pipeline.config import BACKLIT, LATENT_DIM, OUT, REF_POSES, WEB_DATA, WIDTHS
from pipeline.features import barycentric, interp_normal, raw_features
from pipeline.gen_data import dragon_mesh
from pipeline.scene import camera_origin, lamp_position
from pipeline.train import forward_numpy

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results.md"


def rel_rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)) / np.mean(y))


def block_mean(img, mask, b=8):
    """Average dragon pixels over b x b blocks (blocks at least half dragon). Removes per-pixel
    Monte Carlo noise of the reference so the error measures the method, not the grain."""
    n = img.shape[0] // b
    img, mask = img[: n * b, : n * b], mask[: n * b, : n * b]
    s = (img * mask[..., None]).reshape(n, b, n, b, 3).sum((1, 3))
    c = mask.reshape(n, b, n, b).sum((1, 3))
    ok = c >= b * b // 2
    return s[ok] / c[ok][:, None]


def main():
    V, _ = dragon_mesh()
    F, vert_th, sh = load_classic_inputs()
    classic = json.loads((WEB_DATA / "classic.json").read_text())
    nets = {w: (json.loads((WEB_DATA / f"weights_{w}.json").read_text()),
                np.frombuffer((WEB_DATA / f"latents_{w}.bin").read_bytes(), np.uint8).reshape(-1, LATENT_DIM)) for w in WIDTHS}
    Nq = np.frombuffer((WEB_DATA / "normals_net.bin").read_bytes(), np.int8).reshape(-1, 3)
    rows = []
    for i, pose in enumerate(REF_POSES):
        r = np.load(OUT / f"ref{i}.npz")
        msk = r["mask"]
        pos, nrm, tri = r["pos"][msk], r["nrm"][msk], r["tri"][msk]
        bary = barycentric(pos, V, F, tri)
        n = len(pos)
        cam = np.tile(camera_origin(pose), (n, 1))
        lamp = np.tile(lamp_position(pose["lamp_az_deg"], pose["lamp_el_deg"]), (n, 1))
        sss = r["both"][..., :3]  # null boundary: radiance on the dragon is the SSS term
        methods = {"classique": lambda: classic_sss(nrm, pos, cam, lamp, (bary * vert_th[F[tri]]).sum(1), sh, classic)}
        for w, (wt, q) in nets.items():
            methods[f"neuronal {w}"] = lambda wt=wt, q=q: forward_numpy(wt, q, F[tri], bary,
                                                                              raw_features(interp_normal(Nq, F, tri, bary), pos, cam, lamp))
        for name, f in methods.items():
            out6 = f()
            pred = np.zeros_like(sss)
            pred[msk] = out6[:, :3] + out6[:, 3:]
            rows.append((i, name, rel_rmse(block_mean(pred, msk), block_mean(sss, msk)), rel_rmse(pred[msk], sss[msk]),
                         rel_rmse((pred + r["spec"])[msk], (sss + r["spec"])[msk])))
    lines = ["# Résultats — dragon de jade", "",
             "Erreur RMSE relative (RMSE / moyenne de la référence) du SSS sur les pixels du dragon, contre les",
             "poses de référence path tracées (caméra + lampe). Poses 1 et 3 : contre-jour.", "",
             "- **SSS, blocs 8×8** : pixels moyennés par blocs avant comparaison (mesure principale : enlève le grain",
             "  du path tracing).",
             "- **SSS, pixel** et **image** (SSS + vernis) : pixel à pixel, bruit de la référence inclus.", "",
             "| pose | méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |", "|---|---|---|---|---|"]
    lines += [f"| {i}{' (contre-jour)' if i in BACKLIT else ''} | {n} | {a:.3f} | {b:.3f} | {c:.3f} |" for i, n, a, b, c in rows]
    lines += ["", "Moyennes :", "", "| méthode | SSS blocs, toutes poses | SSS blocs, contre-jour | image pixel |",
              "|---|---|---|---|"]
    for name in dict.fromkeys(n for _, n, *_ in rows):
        sel = [(i, a, c) for i, n, a, _, c in rows if n == name]
        lines.append(f"| {name} | {np.mean([a for _, a, _ in sel]):.3f} | "
                     f"{np.mean([a for i, a, _ in sel if i in BACKLIT]):.3f} | {np.mean([c for *_, c in sel]):.3f} |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
