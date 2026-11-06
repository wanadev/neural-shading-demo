"""Error vs the path-traced references -> docs/results.md.
python -m pipeline.evaluate"""
import json
from pathlib import Path

import numpy as np

from pipeline.assets import WEB_DATA
from pipeline.classic import classic_sss
from pipeline.config import FLAME, REF_VIEWS, WIDTHS
from pipeline.features import features
from pipeline.render_refs import OUT
from pipeline.scene import camera_origin
from pipeline.train import forward_numpy

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results.md"


def rel_rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)) / np.mean(y))


def block_mean(img, mask, b=8):
    """Average wax pixels over b x b blocks (blocks at least half wax). Removes per-pixel
    Monte Carlo noise of the reference so the error measures the method, not the grain."""
    n = img.shape[0] // b
    img, mask = img[: n * b, : n * b], mask[: n * b, : n * b]
    s = (img * mask[..., None]).reshape(n, b, n, b, 3).sum((1, 3))
    c = mask.reshape(n, b, n, b).sum((1, 3))
    ok = c >= b * b // 2
    return s[ok] / c[ok][:, None]


def main():
    methods = {"classique": None, **{f"neural {w}": json.loads((WEB_DATA / f"weights_{w}.json").read_text())
                                     for w in WIDTHS}}
    classic = json.loads((WEB_DATA / "classic.json").read_text())
    rows = []
    for i, view in enumerate(REF_VIEWS):
        r = np.load(OUT / f"ref{i}.npz")
        m = r["mask"]
        pos, nrm, cam = r["pos"][m], r["nrm"][m], camera_origin(view)
        sss = r["full"] - r["spec"]
        for name, w in methods.items():
            pred = np.zeros_like(sss)
            pred[m] = (classic_sss(pos, nrm, cam, np.array(FLAME["base"]), classic) if w is None
                       else forward_numpy(w, features(pos, nrm, cam, np.zeros(3))))
            rows.append((i, name, rel_rmse(block_mean(pred, m), block_mean(sss, m)),
                         rel_rmse(pred[m], sss[m]),
                         rel_rmse((pred + r["spec"])[m], r["full"][m])))
    lines = ["# Résultats", "",
             "Erreur RMSE relative (RMSE / moyenne de la référence) sur les pixels de cire, contre les vues",
             "de référence path tracées (1024², 2048 spp), flamme au repos.", "",
             "- **SSS, blocs 8×8** : pixels moyennés par blocs avant comparaison. Mesure principale : le grain",
             "  du path tracing dans la partie sombre domine sinon l'erreur pixel à pixel.",
             "- **SSS, pixel** et **image** (SSS + reflet) : pixel à pixel, bruit de la référence inclus.", "",
             "| vue | méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {n} | {a:.3f} | {b:.3f} | {c:.3f} |" for i, n, a, b, c in rows]
    lines += ["", "Moyenne sur les vues :", "", "| méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |",
              "|---|---|---|---|"]
    for n in methods:
        sel = np.array([(a, b, c) for _, m_, a, b, c in rows if m_ == n]).mean(0)
        lines.append(f"| {n} | {sel[0]:.3f} | {sel[1]:.3f} | {sel[2]:.3f} |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
