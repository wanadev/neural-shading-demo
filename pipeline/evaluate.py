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


def main():
    methods = {"classique": None, **{f"neural {w}": json.loads((WEB_DATA / f"weights_{w}.json").read_text())
                                     for w in WIDTHS}}
    classic = json.loads((WEB_DATA / "classic.json").read_text())
    rows = []
    for i, view in enumerate(REF_VIEWS):
        r = np.load(OUT / f"ref{i}.npz")
        m = r["mask"]
        pos, nrm, cam = r["pos"][m], r["nrm"][m], camera_origin(view)
        spec, full = r["spec"][m], r["full"][m]
        sss = full - spec
        for name, w in methods.items():
            pred = (classic_sss(pos, nrm, cam, np.array(FLAME["base"]), classic) if w is None
                    else forward_numpy(w, features(pos, nrm, cam, np.zeros(3))))
            rows.append((i, name, rel_rmse(pred, sss), rel_rmse(pred + spec, full)))
    lines = ["# Résultats", "", "Erreur RMSE relative (RMSE / moyenne de la référence) sur les pixels de cire,",
             "vues de référence path tracées, flamme au repos.", "",
             "| vue | méthode | terme SSS seul | image (SSS + reflet) |", "|---|---|---|---|"]
    lines += [f"| {i} | {n} | {a:.3f} | {b:.3f} |" for i, n, a, b in rows]
    lines += ["", "Moyenne sur les vues :", "", "| méthode | terme SSS seul | image |", "|---|---|---|"]
    for n in methods:
        sel = [(a, b) for _, m_, a, b in rows if m_ == n]
        lines.append(f"| {n} | {np.mean([a for a, _ in sel]):.3f} | {np.mean([b for _, b in sel]):.3f} |")
    RESULTS.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
