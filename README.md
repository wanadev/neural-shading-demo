# Dragon de jade neuronal

Le dragon de Stanford en jade, éclairé par une vraie pièce (HDRI) et une lampe qu'on déplace. La
lumière qui traverse le jade est calculée par un petit réseau de neurones (16 à 64 neurones) et
un vecteur appris par sommet, entraînés sur un rendu path tracé Mitsuba 3 et évalués dans un
shader WebGL2 avec three.js. Un rideau compare le réseau à l'approximation classique des jeux et
à la référence path tracée.

- Démo : `web/` (page statique, sans bundler, three.js 0.186.1 via importmap)
- Pipeline : `pipeline/` (Python 3.13, Mitsuba 3.9.1, PyTorch, CPU uniquement)
- Conception : `docs/superpowers/specs/2026-11-06-neural-dragon-design.md`
- Résultats : `docs/results.md` · notes pour l'article : `docs/notes.md`
- La démo bougie précédente : branche `neural-candle`

## Lancer la démo en local

```bash
python3 -m http.server -d web 8000
# http://localhost:8000/                              démo
# http://localhost:8000/?pose=1&left=classic&right=neural   contre-jour, classique | neural
# http://localhost:8000/selftest.html                 auto-test GPU (doit afficher PASS)
```

Paramètres d'URL : `left` / `right` (`classic`, `neural`, `ref`), `split` (0–1), `pose` (0–3),
`lamp=az,el`.

## Régénérer les données

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline.dragon        # une seule fois : web/data/dragon.ply (--force pour refaire)
.venv/bin/python -m pipeline.config        # web/data/config.json
.venv/bin/python -m pipeline.env           # HDRI + harmoniques sphériques
.venv/bin/python -m pipeline.thickness     # épaisseur par sommet (approximation classique)
.venv/bin/python -m pipeline.gen_data      # ~2 h 40, reprenable
.venv/bin/python -m pipeline.train         # ~5 min : weights_*.json, latents_*.bin, selftest.json
.venv/bin/python -m pipeline.classic       # ~6 min : classic.json
.venv/bin/python -m pipeline.render_refs   # ~1 h : refs/view{0..3}.exr
.venv/bin/python -m pipeline.evaluate      # docs/results.md
.venv/bin/pytest pipeline/tests && (cd web && node --test)
```

⚠️ `dragon.ply` ne doit pas être régénéré en cours de route : la décimation n'est pas reproductible
au bit près et tous les fichiers par sommet (données, vecteurs, épaisseur) en dépendent. Les tâches
longues se lancent de préférence avec `systemd-run --user …` pour survivre à la fermeture du terminal.

## Publier

Le workflow `.github/workflows/pages.yml` publie `web/` sur GitHub Pages à chaque push sur `main`
(Settings → Pages → Source : GitHub Actions).

## Crédits

Stanford Dragon — Stanford Computer Graphics Laboratory. HDRI « anniversary_lounge » — Poly Haven (CC0).
