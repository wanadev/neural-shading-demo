# Bougie neuronale

Une bougie dont la lueur dans la cire (subsurface scattering) est calculée par un petit réseau de
neurones (MLP de 8 à 32 neurones), entraîné sur un rendu path tracé Mitsuba 3 et évalué dans un
fragment shader WebGL2 avec three.js. Sur la même page, un rideau compare le réseau à
l'approximation classique et à la référence path tracée.

- Démo : `web/` (page statique, sans bundler, three.js 0.186.1 via importmap)
- Pipeline : `pipeline/` (Python 3.13, Mitsuba 3.9.1, PyTorch, CPU uniquement)
- Conception : `docs/superpowers/specs/2026-11-05-neural-candle-design.md`
- Résultats : `docs/results.md`

## Lancer la démo en local

```bash
python3 -m http.server -d web 8000
# http://localhost:8000/            démo
# http://localhost:8000/selftest.html  auto-test GLSL (doit afficher PASS 64/64)
```

## Régénérer les données

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline.assets                       # géométrie + config.json
.venv/bin/python -m pipeline.gen_data                     # rendus d'entraînement (reprenable)
.venv/bin/python -m pipeline.train                        # weights_{8,16,32}.json + selftest.json
.venv/bin/python -m pipeline.classic                      # classic.json (baseline ajustée)
.venv/bin/python -m pipeline.render_refs                  # refs/view{0..3}.png
.venv/bin/python -m pipeline.evaluate                     # docs/results.md
.venv/bin/pytest pipeline/tests && (cd web && node --test)
```

Toutes les constantes partagées (géométrie, flamme, cire, caméra, exposition) sont dans
`pipeline/config.py`. Après une modification, relancer toute la chaîne.

Durées mesurées (WSL2, 14 cœurs, sans GPU) : voir `docs/results.md`.

## Publier

Le workflow `.github/workflows/pages.yml` publie `web/` sur GitHub Pages à chaque push sur `main`
(Settings → Pages → Source : GitHub Actions).
