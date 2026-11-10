# Neural shading en WebGL : un dragon de jade

Un petit réseau de neurones (2 couches de 16 neurones, 8 Ko de poids) remplace, dans un shader
WebGL2, le calcul de la lumière qui traverse le jade (subsurface scattering). Il est entraîné sur des
rendus path tracés Mitsuba 3, et tourne en GLSL ordinaire sur n'importe quel GPU, téléphone compris.

- **Démo** : https://wanadev.github.io/neural-shading-demo/ · **Labo** (AO et verre) : https://wanadev.github.io/neural-shading-demo/labo.html
- **Article** : *Neural shading en WebGL : ce que vaut un réseau de neurones dans un shader* *(lien à venir)*

Un rideau compare, sur le dragon de Stanford, l'approximation classique des jeux, le réseau et la
référence path tracée. La caméra et la lampe se déplacent librement.

## Résultats

Erreur relative du SSS par rapport à la référence path tracée (4 poses, dont 2 à contre-jour). Le
détail est dans [`docs/results.md`](docs/results.md).

| méthode | toutes poses | contre-jour |
|---|---|---|
| approximation classique (épaisseur par vertex + SH), ajustée sur les mêmes données | 1,04 | 1,69 |
| réseau 16 neurones | 0,39 | 0,58 |
| réseau 32 neurones | 0,38 | 0,57 |
| réseau 64 neurones | 0,42 | 0,68 |

La démo charge 3,0 Mo au démarrage (HDRI et maillage pour l'essentiel) et reste jouable sur un
Pixel 6a avec le réseau 16. Le bouton « Mesurer » de la page donne le coût sur votre appareil.

## Comment ça marche

```
Mitsuba 3 (path tracer, CPU)        PyTorch (CPU, ~5 min)              navigateur (three.js, WebGL2)
400 vues aléatoires          ──▶    MLP + 8 latents par sommet  ──▶    latents = attributs de sommet
lumière exacte en 2,7 M points      entraînés ensemble                 poids dans une texture RGBA32F
                                                                       MLP évalué dans le fragment shader
```

1. **Ground truth** : Mitsuba rend le dragon sous des caméras et des lampes tirées au hasard, en passes
   séparées (pièce, lampe), et garde pour chaque pixel la lumière, la position, la normale et le
   triangle touché.
2. **Entraînement** : chaque sommet porte 8 valeurs apprises (les *latents*) qui résument comment la
   lumière se comporte à cet endroit. Interpolées sur le triangle et combinées à la géométrie
   (normale, directions de la caméra et de la lampe), elles passent dans un MLP à activation ReLU²
   qui prédit 6 valeurs : la lumière venant de la pièce et celle venant de la lampe.
3. **Rendu** : le résultat est injecté comme lumière émise dans un `MeshPhysicalMaterial`. three.js
   garde les reflets, les ombres et le bloom.

## Structure

| dossier | contenu |
|---|---|
| `web/` | la démo statique (sans bundler, three.js 0.186.1 via importmap) |
| `web/src/` | `main.js` (démo), `labo*.js` (page Labo), `mlp.js` (réseau → texture + GLSL), `materials.js`, `selftest.js` |
| `web/data/` | maillage, latents, poids, références EXR, données du Labo |
| `pipeline/` | génération des données, entraînement et évaluation (Python 3.13, Mitsuba 3.9.1, PyTorch) |
| `spikes/` | deux tests rapides avec le même pipeline : AO apprise (`ao/`) et verre (`glass/`) |
| `docs/` | résultats, notes de construction, conception |

Le code et ses commentaires sont en anglais. L'interface et la documentation sont en français.

## Lancer la démo en local

```bash
python3 -m http.server -d web 8000
```

- http://localhost:8000/ : la démo
- http://localhost:8000/labo.html : AO et verre
- http://localhost:8000/selftest.html : auto-test GPU, compare le shader au même calcul en Python (doit afficher PASS)

Paramètres d'URL de la démo : `left` / `right` (`classic`, `neural`, `ref`), `split` (0 à 1),
`pose` (0 à 3), `lamp=az,el`. Exemple, contre-jour : `?pose=1&left=classic&right=neural`.
Labo : `effect` (`ao`, `glass`), `left`, `right`, `display=0` (AO seule), `aocheck` / `glasscheck`
(contrôle de parité avec les images du spike).

## Régénérer les données

Tout tourne sur CPU (machine de référence : 14 cœurs, sans GPU).

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
```

⚠️ Ne régénérez pas `dragon.ply` en cours de route : la décimation n'est pas reproductible au bit
près, et tous les fichiers par sommet (données, latents, épaisseur) en dépendent. Après un
`pipeline.dragon --force`, supprimez `pipeline/out/*_shards` avant de relancer `gen_data`.

Les spikes du Labo se relancent depuis la racine du dépôt :

```bash
.venv/bin/python -m spikes.ao.ao_spike       # -> web/data/labo/ao*
.venv/bin/python -m spikes.glass.glass_spike # -> web/data/labo/glass*
```

## Tests

```bash
.venv/bin/pytest pipeline/tests    # pipeline Python
(cd web && node --test)            # logique de la page
```

Le shader lui-même se vérifie dans le navigateur, sur le GPU visé, avec `selftest.html`.

## Publier

Le workflow `.github/workflows/pages.yml` publie `web/` sur GitHub Pages à chaque push sur `main`
(Settings → Pages → Source : GitHub Actions).

## Limites

- Le réseau est appris pour **ce** dragon, cette pièce et cette lampe (position libre, intensité fixe).
  Un autre objet demande de tout réentraîner.
- L'approche marche sur des effets qui varient doucement (jade, AO), pas sur le verre : voir la page Labo.

## Licence et crédits

Code sous licence MIT, voir [`LICENSE`](LICENSE).

- Stanford Dragon : [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/), Stanford Computer Graphics Laboratory.
- HDRI « Anniversary Lounge » : [Poly Haven](https://polyhaven.com/a/anniversary_lounge) (CC0).
- Références : [Mitsuba 3](https://www.mitsuba-renderer.org/). Rendu : [three.js](https://threejs.org/), [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh).
- Idée de départ : Charles Grassi, [Neural volumetric clouds](https://charlesgrassi.dev/blog/neural-volumetric-clouds/).
