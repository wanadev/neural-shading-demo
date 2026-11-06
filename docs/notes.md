# Notes pour l'article

Chiffres bruts relevés pendant la construction de la démo. Les écarts à la référence sont dans
`docs/results.md` (généré par `pipeline.evaluate`).

## Pipeline (WSL2, 14 cœurs, sans GPU)

| étape | durée |
|---|---|
| Données d'entraînement : 200 vues, 128², 1024 spp (Mitsuba `llvm_ad_rgb`) | 72 min (≈ 22 s/vue), 559 193 échantillons |
| Entraînement des 3 réseaux (PyTorch CPU) | 32 s au total |
| Ajustement de l'approximation classique | 66 s |
| 4 vues de référence, 1024², 2048 spp | 73 min (16 à 20 min/vue) |

## Poids téléchargés

| fichier | brut | gzip |
|---|---|---|
| `weights_8.json` | 6,1 Ko | |
| `weights_16.json` | 13,9 Ko | 6,5 Ko |
| `weights_32.json` | 38,0 Ko | |
| `candle.obj` (géométrie) | 1,7 Mo | 401 Ko |
| tout `web/data/` (références PNG incluses) | 2,8 Mo | |

Le réseau pèse moins qu'une vignette JPEG ; c'est le maillage qui coûte.

## Qualité (résumé de results.md)

Erreur relative du terme SSS contre la référence, moyennée par blocs 8×8 :
classique 1,15 · neural 8 : 0,30 · neural 16 : 0,31 · neural 32 : 0,26.
Le réseau est environ 4× plus proche de la référence ; la taille compte peu à ce niveau.

Erreur de validation à l'entraînement (pixel à pixel, cibles bruitées) : 0,139 / 0,141 / 0,135.
Les trois tailles font jeu égal : le bruit Monte Carlo des cibles (1024 spp) est le plafond.

## Coût GPU (bouton « Mesurer », ms par image 1024²)

| appareil | classique | neural 8 | neural 16 | neural 32 |
|---|---|---|---|---|
| PC (à remplir) | | | | |
| Pixel 6a (à remplir) | | | | |
| téléphone d'entrée de gamme (si emprunté) | | | | |

Auto-test GLSL (`selftest.html`) : PASS 64/64 en Chromium/SwiftShader (écart max 5,8e-6).
À relancer sur le Pixel 6a.

## Choix et limites à mentionner

- La cire est modélisée comme un volume diffusant à indice adapté + un vernis réfléchissant
  (GGX), pas comme un vrai diélectrique : avec une interface réfractive, le path tracer CPU ne
  convergeait pas (les rayons d'ombre depuis l'intérieur ne voient pas la flamme). three.js utilise
  exactement le même modèle, la comparaison reste juste.
- Le réseau est appris pour **cette** bougie : autre forme = nouvel entraînement.
- L'intensité de la flamme est multipliée après le réseau (le SSS est linéaire en intensité) ; seule
  la position de la flamme est une entrée.
- La table ne reçoit pas la lueur transmise par la cire dans three.js (elle est presque noire de
  toute façon).
- La référence garde du grain dans le bas de la bougie même après 16-20 min par image ; le réseau,
  lui, est lisse. Bon argument pour l'article.
- L'approximation classique a été ajustée sur les mêmes données que le réseau ; l'optimiseur a
  abandonné le terme de wrap lighting (a ≈ 0) et ne garde que la transmittance par épaisseur.
