# Notes pour l'article — dragon de jade

Chiffres bruts relevés pendant la construction de la démo. Les écarts à la référence détaillés sont
dans `docs/results.md` (généré par `pipeline.evaluate`). La démo bougie précédente est sur la branche
`neural-candle`.

## Pipeline (WSL2, 14 cœurs, sans GPU)

| étape | durée |
|---|---|
| Dragon de Stanford : plus grand morceau du scan, bouchage, décimation 871 k → 80 k triangles (pymeshlab) | ~1 min |
| Données d'entraînement : 400 vues 192², 512 spp, passes pièce + lampe (Mitsuba `llvm_ad_rgb`) | ~2 h 40 (≈ 17–23 s/vue), 2,73 M échantillons |
| Entraînement des 3 réseaux + vecteurs par sommet (PyTorch CPU) | 5 min 11 s au total |
| Ajustement de l'approximation classique | 6 min |
| 4 poses de référence, 1024², 1024 spp | 12 à 16 min/pose (calcul en parallèle de l'entraînement) |

## Ce qu'on télécharge (gzip)

| fichier | gzip |
|---|---|
| `dragon.ply` (40 k sommets, 80 k triangles) | 1,34 Mo |
| `latents_32.bin` (8 octets par sommet) | 268 Ko |
| `weights_32.json` | 19 Ko (16 : 7,6 Ko · 64 : 56 Ko) |
| `thickness.bin` (approximation classique) | 38 Ko |
| **démarrage, réseau 32** | **1,67 Mo** |
| HDRI de la pièce (1k) | 1,6 Mo |
| chaque référence EXR (chargée seulement si « Path tracé ») | ~3,3 Mo |

Le réseau lui-même pèse moins qu'une icône ; ce sont les vecteurs par sommet (et surtout le maillage) qui comptent.

## Qualité (résumé de results.md)

Erreur relative du SSS contre la référence path tracée, pixels moyennés par blocs 8×8 :

| méthode | toutes poses | contre-jour (poses 1 et 3) |
|---|---|---|
| classique (épaisseur locale + SH, ajustée sur les mêmes données) | 1,04 | 1,69 |
| neural 16 | 0,39 | 0,58 |
| neural 32 | 0,38 | 0,57 |
| neural 64 | 0,42 | 0,68 |

Le réseau est 2,5 à 3× plus proche de la référence, et l'écart est le plus grand en contre-jour — là
où la technique classique décroche. La taille du réseau compte peu : ce sont les vecteurs par sommet
qui portent l'information de forme. Erreur de validation à l'entraînement : 0,217 / 0,205 / 0,187
(la quantification des vecteurs sur 8 bits ne coûte rien).

## Coût GPU (bouton « Mesurer », pose de référence 0)

| appareil | classique | neural 16 | neural 32 | neural 64 |
|---|---|---|---|---|
| PC (à remplir) | | | | |
| Pixel 6a (à remplir) | | | | |

Deux colonnes par cellule dans la page : shading seul (1024²) et image complète (avec halo).
Auto-test GPU (`selftest.html`) : PASS 32/32 pour 16, 32 et 64 en Chromium/SwiftShader (écart max
5,8e-4, contrôle négatif `?corrupt=1` → FAIL). À relancer sur le Pixel 6a.

## Ce qui a coincé (matière pour l'article)

- **Le scan brut n'est pas étanche** : 151 morceaux (débris du scanner) et des trous ; une décimation
  classique crée des arêtes non manifold. Il a fallu un décimateur qui préserve la topologie.
- **La décimation n'est pas reproductible au bit près** : la relancer réordonne les sommets sous
  tous les fichiers par sommet. Un incident l'a montré ; le script refuse désormais d'écraser le maillage.
- **Des « fireflies » dans les données** : 115 échantillons sur 2,7 M (jusqu'à 1 770 quand 99,99 %
  sont sous 7,7) ont suffi à ruiner une pose en contre-jour. Écrêtés avant l'entraînement.
- **Une sortie `exp()` non bornée** produisait des valeurs 60× trop fortes sur quelques entrées jamais
  vues ; bornée à 1,2× la cible max.
- **L'auto-test lui-même avait un bug invisible** tant que les vecteurs étaient presque constants :
  l'arrondi des sommets du rasteriseur sur des triangles de 0,6 px faussait l'interpolation (~2 %).

## Choix et limites à mentionner

- Le jade est un volume diffusant à indice adapté + un vernis réfléchissant (pas de réfraction
  interne), dans Mitsuba comme dans three.js : la comparaison reste juste.
- Le vernis côté web est celui de three.js (`MeshPhysicalMaterial`, environnement préfiltré, Fresnel de
  Schlick) ; la référence utilise l'intégration exacte.
- Pas de shadow map sur le dragon : ses ombres sur lui-même viennent du réseau. Quelques taches
  sombres restent visibles sur la tête à certaines poses.
- La lueur verte projetée sur le sol n'existe que dans la référence.
- Bug instructif trouvé en revue : la caméra d'ombre de la lampe (three.js, near = 0,5 m par défaut)
  coupait tout le dragon, qui ne projetait donc aucune ombre — on l'avait pris pour « un reflet un peu
  fort ». La comparaison avec la référence path tracée l'a rendu évident.
- Réseau appris pour **ce** dragon, cette pièce et cette lampe (position libre, intensité fixe).
- Crédits : Stanford Dragon — Stanford Computer Graphics Laboratory ; HDRI « anniversary_lounge » — Poly Haven (CC0).
