# Résultats — dragon de jade

Erreur RMSE relative (RMSE / moyenne de la référence) du SSS sur les pixels du dragon, contre les
poses de référence path tracées (caméra + lampe). Poses 1 et 3 : contre-jour.

- **SSS, blocs 8×8** : pixels moyennés par blocs avant comparaison (mesure principale : enlève le grain
  du path tracing).
- **SSS, pixel** et **image** (SSS + vernis) : pixel à pixel, bruit de la référence inclus.

| pose | méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |
|---|---|---|---|---|
| 0 | classique | 0.353 | 0.369 | 0.329 |
| 0 | neural 16 | 0.214 | 0.252 | 0.225 |
| 0 | neural 32 | 0.220 | 0.249 | 0.222 |
| 0 | neural 64 | 0.170 | 0.198 | 0.176 |
| 1 (contre-jour) | classique | 1.507 | 1.581 | 1.283 |
| 1 (contre-jour) | neural 16 | 0.502 | 0.677 | 0.549 |
| 1 (contre-jour) | neural 32 | 0.619 | 1.117 | 0.907 |
| 1 (contre-jour) | neural 64 | 0.757 | 0.984 | 0.799 |
| 2 | classique | 0.398 | 0.414 | 0.362 |
| 2 | neural 16 | 0.199 | 0.232 | 0.203 |
| 2 | neural 32 | 0.163 | 0.196 | 0.172 |
| 2 | neural 64 | 0.152 | 0.186 | 0.163 |
| 3 (contre-jour) | classique | 1.880 | 1.955 | 1.587 |
| 3 (contre-jour) | neural 16 | 0.661 | 0.770 | 0.625 |
| 3 (contre-jour) | neural 32 | 0.513 | 0.671 | 0.545 |
| 3 (contre-jour) | neural 64 | 0.597 | 0.851 | 0.691 |

Moyennes :

| méthode | SSS blocs, toutes poses | SSS blocs, contre-jour | image pixel |
|---|---|---|---|
| classique | 1.035 | 1.694 | 0.890 |
| neural 16 | 0.394 | 0.582 | 0.401 |
| neural 32 | 0.379 | 0.566 | 0.461 |
| neural 64 | 0.419 | 0.677 | 0.457 |
