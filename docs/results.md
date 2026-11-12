# Résultats — dragon de jade

Erreur RMSE relative (RMSE / moyenne de la référence) du SSS sur les pixels du dragon, contre les
poses de référence path tracées (caméra + lampe). Poses 1 et 3 : contre-jour.

- **SSS, blocs 8×8** : pixels moyennés par blocs avant comparaison (mesure principale : enlève le grain
  du path tracing).
- **SSS, pixel** et **image** (SSS + vernis) : pixel à pixel, bruit de la référence inclus.

| pose | méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |
|---|---|---|---|---|
| 0 | classique | 0.353 | 0.369 | 0.329 |
| 0 | neuronal 16 | 0.207 | 0.228 | 0.204 |
| 0 | neuronal 32 | 0.225 | 0.242 | 0.216 |
| 0 | neuronal 64 | 0.161 | 0.182 | 0.162 |
| 1 (contre-jour) | classique | 1.507 | 1.581 | 1.283 |
| 1 (contre-jour) | neuronal 16 | 0.430 | 0.497 | 0.403 |
| 1 (contre-jour) | neuronal 32 | 0.559 | 0.635 | 0.515 |
| 1 (contre-jour) | neuronal 64 | 0.592 | 0.653 | 0.530 |
| 2 | classique | 0.398 | 0.414 | 0.362 |
| 2 | neuronal 16 | 0.181 | 0.205 | 0.179 |
| 2 | neuronal 32 | 0.148 | 0.173 | 0.151 |
| 2 | neuronal 64 | 0.152 | 0.176 | 0.154 |
| 3 (contre-jour) | classique | 1.880 | 1.955 | 1.587 |
| 3 (contre-jour) | neuronal 16 | 0.572 | 0.652 | 0.529 |
| 3 (contre-jour) | neuronal 32 | 0.564 | 0.637 | 0.517 |
| 3 (contre-jour) | neuronal 64 | 0.776 | 0.908 | 0.737 |

Moyennes :

| méthode | SSS blocs, toutes poses | SSS blocs, contre-jour | image pixel |
|---|---|---|---|
| classique | 1.035 | 1.694 | 0.890 |
| neuronal 16 | 0.347 | 0.501 | 0.329 |
| neuronal 32 | 0.374 | 0.561 | 0.350 |
| neuronal 64 | 0.420 | 0.684 | 0.396 |
