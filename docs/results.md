# Résultats

Erreur RMSE relative (RMSE / moyenne de la référence) sur les pixels de cire, contre les vues
de référence path tracées (1024², 2048 spp), flamme au repos.

- **SSS, blocs 8×8** : pixels moyennés par blocs avant comparaison. Mesure principale : le grain
  du path tracing dans la partie sombre domine sinon l'erreur pixel à pixel.
- **SSS, pixel** et **image** (SSS + reflet) : pixel à pixel, bruit de la référence inclus.

| vue | méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |
|---|---|---|---|---|
| 0 | classique | 1.075 | 1.272 | 1.107 |
| 0 | neural 8 | 0.308 | 0.611 | 0.531 |
| 0 | neural 16 | 0.317 | 0.615 | 0.535 |
| 0 | neural 32 | 0.263 | 0.588 | 0.511 |
| 1 | classique | 1.380 | 1.774 | 1.361 |
| 1 | neural 8 | 0.259 | 1.119 | 0.859 |
| 1 | neural 16 | 0.238 | 1.114 | 0.855 |
| 1 | neural 32 | 0.215 | 1.109 | 0.851 |
| 2 | classique | 0.971 | 2.665 | 2.647 |
| 2 | neural 8 | 0.506 | 2.531 | 2.515 |
| 2 | neural 16 | 0.507 | 2.542 | 2.525 |
| 2 | neural 32 | 0.369 | 2.510 | 2.493 |
| 3 | classique | 1.152 | 1.212 | 0.869 |
| 3 | neural 8 | 0.142 | 0.372 | 0.267 |
| 3 | neural 16 | 0.190 | 0.392 | 0.281 |
| 3 | neural 32 | 0.191 | 0.392 | 0.281 |

Moyenne sur les vues :

| méthode | SSS, blocs 8×8 | SSS, pixel | image, pixel |
|---|---|---|---|
| classique | 1.145 | 1.730 | 1.496 |
| neural 8 | 0.304 | 1.158 | 1.043 |
| neural 16 | 0.313 | 1.166 | 1.049 |
| neural 32 | 0.260 | 1.150 | 1.034 |
