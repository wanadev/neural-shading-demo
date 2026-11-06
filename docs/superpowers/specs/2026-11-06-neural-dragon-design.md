# Dragon de jade neuronal — conception

Date : 2026-11-06. Remplace la démo bougie (`2026-11-05-neural-candle-design.md`), qui reste dans
l'historique git (branche `neural-candle`) et dont tout le pipeline est réutilisé.

## Intention

Même article (blog Wanadev, retour d'expérience perso « le neural shading en WebGL, ça vaut le
coup ? », angle web/mobile), mais une démo qui **impressionne visuellement** et qui paraît **proche
d'une photo**. Constat de l'auteur sur la bougie : la lumière dans la cire était juste, mais la
flamme et l'objet faisaient « image de synthèse ». Leçon retenue : le réalisme vient d'un vrai
objet scanné, d'un vrai éclairage d'ambiance et d'un rendu « appareil photo », et ce qui
impressionne à l'écran doit être ce que calcule le réseau.

## Objectifs

- Le **dragon de Stanford en jade**, éclairé par une vraie pièce (HDRI) + **une lampe que le
  lecteur déplace**, notamment derrière le dragon : les parties fines (crêtes, cornes, griffes)
  s'illuminent par transparence. C'est le moment « waouh » et c'est le réseau qui le calcule.
- Rendu « photo de studio » : surface sombre légèrement brillante, halo (bloom) sur les zones
  très lumineuses, léger flou de profondeur sur l'arrière-plan, tone mapping soigné.
- Même démo qu'avant par ailleurs : rideau Classique / Neural / Path tracé, caméra libre,
  réglage de taille de réseau, bouton « Mesurer », auto-test GLSL, page statique GitHub Pages,
  fluide sur Pixel 6a.
- Montrer que l'approximation classique (basée sur l'épaisseur) se trompe visiblement dans les
  parties fines et à contre-jour, alors que le réseau colle à la référence.

## Non-objectifs

- Flamme, feu, animation de l'objet (et interdit par Stanford : pas de déformation/animation du dragon).
- Généraliser à d'autres objets : le réseau est appris pour ce dragon.
- WebGPU / tensor cores.

## Anti-objectifs

- Un rendu qui fait « image de synthèse » (plastique vert, fond noir vide, pas d'ambiance).
- Un effet de transparence invisible ou réservé à une pose précise.
- Une démo qui ne tient pas sur Pixel 6a.

## Contraintes

- Stack inchangée : Mitsuba 3.9.1 (CPU, `llvm_ad_rgb`), PyTorch CPU, three.js 0.186.1 en
  importmap, sans bundler, WebGL2, GitHub Pages.
- Machine : WSL2, 14 cœurs, pas de GPU NVIDIA. Budget calcul accepté par l'auteur : **5 à 6 h au
  total, références comprises** (lancement le soir possible).
- **Cible mobile (validée par l'auteur)** : ≥ 30 fps sur Pixel 6a avec le réseau 32 et les effets
  photo activés. Bloom et flou de profondeur en demi-résolution ; réseau évalué par fragment à
  pleine résolution ; DPR plafonné à 2. Le réseau 64 qui ne tient pas = résultat, pas bug.
  Tâches longues lancées via `systemd-run --user` (les jobs d'arrière-plan meurent avec la session).
- **Licence** : dragon de Stanford — l'auteur juge l'usage dans un article acceptable. Obligatoire :
  crédit « Stanford Computer Graphics Laboratory » dans la page et l'article ; ne pas déformer ni
  animer le dragon (rotation de caméra et éclairage OK). HDRI : Poly Haven (CC0).
- Poids réseau + données par sommet raisonnables pour un téléchargement mobile (cible < 2 Mo gzip
  pour maillage + données réseau).

## Décisions de l'auteur

- Sujet : un objet classique de démo → le dragon de Stanford.
- Réalisme photo comme critère principal.
- Tout le reste du périmètre de la démo bougie est conservé (rideau, mesures, mobile, Pages).

## Choix faits par Claude (validés en bloc par l'auteur)

### Objet et matière
- Dragon de Stanford (`dragon_recon`, résolution max), mis à l'échelle d'une figurine de ~15 cm,
  posé sur le sol. **Décimé** (~100 k triangles, ajustable) pour le web ; le même maillage décimé
  sert à Mitsuba et à three.js. Maillage fermé exigé (le jade est un volume) : contrôle d'étanchéité,
  bouchage des trous si besoin.
- Jade : même modèle que la bougie — volume diffusant à indice adapté (BSDF `null`) + vernis
  réfléchissant GGX rendu à part (passe spéculaire). Coefficients choisis pour que les parties
  de quelques mm soient très translucides et le corps plus opaque ; teinte verte. Valeurs dans `config.py`.

### Éclairage
- **Environnement** : une HDRI Poly Haven CC0 d'intérieur, plutôt sombre, pour que la lampe domine.
  Fixe (ni rotation ni changement).
- **Lampe** : petite sphère émissive, visible (ampoule), déplacée sur une sphère autour du dragon
  (azimut libre, élévation bornée, distance fixe). Intensité fixe.
- La lumière étant additive, le jade reçoit deux termes indépendants :
  `SSS = SSS_env(point, vue) + SSS_lampe(point, vue, position lampe)`.

### Ce que calcule le réseau
- **Un seul MLP, 6 sorties** : SSS dû à l'environnement (RGB) + SSS dû à la lampe (RGB).
- **Latents par sommet** (approche des « neural materials » NVIDIA, appliquée aux sommets faute
  d'UV sur le dragon) : chaque sommet porte un petit vecteur appris (8 valeurs), interpolé par
  le GPU comme un attribut. Il encode ce que la géométrie fait à la lumière localement
  (épaisseur, parties fines), ce qu'un petit MLP ne peut pas apprendre à partir de la seule
  position sur une forme aussi complexe. Latents et MLP sont entraînés ensemble (auto-décodeur).
- Entrées : latent interpolé (8), normale (3), direction de vue (3), direction et distance de la
  lampe + n·L (5). Ordre figé, reproduit à l'identique en GLSL.
- **Un jeu de latents par taille de réseau** (16/32/64), chacun entraîné avec son MLP : la
  comparaison des tailles reste honnête. Seul le jeu de la taille affichée est téléchargé
  (chargement à la demande, défaut 32 au démarrage) pour tenir le budget mobile.
- **Pas de shadow map sur le dragon** : l'auto-occultation de la lampe (crêtes cachées par le corps)
  est laissée au réseau via les latents, ce qui garde intact « c'est le réseau qui le calcule ».
  Si le résultat fuit visiblement (lumière là où elle devrait être bloquée), repli documenté :
  latents plus larges (16) avant toute autre béquille. La shadow map ne sert qu'au sol.
- Latents quantifiés sur 8 bits à l'export (attribut `Uint8` normalisé) ; la perte de qualité est
  mesurée.
- Tailles proposées au lecteur : **16 / 32 / 64** neurones (défaut 32), à ajuster après mesure.

### Approximation classique (baseline honnête)
- Technique « épaisseur locale » des jeux (type DICE/Frostbite) : épaisseur par sommet précalculée
  (lancer de rayons vers l'intérieur), wrap lighting + translucidité à contre-jour pour la lampe,
  irradiance de l'environnement en harmoniques sphériques pour le terme d'ambiance. Paramètres
  **ajustés sur les mêmes données** que le réseau.

### Scène three.js
- Fond et reflets : la même HDRI (PMREM), affichée floue en arrière-plan.
- Sol : surface sombre légèrement brillante, ombre de la lampe (shadow map). Écart connu et
  documenté : la lueur verte transmise par le dragon sur le sol n'est pas reproduite.
- Spéculaire du vernis : GGX, environnement préfiltré + lampe ; identique dans les modes Classique
  et Neural (le côté Path tracé montre l'image Mitsuba exacte).
- Post-traitement appliqué **une fois sur l'image composée** (les deux côtés du rideau, y compris
  la référence) : bloom, tone mapping, léger flou de profondeur. Les références sont donc stockées
  en HDR (RGBE `.hdr`, chargées à la demande).

### Interaction
- Caméra : orbite libre (bornée au domaine d'entraînement).
- Lampe : bascule « Déplacer : caméra | lampe » ; en mode lampe, le glisser déplace la lampe
  autour du dragon. Raccourci « Contre-jour » qui place la lampe derrière le dragon par rapport à
  la caméra.
- « Path tracé » : ~4 poses de référence (caméra + lampe) ; choisir ce mode cale caméra et lampe
  sur la pose la plus proche ; tout mouvement repasse en Neural (même logique que la bougie).

### Données et mesures
- Données d'entraînement : vues aléatoires (caméra + lampe), passes séparées environnement /
  lampe / spéculaire, AOV position, normale, triangle touché (pour l'interpolation des latents).
- Mêmes mesures que la bougie : écart à la référence par blocs 8×8, « Mesurer » à pose fixe
  (shading seul, puis image complète avec post-traitement), auto-test GLSL (incluant
  l'interpolation des latents et toutes les tailles de réseau).

## Laissé au builder

- Choix précis de la HDRI, des coefficients du jade, de l'exposition (validation visuelle).
- Niveau de décimation, nombre de vues, spp, résolution (dans le budget 5-6 h).
- Architecture exacte du MLP au-delà des largeurs, hyperparamètres d'entraînement.
- Outil de décimation / réparation de maillage (petite dépendance Python acceptable).
- Implémentation du bloom / flou (addons three.js ou shader maison).

## Livrables

- Pipeline Python reproductible adapté au dragon ; page three.js mise à jour ; `docs/results.md`
  et `docs/notes.md` refaits pour le dragon ; crédit Stanford + Poly Haven dans la page.
