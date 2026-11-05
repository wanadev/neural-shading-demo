# Bougie neuronale : neural shading en WebGL — conception

Date : 2026-11-05

## Intention

Démo web servant de base à un article pour le blog Wanadev : un retour d'expérience
personnel sur le neural shading appliqué à WebGL. Le fil rouge : « est-ce que ça vaut le
coup chez nous, sur du matériel ordinaire, téléphones compris ? ».

Inspiration (à citer, pas à copier) : https://charlesgrassi.dev/blog/neural-volumetric-clouds/
(un MLP minuscule remplace le multiple scattering d'un nuage, entraîné sur un path tracer).
Notre différence : angle web / mobile, source de lumière interne et mobile (flamme qui
vacille), compromis taille de réseau manipulable par le lecteur.

L'article sera écrit après, par l'auteur, une fois la démo fonctionnelle. Hors périmètre ici.

## Objectifs

- Une bougie de cire photoréaliste éclairée de l'intérieur par sa flamme qui vacille,
  rendue en temps réel dans three.js, fluide sur desktop **et** sur téléphone.
- La lueur de subsurface scattering dans la cire est calculée par un petit MLP entraîné
  sur un rendu path tracé (Mitsuba 3).
- Comparaison visuelle claire entre : approximation classique, rendu neuronal, référence
  path tracée.
- Chiffres de coût mesurés dans le navigateur, affichés en direct.

## Non-objectifs

- Généraliser à d'autres formes : le réseau est appris pour cette bougie précise (assumé,
  à mentionner dans l'article).
- Tensor cores / cooperative vectors / WebGPU : WebGL2 en code scalaire, c'est le propos.
- Rédaction de l'article.
- L'éclairage global d'une pièce (idée retenue pour un éventuel second article).

## Anti-objectifs (échec même si « ça marche »)

- Démo qui ne tourne bien que sur une grosse carte graphique.
- Démo jolie sous un seul angle.
- Réseau beaucoup plus cher que l'approximation pour un gain invisible.
- Ton « plaquette NVIDIA ».
- Structure ou présentation qui ressemble à une copie de l'article de Grassi
  (pas de tableau de scores par angle de soleil, etc.).

## Contraintes

- Stack décidée par l'auteur : **Mitsuba 3** (référence path tracée, v3.9.x),
  **PyTorch** (entraînement), **three.js** (démo).
- Machine : WSL2, pas de GPU NVIDIA visible, Python 3.13, Node 24. Rendu Mitsuba et
  entraînement sur CPU (variante `llvm_*`). Budget : quelques heures de rendu acceptable.
- Publication : page statique sur **GitHub Pages**, autonome (pas intégrée au blog),
  liée depuis l'article. Aucun serveur.
- WebGL2 uniquement côté rendu (le plus petit dénominateur commun mobile).

## Décisions de l'auteur

- Sujet : une bougie.
- Mise en page : **rideau glissant** avant/après (option A), une seule vue, pensée mobile
  d'abord.
- Caméra : **orbite libre** autour de la bougie en rendu neuronal. Si le path tracé ne peut
  pas s'afficher pour la vue courante, ce n'est pas grave : on reste en neuronal.
- Démo publiée sur GitHub Pages.

## Choix faits par Claude (validés en bloc par l'auteur, à vérifier à la lecture)

### Page

- Rideau vertical déplaçable au doigt/à la souris. Chaque côté a un menu :
  Classique / Neural / Path tracé. Défaut : gauche Classique, droite Neural.
- Barre de chiffres en bas : temps GPU des deux côtés + nom/type d'appareil si détectable.
- Réglage taille du réseau : 8 / 16 / 32 neurones (défaut 16), bascule à chaud.
- Interrupteur flamme : vacille / fixe.
- « Path tracé » sélectionné : la caméra se cale (animation courte) sur le point de vue
  précalculé le plus proche (~4 vues), la flamme se fige à la pose de référence. Dès que
  l'utilisateur fait tourner la caméra, ce côté repasse en Neural.

### Éclairage (choix délégué par l'auteur : « ce qui est le mieux pour la démo »)

- **La flamme est la seule source de lumière.** Bougie posée sur une surface sombre dans
  une pièce noire. La surface reçoit la lumière de la flamme (analytique, identique dans
  les modes three.js). Pas d'environment map ni de lumière d'ambiance, pour que la lueur de
  la cire reste le sujet et que le réseau n'ait qu'une source à apprendre.

### Côté « Path tracé »

- Affiche l'image Mitsuba **complète** (pas de composition), rendue avec exactement la même
  géométrie, la même caméra (FOV, position) et le même tone mapping que three.js, pour que le
  rideau aligne les deux au pixel près.
- L'écart à la référence est mesuré deux fois : sur l'image entière et sur le terme SSS seul
  (AOV/rendu séparé).

### Ce que calcule le réseau

- Seulement la radiance sortante due au subsurface scattering dans la cire.
  Reflets spéculaires, flamme (émissif/billboard), mèche, support et décor restent
  analytiques/classiques et **identiques** dans les trois modes, pour une comparaison honnête.
- Entrées par fragment (repère local de la bougie) : position surface, normale, direction
  de vue, position de la flamme (offset de vacillement) ; éventuellement l'intensité de la
  flamme. Encodage (ex. quelques fréquences de Fourier) laissé au builder.
- Sortie : RGB linéaire, non négative.
- Réseau propre à cette géométrie.

### Approximation classique (côté « Classique »)

- La technique translucide habituelle de three.js : wrap lighting + carte d'épaisseur
  (type `MeshTranslucentShader` / transmission par épaisseur), réglée honnêtement au mieux
  pour ressembler à la référence. Pas d'homme de paille.

### Chaîne de fabrication

1. **Scène Mitsuba** : bougie (cylindre légèrement irrégulier, cuvette de cire fondue,
   mèche), cire = diélectrique rugueux + milieu homogène à fort albédo, flamme = petite
   source émissive. Paramètres de cire choisis pour un rendu crédible (valeurs publiées de
   paraffine si disponibles, sinon réglage à l'œil, à documenter).
2. **Génération de données** : quelques centaines de rendus à points de vue et poses de
   flamme aléatoires, avec AOV position + normale par pixel → échantillons
   (entrées, radiance SSS). Isoler le terme SSS (ex. rendu séparé du spéculaire, ou
   soustraction) : méthode au choix du builder, à documenter.
3. **Rendus de référence** pour l'affichage : ~4 vues fixes, flamme figée, haute qualité.
4. **Entraînement** PyTorch : trois réseaux (8, 16, 32 neurones par couche cachée,
   nombre de couches au choix du builder, petit). Fonction de perte et activation au choix
   (attention aux artefacts d'activation et aux biais de transformation de cible relevés
   par Grassi).
5. **Export** : poids en JSON (quelques Ko), plus mean/scale de normalisation.
6. **three.js** : le MLP est écrit à la main en GLSL (WebGL2), poids en uniforms.

### Mesures

- Barre en direct : temps GPU via `EXT_disjoint_timer_query_webgl2` quand dispo, sinon
  temps de frame. Indicatif seulement (absent sur la plupart des mobiles et sur iOS,
  plafonné par la vsync, ne sépare pas les deux côtés du rideau).
- **Bouton « Mesurer »** (mode benchmark) : rend une technique à la fois en plein écran,
  la passe de shading répétée / rendue hors écran pour sortir du plafond vsync, sur
  quelques secondes ; affiche un temps moyen par technique et par taille de réseau.
  C'est la source des chiffres de l'article.
- Appareils : le PC de l'auteur et un **Pixel 6a** (seul téléphone disponible ; pas de
  téléphone d'entrée de gamme, à signaler dans l'article ou à emprunter).
- Cible : **60 fps sur le Pixel 6a** avec le réseau à 16 neurones, MLP évalué par fragment,
  sans rendu en résolution réduite. DPR plafonné à 2 pour tous les modes (pratique courante
  three.js, identique pour chaque technique donc comparaison juste). Le réseau 32 peut
  ne pas tenir 60 fps : c'est un résultat, pas un échec.
- Écart à la référence calculé hors ligne sur les ~4 vues de référence (une métrique
  simple type erreur relative / RMSE), utile pour l'article.

## Laissé au builder

- Structure du dépôt, outillage front (bundler ou non), style visuel de la page.
- Détails de la scène Mitsuba, nombre d'échantillons, résolution des rendus.
- Architecture exacte du MLP au-delà de la largeur (8/16/32), encodage des entrées,
  hyperparamètres.
- Modèle 3D de la bougie (procédural ou fichier), tant que Mitsuba et three.js
  utilisent exactement la même géométrie.

## Livrables

- Scripts Python reproductibles : scène + génération de données + rendus de référence +
  entraînement + export.
- Page three.js statique prête pour GitHub Pages.
- Une courte note de résultats (chiffres, captures, impressions techniques) pour nourrir
  l'article.
