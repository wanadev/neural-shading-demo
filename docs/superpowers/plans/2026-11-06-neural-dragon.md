# Dragon de jade neuronal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer la bougie par le dragon de Stanford en jade, éclairé par une HDRI + une lampe mobile, dont le SSS (ambiance + lampe) est calculé par un MLP à latents par sommet dans un shader WebGL2, avec un rendu « photo » (bloom, fond flou) et la même démo de comparaison.

**Architecture:** On réutilise la chaîne existante (Mitsuba CPU → npz → PyTorch → JSON/bin → GLSL généré → three.js) en remplaçant les parties propres à la bougie. Les données d'entraînement viennent de deux passes Mitsuba séparées (HDRI seule, lampe seule) sur un jade à interface nulle. Un auto-décodeur apprend ensemble un latent de 8 valeurs par sommet et un MLP 19→w→w→6. Côté web, les deux côtés du rideau sont rendus dans une cible HDR, puis bloom et tone mapping sont appliqués une seule fois à l'image composée.

**Tech Stack:** Python 3.13, mitsuba 3.9.1 (`llvm_ad_rgb`), torch CPU, numpy, trimesh, fast-simplification, pytest ; three.js 0.186.1 (importmap jsdelivr, addons `PLYLoader`, `HDRLoader`/`RGBELoader` selon le nom exporté en 0.186.1, `EffectComposer`, `UnrealBloomPass`, `OutputPass`) ; `node --test`.

**Spec:** `docs/superpowers/specs/2026-11-06-neural-dragon-design.md` (la spec de la bougie `2026-11-05-neural-candle-design.md` reste valable pour tout ce que la nouvelle ne change pas).

## Global Constraints

- WebGL2, `precision highp float;`, pas de bundler, three@0.186.1 épinglé, page statique servie depuis `web/` (chemins relatifs, aucun `../`).
- Source unique des constantes : `pipeline/config.py` → `web/data/config.json`.
- Unités : mètres, Y vers le haut, dragon posé sur le sol (y=0), hauteur ≈ 0.15 m.
- Le jade = volume diffusant à interface `null` + vernis GGX en passe séparée (comme la bougie). Les **mêmes** formules de vernis, d'éclairage et d'environnement côté Classique et Neural ; seul le terme SSS change.
- Pas de shadow map sur le dragon ; shadow map de la lampe uniquement pour le sol.
- Un jeu de latents **par largeur** (16/32/64, défaut 32), quantifié uint8, téléchargé à la demande.
- Budget téléchargement : maillage + données réseau de la largeur 32 < 2 Mo gzip.
- Cible : ≥ 30 fps sur Pixel 6a, réseau 32, bloom activé ; bloom en demi-résolution ; réseau par fragment à pleine résolution ; DPR ≤ 2.
- Budget calcul total 5–6 h ; jobs longs via `systemd-run --user`, écriture atomique des fichiers produits.
- Dragon : ne jamais le déformer ni l'animer. Crédit visible dans la page : « Stanford Dragon — Stanford Computer Graphics Laboratory » et « HDRI : Poly Haven (CC0) ».

## Review Focus

1. **Lampe glissée hors du domaine d'entraînement** (élévation sous le sol, au-dessus du max) → la lampe est bornée aux plages de `config.json`. Test : Task 8 `lamp.test.js` (`clampLamp`).
2. **Ordre des sommets désaligné** entre `dragon.ply`, `latents_*.bin` et `thickness.bin` → shading faux silencieux. Test : Task 1 `test_ply_roundtrip_keeps_vertex_order` + Task 5 `test_export_sizes_match_mesh` ; l'auto-test GPU (Task 9) utilise les vrais sommets.
3. **Changement de largeur avant la fin du téléchargement des latents** → on garde l'ancienne largeur affichée jusqu'à disponibilité, sans écran noir ni erreur. Test : Task 8 `policy.test.js` (`effectiveWidth`).
4. **« Path tracé » choisi avant que l'image HDR de référence soit chargée** (plusieurs Mo sur mobile) → le côté affiche « chargement… » et reste en Neural jusqu'au chargement. Test : Task 8 `policy.test.js` (`sideMode`).
5. **GPU sans cible de rendu flottante** (`EXT_color_buffer_half_float` absent) → repli sans bloom (rendu direct à l'écran avec tone mapping), jamais d'écran noir. Test : Task 8 `policy.test.js` (`pickPipeline`).

---

## Structure des fichiers

```
requirements.txt          + trimesh, fast-simplification
pipeline/
  config.py               réécrit : DRAGON, JADE, LAMP, ENV, GROUND, CAMERA, EXPOSURE, REF_POSES, WIDTHS, LATENT_DIM
  dragon.py               NOUVEAU : téléchargement, réparation, décimation, mise à l'échelle -> web/data/dragon.ply
  env.py                  NOUVEAU : HDRI Poly Haven + harmoniques sphériques -> web/data/env/*.hdr, env_sh.json
  thickness.py            NOUVEAU : épaisseur par sommet (lancer de rayons Mitsuba) -> web/data/thickness.bin
  scene.py                réécrit : scène dragon, passes env / lampe / spéculaire, AOV
  features.py             réécrit : 11 entrées brutes + interpolation barycentrique
  gen_data.py             adapté : vues + lampe aléatoires, écrit Y_env / Y_lamp
  train.py                réécrit : auto-décodeur latents + MLP 6 sorties, quantification, export
  classic.py              réécrit : épaisseur locale + SH, ajusté sur les données
  render_refs.py          adapté : poses caméra+lampe, sortie .hdr RGBA
  evaluate.py             adapté : somme des deux têtes contre la référence
  tonemap.py              inchangé
  assets.py               SUPPRIMÉ (bougie)
  tests/                  test_assets.py, test_features.py, test_classic.py remplacés ; autres adaptés
web/
  index.html              UI : + bascule caméra/lampe, Contre-jour, crédits
  src/main.js             réécrit autour du compositeur HDR
  src/materials.js        réécrit : jade (neural/classic), sol
  src/glsl.js             réécrit : FEATURES_GLSL (19), COAT_GLSL (lampe + env), CLASSIC_GLSL, SH_GLSL
  src/mlp.js              adapté : sorties multiples
  src/lamp.js             NOUVEAU (pur) : lampPosition, clampLamp, backlight
  src/policy.js           NOUVEAU (pur) : pickPipeline, sideMode, effectiveWidth
  src/refs.js             adapté : poses avec lampe
  src/flame.js            SUPPRIMÉ
  src/selftest.js         réécrit : toutes largeurs + interpolation des latents
  test/                   flame.test.js supprimé ; mlp/refs adaptés ; lamp/policy ajoutés
```

---

### Task 1: Dragon et environnement

**Files:**
- Create: `pipeline/dragon.py`, `pipeline/env.py`
- Modify: `pipeline/config.py` (réécrit), `requirements.txt`
- Delete: `pipeline/assets.py`, `pipeline/tests/test_assets.py`
- Test: `pipeline/tests/test_dragon.py`, `pipeline/tests/test_env.py`

**Interfaces:**
- Produces:
  - `config.py` : `WEB_DATA` (déplacé depuis assets.py), `DRAGON = dict(url="https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz", member="dragon_recon/dragon_vrip.ply", target_faces=80_000, height=0.15)`, `JADE = dict(ior=1.61, roughness=0.08, sigma_t=600.0, albedo=(0.90, 0.985, 0.93))`, `LAMP = dict(center=(0, 0.07, 0), radius=0.25, sphere_radius=0.008, radiance=(3000.0, 2700.0, 2300.0), elevation_deg=(-10.0, 70.0))`, `ENV = dict(name=<choisi>, scale=1.0)`, `GROUND = dict(size=4.0, albedo=(0.03, 0.03, 0.03), roughness=0.3)`, `CAMERA = dict(fov_y=30.0, target=(0, 0.07, 0), radius=(0.25, 0.6), elevation_deg=(-3.0, 60.0))`, `EXPOSURE`, `REF_POSES` (4 × `dict(az_deg, el_deg, radius, lamp_az_deg, lamp_el_deg)`, dont 2 en contre-jour), `WIDTHS = (16, 32, 64)`, `LATENT_DIM = 8`, `to_json()`. Toutes les valeurs de jade/lampe/exposition sont des points de départ réglés à la Task 2.
  - `dragon.py` : `load_dragon(ply_path) -> trimesh.Trimesh` ; `prepare(mesh, target_faces, height) -> (V float32[n,3], F int32[m,3], N float32[n,3])` (répare : fusion des sommets, bouchage des trous, faces orientées vers l'extérieur ; décime ; met à l'échelle `height` ; centre en x/z ; min y = 0) ; `write_ply(path, V, F, N)` binaire little-endian (`x y z nx ny nz` float32, faces `uchar` + `int32`) ; CLI `python -m pipeline.dragon` (télécharge dans `pipeline/out/` si absent, écrit `web/data/dragon.ply` + `config.json`).
  - `env.py` : `sh9_irradiance(hdr: float32[h,w,3]) -> float32[9,3]` (coefficients d'irradiance de Ramamoorthi–Hanrahan, convention de direction **de three.js** `equirectUv`, voir test) ; CLI télécharge `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{name}_1k.hdr` vers `web/data/env/`, écrit `web/data/env_sh.json`.
  - Convention de direction partagée (three.js `equirectUv`, texture `flipY`) : `u = atan2(d.z, d.x)/(2π) + 0.5`, `v = asin(d.y)/π + 0.5`, ligne 0 du fichier = haut (v = 1).

- [ ] **Step 1:** Ajouter `trimesh` et `fast-simplification` à `requirements.txt`, installer (PyPI uniquement).
- [ ] **Step 2: Write the failing tests:**

```python
# test_dragon.py (sur un maillage synthétique : icosphère trimesh percée d'un trou)
def test_prepare_is_closed_scaled_grounded():
    V, F, N = prepare(holed_sphere(), target_faces=500, height=0.15)
    m = trimesh.Trimesh(V, F, process=False)
    assert m.is_watertight and m.volume > 0          # fermé, orienté vers l'extérieur
    assert abs(V[:, 1].min()) < 1e-6 and abs(V[:, 1].max() - 0.15) < 1e-4
    assert abs(V[:, 0].mean()) < 0.01 and len(F) <= 550

def test_ply_roundtrip_keeps_vertex_order(tmp_path):   # Review Focus #2
    V, F, N = prepare(holed_sphere(), 500, 0.15)
    write_ply(tmp_path / "d.ply", V, F, N)
    m = trimesh.load(tmp_path / "d.ply", process=False)
    assert np.allclose(m.vertices, V) and (m.faces == F).all()

# test_env.py
def test_sh_constant_env():          # environnement uniforme L=1 -> irradiance E(n) = π pour toute normale
    sh = sh9_irradiance(np.ones((64, 128, 3), np.float32))
    assert np.allclose(eval_sh_irradiance(sh, np.array([[0, 1, 0], [1, 0, 0]])), np.pi, rtol=0.02)

def test_sh_direction_convention():  # texel lumineux en haut de l'image -> irradiance max vers +Y
    img = np.zeros((64, 128, 3), np.float32); img[0:4] = 50.0
    sh = sh9_irradiance(img)
    up, down = eval_sh_irradiance(sh, np.array([[0, 1, 0], [0, -1, 0]]))[:, 0]
    assert up > 5 * down
```

  (`eval_sh_irradiance(sh, normals) -> (N,3)` est exporté par `env.py`, reproduit en GLSL à la Task 7.)
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implémenter `config.py`, `dragon.py`, `env.py`.
- [ ] **Step 5:** Run → PASS. Lancer les deux CLI. Choisir la HDRI : une pièce intérieure plutôt sombre de Poly Haven (`api.polyhaven.com/assets?t=hdris&c=indoor`), noter son nom dans `ENV.name`. Vérifier : `dragon.ply` étanche (`trimesh.load(...).is_watertight`), et `gzip -c web/data/dragon.ply | wc -c` < 1,6 Mo (sinon baisser `target_faces`).
- [ ] **Step 6:** Commit `feat: dragon mesh and HDRI environment` (supprime `assets.py` et ses tests).

---

### Task 2: Scène Mitsuba du dragon

**Files:**
- Modify: `pipeline/scene.py` (réécrit)
- Test: `pipeline/tests/test_scene.py` (adapté)

**Interfaces:**
- Consumes: `config.*`, `web/data/dragon.ply`, `web/data/env/*.hdr`
- Produces:
  - `camera_origin(view: dict) -> np.ndarray(3)` (inchangé) ; `lamp_position(az_deg, el_deg) -> np.ndarray(3)` (même convention que `viewDir`, centrée sur `LAMP.center`, rayon `LAMP.radius`).
  - `render(view: dict, lamp: (az_deg, el_deg), res, spp, passes=("env", "lamp"), max_pass_samples=2**27) -> dict` avec, selon `passes` : `env (H,W,3)` = SSS dû à la HDRI seule, `lamp (H,W,3)` = SSS dû à la lampe seule, `both (H,W,4)` = image complète RGBA (alpha = géométrie touchée) avec HDRI + lampe, `spec (H,W,3)` = vernis seul sur les pixels du dragon ; et toujours `pos, nrm (H,W,3)`, `tri (H,W) int32` (indice de triangle, −1 hors dragon), `mask (H,W) bool`.
- Décisions :
  - Passe `env` : envmap (`ENV.scale`, rotation `ENV_TO_WORLD` fixée par le test d'orientation) + sol ; pas de lampe. Passe `lamp` : lampe (sphère émissive) + sol ; envmap absente. Dragon = `null` + `homogeneous(JADE)`, `volpath`, `max_depth=1024`. **Pas de passe spéculaire pour l'entraînement** (avec l'interface nulle, la radiance sur le dragon *est* le SSS).
  - `both` : `hdrfilm` `pixel_format="rgba"`, HDRI + lampe, dragon nul ; `spec` : même scène, dragon `roughdielectric(distribution="ggx", int_ior=JADE.ior, alpha=JADE.roughness)` sur absorbeur.
  - AOV sur une scène contenant **uniquement le dragon** : `pos:position, nn:sh_normal, pi:prim_index` ; `mask` = touché dans cette scène **et** position identique (< 1e-5) à l'AOV de la scène complète (rien ne cache le dragon), puis érosion d'un pixel des silhouettes (comme la bougie).

- [ ] **Step 1: Write the failing tests:**

```python
def test_env_orientation_matches_three():
    # texel lumineux à (u,v) -> la direction three.js correspondante doit voir ce texel dans Mitsuba
    d = three_equirect_dir(u=0.8, v=0.7)
    assert eval_env_direction(marker_hdr(u=0.8, v=0.7), d) > 10 * eval_env_direction(marker_hdr(u=0.8, v=0.7), -d)

def test_lamp_position_convention():
    p = lamp_position(90, 0)
    assert np.allclose(p, np.array(LAMP["center"]) + [LAMP["radius"], 0, 0], atol=1e-6)

def test_render_passes_smoke():
    r = render(REF_POSES[0], (REF_POSES[0]["lamp_az_deg"], REF_POSES[0]["lamp_el_deg"]), res=32, spp=8,
               passes=("env", "lamp", "both", "spec"))
    m = r["mask"]
    assert m.sum() > 30 and (r["tri"][m] >= 0).all()
    assert r["env"][m].mean() > 0 and r["lamp"][m].mean() > 0
    assert r["both"].shape == (32, 32, 4)
    assert (r["both"][..., 3][m] > 0.99).all()                 # dragon pixels are opaque in alpha
    assert np.allclose(np.linalg.norm(r["nrm"][m], axis=1), 1, atol=1e-3)
```

  (`three_equirect_dir`, `marker_hdr`, `eval_env_direction` sont des helpers de test : le dernier charge un `envmap` Mitsuba avec `ENV_TO_WORLD` et appelle `emitter.eval` pour une direction.)
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implémenter ; régler `ENV_TO_WORLD` jusqu'à ce que le test d'orientation passe.
- [ ] **Step 4:** Run → PASS. **Réglage visuel** : rendre `REF_POSES` 0 (face) et un contre-jour en 256², 256 spp (`both` + `spec`, tonemappé avec `tonemap.py`) dans `pipeline/out/preview_*.png` et **les regarder** : jade vert crédible, parties fines lumineuses en contre-jour, corps plus dense, ambiance de pièce visible. Ajuster `JADE`, `LAMP.radiance`, `ENV.scale`, `EXPOSURE` dans `config.py` uniquement.
- [ ] **Step 5:** Mesurer le temps d'une vue d'entraînement (192², 512 spp, passes env+lamp) et le noter dans le message de commit.
- [ ] **Step 6:** Commit `feat: mitsuba dragon scene with env/lamp passes`.

---

### Task 3: Épaisseur par sommet

**Files:**
- Create: `pipeline/thickness.py`
- Test: `pipeline/tests/test_thickness.py`

**Interfaces:**
- Produces: `vertex_thickness(V, F, N, rays=32, seed=0) -> float32[n]` : moyenne, sur `rays` directions tirées en cône (cosinus) autour de `-N`, de la distance au premier impact sur le maillage (rayons partant de `V - 1e-5 N`), via `scene.ray_intersect` Mitsuba ; CLI écrit `web/data/thickness.bin` (`uint8`, valeur / `thickness_max`) et ajoute `thickness_max` à `web/data/config.json`.

- [ ] **Step 1: Failing test:** sphère unité (rayon r = 0.05) : `vertex_thickness` ≈ la corde moyenne en cône cosinus autour du diamètre, donc entre `1.3·r` et `2·r` pour tous les sommets ; et un cylindre fin (r = 0.003) donne une épaisseur < 0.01 partout.
- [ ] **Step 2–4:** FAIL → implémenter → PASS ; lancer la CLI sur le dragon.
- [ ] **Step 5:** Commit `feat: per-vertex thickness for the classic baseline`.

---

### Task 4: Entrées et génération des données

**Files:**
- Modify: `pipeline/features.py` (réécrit), `pipeline/gen_data.py`
- Test: `pipeline/tests/test_features.py` (réécrit), `pipeline/tests/test_gen_data.py` (adapté)

**Interfaces:**
- Produces:
  - `raw_features(nrm (N,3), pos (N,3), cam (N,3), lamp (N,3)) -> float32 (N,11)`, ordre **figé** : `nrm(3) | view=normalize(cam-pos)(3) | L=normalize(lamp-pos)(3) | dist=|lamp-pos|(1) | dot(nrm,L)(1)`. L'entrée complète du MLP est `latent(8) | raw(11)` = 19.
  - `barycentric(pos (N,3), V, F, tri (N,)) -> float32 (N,3)` (coordonnées barycentriques du point dans son triangle, bornées et renormalisées).
  - `gen_data` : CLI `--views 400 --res 192 --spp 512`, vue aléatoire (az, el, rayon dans `CAMERA`) + lampe aléatoire (az ∈ [0,360), el ∈ `LAMP.elevation_deg`), seed par vue. Échantillon par pixel du masque : `tri, bary(3), nrm, pos, cam, lamp, Y_env(3), Y_lamp(3), view_id`. Shards atomiques (`save_atomic`, `shard_dir` existants).

- [ ] **Step 1: Failing tests:** `test_raw_feature_layout` (valeurs connues comme pour la bougie, sur 11 colonnes) ; `test_barycentric_recovers_vertices_and_centroid` (sommet k → e_k, centroïde → (1/3,1/3,1/3), somme = 1).
- [ ] **Step 2–4:** FAIL → implémenter → PASS ; mini-run `--views 3 --res 32 --spp 16` sans NaN, `Y_env` et `Y_lamp` > 0 en moyenne.
- [ ] **Step 5:** Commit `feat: dragon features and training data generation`.
- [ ] **Step 6:** Lancer le vrai run via `systemd-run --user` (budget ≈ 2 h 30 ; si la Task 2 a mesuré > 25 s/vue, réduire `--views` à 300 plutôt que la résolution).

---

### Task 5: Auto-décodeur latents + MLP, quantification, export

**Files:**
- Modify: `pipeline/train.py` (réécrit)
- Test: `pipeline/tests/test_train.py` (réécrit)

**Interfaces:**
- Produces:
  - `class NeuralSSS(nn.Module)(n_vertices, width, latent_dim=8, depth=2)` : `latents = nn.Parameter(N(0, 0.01))` de forme `(n_vertices, 8)` ; `forward(tri_verts (B,3) long, bary (B,3), raw_std (B,11)) -> (B,6)` = `exp(MLP(cat(Σ bary_k·latents[tri_verts_k], raw_std)))`, MLP `Linear → relu²` × depth puis `Linear(width, 6)`, sorties 0–2 = env, 3–5 = lampe.
  - `train(npz, F, n_vertices, width, epochs, batch=8192) -> (model, mean, std, val_err)` : perte relative `((p−y)²/(p.detach()+0.01)²)` sur les 6 sorties, split par `view_id`, tirage avec remise.
  - `quantize(latents) -> (q uint8 (n,8), lo float[8], hi float[8])` par dimension ; puis **affinage du MLP seul** quelques epochs sur les latents déquantifiés gelés.
  - `export(model, mean, std, lo, hi, json_path, bin_path)` : JSON `{"width", "latent_dim": 8, "latent_lo", "latent_hi", "in_mean"[11], "in_std"[11], "layers": [{"W": [out][in], "b"}...]}` (couche 0 : in = 19) + `latents_{w}.bin` (uint8, n×8, ordre des sommets de `dragon.ply`).
  - `forward_numpy(weights, q_latents, tri_verts, bary, raw) -> (N,6)` (déquantifie, interpole, standardise, MLP).
  - CLI `python -m pipeline.train` → `web/data/weights_{16,32,64}.json`, `latents_{16,32,64}.bin`, `selftest.json` = par largeur, 32 échantillons de validation `{tri_verts[3], bary[3], nrm, pos, cam, lamp, expected[6]}`.

- [ ] **Step 1: Failing tests:**

```python
def test_forward_numpy_matches_torch(tmp_path):      # latents quantifiés, 10 échantillons, rtol 1e-4
def test_export_sizes_match_mesh(tmp_path):          # Review Focus #2
    # bin de n_vertices*8 octets, JSON W0 de forme (w, 19), W2 de forme (6, w), lo/hi de longueur 8
def test_overfits_toy(tmp_path):                     # 50 sommets, cible = fonction lisse des latents vrais + raw ; val_err < 0.05
def test_quantization_roundtrip():                   # |deq(q(x)) - x| <= (hi-lo)/255/2 + 1e-7
```

- [ ] **Step 2–4:** FAIL → implémenter → PASS.
- [ ] **Step 5:** Sur les vraies données : entraîner les 3 largeurs, noter `val_err` avant/après quantification dans le message de commit ; vérifier `gzip -c dragon.ply latents_32.bin weights_32.json thickness.bin | wc -c` < 2 Mo.
- [ ] **Step 6:** Commit `feat: per-vertex latent auto-decoder and export`.

---

### Task 6: Approximation classique ajustée

**Files:**
- Modify: `pipeline/classic.py` (réécrit)
- Test: `pipeline/tests/test_classic.py` (réécrit)

**Interfaces:**
- Consumes: `web/data/thickness.bin` (lu et déquantifié avec `thickness_max`, interpolé par `bary` sur `tri`), `env_sh.json`, `eval_sh_irradiance`, le npz de la Task 4.
- Produces: `classic_sss(nrm, pos, cam, lamp, th, sh, p) -> (N,6)` (env RGB, lampe RGB), avec `th` = épaisseur interpolée (barycentrique), `L, d, V` comme Task 4 :

```
lamp = (1/d²) · ( tint_d · max((n·L + w)/(1+w), 0)
                + tint_t · exp(−k·th) · max(dot(V, −normalize(L + n·dist)), 0)^p )
env  = tint_e · E(n) + tint_eb · E(−n) · exp(−k·th)        E = eval_sh_irradiance(sh, ·)
```
  paramètres `p = {w, k, p, dist, tint_d[3], tint_t[3], tint_e[3], tint_eb[3]}` (positifs via softplus, `p ≥ 1`). `fit_classic(npz) -> (params, val_err)` ; CLI → `web/data/classic.json`.

- [ ] **Step 1: Failing tests:** sortie ≥ 0 ; le terme lampe décroît quand `th` augmente (tint_d = 0) ; `fit_classic` retrouve des paramètres synthétiques (val_err < 0.05, comme pour la bougie).
- [ ] **Step 2–4:** FAIL → implémenter → PASS ; ajuster sur les vraies données, noter `val_err`.
- [ ] **Step 5:** Commit `feat: fitted local-thickness baseline`.

---

### Task 7: Shaders : jade neural / classique, vernis, sol

**Files:**
- Modify: `web/src/glsl.js` (réécrit), `web/src/mlp.js`, `web/src/materials.js` (réécrit)
- Delete: `web/src/flame.js`, `web/test/flame.test.js`
- Test: `web/test/mlp.test.js` (adapté)

**Interfaces:**
- Produces:
  - `mlpGlsl(w)` : génère `void neuralSSS(float x[19], out vec3 envSSS, out vec3 lampSSS)` ; la standardisation ne s'applique qu'aux indices 8–18 (`in_mean/in_std` de longueur 11) ; sortie `exp`.
  - `glsl.js` : `FEATURES_GLSL` = `void features(vec4 latA, vec4 latB, vec3 nrm, vec3 pos, vec3 cam, vec3 lamp, out float x[19])` (déquantification `lo + (hi−lo)·lat` via uniforms `uLatLo[8]`, `uLatHi[8]`, puis l'ordre de la Task 4) ; `SH_GLSL` = `vec3 shIrradiance(vec3 n)` (uniform `uSH[9]`, mêmes constantes que `eval_sh_irradiance`) ; `CLASSIC_GLSL` = formule de la Task 6 mot pour mot ; `COAT_GLSL` = GGX lampe (point, intensité `I = radiance·π·r²`) + environnement préfiltré via `textureCubeUV` du PMREM de three.js (rugosité `JADE.roughness`), Fresnel de Schlick (F0 de `JADE.ior`).
  - `makeJadeMaterial({mode: "neural"|"classic", weights, classic, cfg, shared}) -> ShaderMaterial` : attributs `latA`, `latB` (`Uint8BufferAttribute`, `normalized: true`, itemSize 4) et `thick` (`Uint8`, normalisé) ; sortie linéaire HDR `envSSS + lampSSS + coat` (pas de `tonemapping_fragment` : le tone mapping est fait par l'`OutputPass`).
  - `makeGround(cfg, shared) -> Mesh` : `MeshStandardMaterial` (albedo/rugosité de `GROUND`, `envMap` PMREM), `receiveShadow`.
  - `makeLamp(cfg) -> { light: SpotLight (castShadow, cible = centre du dragon, angle couvrant l'ombre au sol), bulb: Mesh émissif }` ; le dragon `castShadow = true` mais son matériau n'échantillonne pas la shadow map.

- [ ] **Step 1: Failing tests (mlp.test.js) :** pour `weights_32.json` : `W0[608]` (19×32), `W2[192]` (32×6), signature `void neuralSSS(float x[19], out vec3 envSSS, out vec3 lampSSS)`, `IN_MEAN[11]` ; `glslFloat` inchangé.
- [ ] **Step 2–4:** FAIL → implémenter → PASS (`cd web && node --test`).
- [ ] **Step 5:** Commit `feat(web): jade neural/classic shaders, coat, ground`.

---

### Task 8: Page : compositeur HDR, lampe, rideau, politiques

**Files:**
- Create: `web/src/lamp.js`, `web/src/policy.js`
- Modify: `web/src/main.js` (réécrit), `web/src/refs.js`, `web/index.html`, `web/src/bench.js`
- Test: `web/test/lamp.test.js`, `web/test/policy.test.js`, `web/test/refs.test.js` (adapté)

**Interfaces:**
- Produces:
  - `lamp.js` : `lampPosition({az_deg, el_deg}, cfg) -> [x,y,z]` (= Python `lamp_position`) ; `clampLamp({az_deg, el_deg}, cfg) -> {az_deg (mod 360), el_deg (borné à LAMP.elevation_deg)}` ; `backlight(camAzDeg) -> {az_deg: camAz+180 mod 360, el_deg: 20}`.
  - `policy.js` : `pickPipeline({halfFloatRT: bool}) -> "hdr"|"ldr"` ; `sideMode(requested, refReady) -> requested === "ref" && !refReady ? "neural" : requested` ; `effectiveWidth(requested, loadedWidths, current) -> loaded ? requested : current`.
  - `refs.js` : `nearestRefView` accepte les `REF_POSES` (champs caméra inchangés) ; `leaveRef`, `withPose`, `orbitLimits`, `viewDir` inchangés.
  - `main.js` :
    - Rendu `hdr` : `EffectComposer` sur cible `HalfFloatType` ; une passe « rideau » rend chaque côté avec `setScissor` (Classique / Neural / Path tracé) ; côté Path tracé = fond HDRI flou puis quad de la référence `.hdr` (RGBA, alpha = géométrie) ; puis `UnrealBloomPass` (résolution moitié) puis `OutputPass` (ACES Filmic, `EXPOSURE`). Rendu `ldr` (repli) : rendu direct, tone mapping du renderer, pas de bloom.
    - Fond : `scene.background = PMREM(HDRI)` avec `backgroundBlurriness ≈ 0.4` ; `scene.environment` pour le sol. **Décision du plan** : le « léger flou de profondeur » de la spec est réalisé par ce fond flou (le seul plan lointain de la scène), pas par un `BokehPass` qui exigerait une profondeur pour la référence Mitsuba et coûterait cher sur mobile.
    - Bascule « Déplacer : caméra | lampe » ; en mode lampe, OrbitControls désactivés et le glisser modifie az/el de la lampe via `clampLamp` ; bouton « Contre-jour » → `backlight`. Uniform `uLamp` partagé, `SpotLight` et ampoule suivent.
    - Largeur 16/32/64 : charge `weights_{w}.json` + `latents_{w}.bin` à la demande, `effectiveWidth` jusqu'à disponibilité ; défaut 32.
    - « Path tracé » : charge `refs/view{i}.hdr` à la demande (`sideMode` tant que non chargé, libellé « chargement… ») ; cale caméra **et lampe** sur `REF_POSES[i]` ; tout mouvement de caméra **ou de lampe** repasse en Neural (`leaveRef`).
    - « Mesurer » à pose fixe (`withPose`, `REF_POSES[0]`) : (a) shading seul 1024² par technique et largeur, (b) image complète avec post-traitement.
    - Crédits Stanford + Poly Haven dans `index.html`.

- [ ] **Step 1: Failing tests:**

```js
// lamp.test.js
test("lampPosition matches python convention", () => { /* az=90, el=0 -> center + [radius,0,0] */ });
test("clampLamp keeps the lamp in the training domain", () => {   // Review Focus #1
  assert.equal(clampLamp({ az_deg: 370, el_deg: 95 }, cfg).el_deg, cfg.lamp.elevation_deg[1]);
  assert.equal(clampLamp({ az_deg: -30, el_deg: -40 }, cfg).el_deg, cfg.lamp.elevation_deg[0]);
  assert.equal(clampLamp({ az_deg: 370, el_deg: 0 }, cfg).az_deg, 10);
});
test("backlight puts the lamp opposite the camera", () => { assert.equal(backlight(30).az_deg, 210); });
// policy.test.js — Review Focus #3, #4, #5
test("ldr fallback without half-float targets", () => { assert.equal(pickPipeline({ halfFloatRT: false }), "ldr"); });
test("ref side stays neural until its image is loaded", () => { assert.equal(sideMode("ref", false), "neural"); assert.equal(sideMode("ref", true), "ref"); });
test("width switch waits for its latents", () => { assert.equal(effectiveWidth(64, new Set([32]), 32), 32); assert.equal(effectiveWidth(64, new Set([32, 64]), 32), 64); });
```

- [ ] **Step 2–4:** FAIL → implémenter → PASS.
- [ ] **Step 5: Vérifs headless** (Chromium Playwright + SwiftShader, comme pour la bougie) : capture `?left=classic&right=neural` et une capture en contre-jour, **les regarder** (jade crédible, fond flou, halo sur l'ampoule, parties fines lumineuses côté Neural) ; aucune erreur console.
- [ ] **Step 6:** Commit `feat(web): HDR compositor, movable lamp, lazy widths and refs`.

---

### Task 9: Auto-test GPU (toutes largeurs + interpolation des latents)

**Files:**
- Modify: `web/src/selftest.js` (réécrit), `web/selftest.html`

**Interfaces:**
- Consumes: `selftest.json` (Task 5), `FEATURES_GLSL`, `mlpGlsl`, `latents_{w}.bin`.
- Produit la page : pour chaque largeur, chaque échantillon i est rendu comme **un petit triangle** en projection orthographique dont les 3 sommets portent les latents uint8 des 3 vrais sommets du dragon (`tri_verts`) et sont placés pour que le centre du pixel i ait exactement les barycentriques `bary` :

```
c = centre du pixel i (NDC) ; d0 = (a, 0), d1 = (0, a), d2 = −(b0·d0 + b1·d1) / b2   (a ≈ 0.6 px)
P_k = c + d_k  ->  Σ b_k P_k = c
```
  (échantillons retenus à l'export avec `b2 ≥ 0.2` pour éviter un triangle dégénéré ; nrm/pos/cam/lamp en uniforms par échantillon, un draw par échantillon). Lecture float, comparaison des 6 sorties à `expected` (tolérance relative 2e-3, la quantification est déjà dans `expected`). Affiche `PASS n/N` par largeur, `FAIL` avec le pire écart, `UNSUPPORTED` ou `ERREUR — …` comme avant.

- [ ] **Step 1:** Adapter `train.py` pour n'exporter que des échantillons `b2 ≥ 0.2` (test Python : `all(s["bary"][2] >= 0.2 for s in selftest)` sur une mini-exportation).
- [ ] **Step 2:** Implémenter la page ; Run headless (`--dump-dom`) → Expected `PASS 32/32` pour 16, 32 et 64.
- [ ] **Step 3:** Vérifier qu'une corruption volontaire (inverser deux latents dans le bin chargé, en console) donne `FAIL`, puis annuler.
- [ ] **Step 4:** Commit `feat(web): GPU selftest covering latents and all widths`.

---

### Task 10: Références, évaluation, notes

**Files:**
- Modify: `pipeline/render_refs.py`, `pipeline/evaluate.py`, `docs/notes.md`, `README.md`

**Interfaces:**
- Produces : `render_refs` → `web/data/refs/view{i}.hdr` (RGBA HDR : `both` + `spec` sur le masque, alpha conservée ; écrit via `mi.Bitmap(...).write(.hdr)` ; si RGBE ne garde pas l'alpha, écrire `view{i}_alpha.png` à côté et l'utiliser comme masque), `pipeline/out/ref{i}.npz` ; défaut `--res 1024 --spp 1024` ; poses = `REF_POSES`. `evaluate` : prédiction = env + lampe (somme des deux têtes) contre le SSS de référence (`both` sur le masque sans le vernis), métrique principale par blocs 8×8 + colonnes pixel, méthodes classique / neural 16/32/64.

- [ ] **Step 1:** Adapter, lancer `render_refs` via `systemd-run --user` (≈ 75 min).
- [ ] **Step 2:** `evaluate` → `docs/results.md` ; Expected : neural 32 meilleur que classique en blocs 8×8, **en particulier sur les 2 poses de contre-jour** (sinon revenir à la Task 5 : epochs, `LATENT_DIM`).
- [ ] **Step 3:** Vérif d'alignement headless `?left=neural&right=ref` sur une pose de contre-jour : silhouette, sol et ampoule raccordés.
- [ ] **Step 4:** `docs/notes.md` refait pour le dragon (durées, tailles gzip, tableau « Mesurer » à remplir PC / Pixel 6a, limites : lueur verte au sol, Fresnel de Schlick vs exact, environnement préfiltré vs exact) ; README (nouvelles commandes, crédits).
- [ ] **Step 5:** Commit `feat: dragon references, evaluation and notes`.
