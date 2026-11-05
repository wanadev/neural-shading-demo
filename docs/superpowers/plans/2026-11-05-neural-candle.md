# Bougie neuronale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Démo three.js statique (GitHub Pages) d'une bougie dont la lueur de cire (SSS) est calculée par un petit MLP entraîné sur Mitsuba 3, comparée par rideau à une approximation classique et à la référence path tracée, avec mesures de coût mobile.

**Architecture:** Un pipeline Python (`pipeline/`) génère la géométrie partagée, rend des données d'entraînement Mitsuba (radiance + AOV position/normale), entraîne 3 MLP (8/16/32) et une approximation classique ajustée sur les mêmes données, rend 4 vues de référence et écrit tout dans `web/data/`. La page (`web/`, sans bundler, three.js via importmap) évalue le MLP dans un fragment shader WebGL2 à partir des poids JSON.

**Tech Stack:** Python 3.13, mitsuba==3.9.1 (variante `llvm_rgb`, repli `scalar_rgb`), torch (CPU), numpy, pytest ; three.js (version exacte épinglée dans l'importmap, `npm view three version` au moment de l'install) depuis cdn.jsdelivr.net ; tests JS avec `node --test` (Node 24).

**Spec:** `docs/superpowers/specs/2026-11-05-neural-candle-design.md`

## Global Constraints

- WebGL2 uniquement, `precision highp float;` dans tous les shaders. Pas de WebGPU, pas de bundler, pas de dépendance npm à l'exécution.
- Aucun serveur : `web/` doit fonctionner servi statiquement (`python -m http.server -d web`).
- Géométrie, caméra (FOV vertical 35°, axe Y, Y vers le haut), unités (mètres) et tone mapping (ACES Filmic de three.js, même exposition) **identiques** entre Mitsuba et three.js. Source unique : `pipeline/config.py` → `web/data/config.json`.
- La flamme est la seule source de lumière. Spéculaire, flamme (sprite), mèche et table sont identiques dans les modes Classique et Neural ; seul le terme SSS de la cire change.
- Réseau évalué par fragment, sans résolution réduite ; DPR plafonné à `min(devicePixelRatio, 2)` pour tous les modes.
- Cible : 60 fps sur Pixel 6a avec le réseau 16. Le 32 qui ne tient pas = résultat, pas bug.
- Le réseau est appris pour cette bougie uniquement.
- `pipeline/out/` (données lourdes) est ignoré par git ; `web/data/` est commité.

## Review Focus

1. **Précision mobile** : un GPU qui dégrade `highp` ou n'a pas `EXT_color_buffer_float` → l'auto-test doit l'annoncer clairement (pas de faux PASS). Test : Task 7 `selftest` vérifie l'extension et affiche `UNSUPPORTED`.
2. **Vacillement hors domaine d'entraînement** : `flameOffset(t)` doit rester dans les amplitudes d'entraînement pour tout `t`. Test : Task 7 `flame.test.js`.
3. **Caméra hors domaine** : élévation/distance hors des plages d'entraînement (sous la table, trop près) → limites OrbitControls tirées de `config.json`. Test : Task 8 `refs.test.js` (`orbitLimits`).
4. **Extension timer absente** (Pixel 6a, iOS) → HUD affiche le temps de frame, jamais `NaN`/crash. Test : Task 9 `hud.test.js`.
5. **Rotation / redimensionnement du téléphone** → canvas carré recalculé, image de référence toujours alignée. Vérif manuelle Task 8 step dédiée.

---

## Structure des fichiers

```
.gitignore                 # .venv/ pipeline/out/ .superpowers/ __pycache__/
requirements.txt           # mitsuba==3.9.1 torch numpy pytest
pipeline/
  __init__.py
  config.py                # toutes les constantes partagées + to_json()
  assets.py                # maillages bougie/mèche, write_obj, écrit web/data/{candle,wick}.obj + config.json
  scene.py                 # scène Mitsuba, render() -> radiance full/spec + AOV
  features.py              # features() 17 entrées, chord_to_flame()
  gen_data.py              # CLI : rendus aléatoires -> pipeline/out/train.npz
  train.py                 # MLP, entraînement, export JSON, forward_numpy, selftest.json
  classic.py               # approximation classique + fit -> web/data/classic.json
  tonemap.py               # port Python de l'ACES Filmic three.js
  render_refs.py           # 4 vues de référence -> web/data/refs/*.png + refs.json
  evaluate.py              # écarts -> docs/results.md
  tests/test_*.py
web/
  package.json             # {"type":"module"} uniquement pour node --test
  index.html               # importmap three, canvas, UI
  selftest.html            # auto-test de parité GLSL
  src/main.js              # scène, orbite, rideau, boucle de rendu
  src/materials.js         # makeWaxMaterial, makeTableMaterial
  src/glsl.js              # chaînes GLSL : features, chord, ggx
  src/mlp.js               # mlpGlsl(weights) (pur)
  src/flame.js             # flameOffset(t, cfg) (pur)
  src/refs.js              # nearestRefView, orbitLimits (purs)
  src/hud.js               # formatHud (pur)
  src/bench.js             # gpuTimer, bench
  src/selftest.js
  test/*.test.js
  data/                    # généré par le pipeline, commité
docs/results.md            # généré par evaluate.py
```

---

### Task 1: Dépôt, configuration partagée, géométrie

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pipeline/__init__.py`, `pipeline/config.py`, `pipeline/assets.py`
- Test: `pipeline/tests/test_assets.py`

**Interfaces:**
- Produces:
  - `pipeline.config` : `CANDLE`, `FLAME`, `WAX`, `TABLE`, `CAMERA`, `EXPOSURE`, `REF_VIEWS`, `WIDTHS`, `to_json() -> dict`
  - `candle_mesh(cfg=CANDLE) -> (V: float32[n,3], F: int32[m,3], N: float32[n,3])`
  - `wick_mesh(cfg=CANDLE, flame=FLAME) -> (V, F, N)`
  - `write_obj(path, V, F, N) -> None` (écrit `v`, `vn`, `f a//a b//b c//c`, indices 1-based)
  - `python -m pipeline.assets` écrit `web/data/candle.obj`, `web/data/wick.obj`, `web/data/config.json`

Valeurs de départ de `config.py` (réglables, mais uniquement ici) :

```python
CANDLE = dict(radius=0.035, height=0.10, cup_depth=0.006, rim_radius=0.003,
              wobble=0.0008, seed=7, segments=128, rings=96)
FLAME = dict(base=(0.0, 0.110, 0.0), sphere_radius=0.003, radiance=(40.0, 22.0, 8.0),
             flicker_xz=0.003, flicker_y=0.002, intensity_range=(0.8, 1.2))
WAX = dict(ior=1.44, roughness=0.1, sigma_t=800.0, albedo=(0.995, 0.99, 0.975))
TABLE = dict(size=1.0, albedo=(0.08, 0.06, 0.05))
CAMERA = dict(fov_y=35.0, target=(0.0, 0.06, 0.0), radius=(0.2, 0.5), elevation_deg=(-5.0, 60.0))
EXPOSURE = 1.0
REF_VIEWS = [dict(az_deg=30, el_deg=15, radius=0.3), dict(az_deg=120, el_deg=35, radius=0.3),
             dict(az_deg=220, el_deg=5, radius=0.3), dict(az_deg=300, el_deg=55, radius=0.3)]
WIDTHS = (8, 16, 32)
```

Bougie : cylindre fermé, axe Y, base à y=0, bord supérieur arrondi (`rim_radius`), cuvette concave au sommet (profondeur `cup_depth` au centre), rayon perturbé par un bruit déterministe d'amplitude `wobble` (seed). Normales lissées sortantes. Mèche : cylindre r=0.0008 de y=`height-cup_depth` à `FLAME.base.y - 0.004`.

- [ ] **Step 1:** `git init`, `.gitignore`, `requirements.txt`, `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Expected : `python -c "import mitsuba, torch"` OK.
- [ ] **Step 2: Write the failing tests** in `pipeline/tests/test_assets.py` :

```python
def test_candle_bounds():
    V, F, N = candle_mesh()
    assert V[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert V[:, 1].max() <= CANDLE["height"] + 1e-6
    r = np.hypot(V[:, 0], V[:, 2])
    assert r.max() <= CANDLE["radius"] + CANDLE["wobble"] + 1e-6

def test_candle_closed():  # chaque arête partagée par exactement 2 faces
    _, F, _ = candle_mesh()
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert (counts == 2).all()

def test_normals_outward_unit():
    V, _, N = candle_mesh()
    assert np.allclose(np.linalg.norm(N, axis=1), 1, atol=1e-4)
    side = (V[:, 1] > 0.01) & (V[:, 1] < 0.08)
    assert (np.einsum("ij,ij->i", N[side][:, [0, 2]], V[side][:, [0, 2]]) > 0).all()

def test_obj_roundtrip(tmp_path):  # write_obj puis relecture : même nb de v/vn/f
```

- [ ] **Step 3:** Run `.venv/bin/pytest pipeline/tests/test_assets.py -v` → FAIL (import).
- [ ] **Step 4:** Implémenter `config.py`, `assets.py`.
- [ ] **Step 5:** Run tests → PASS ; `python -m pipeline.assets` → les 3 fichiers existent dans `web/data/`.
- [ ] **Step 6:** Commit `feat: shared config and candle geometry`.

---

### Task 2: Scène Mitsuba et rendu avec AOV

**Files:**
- Create: `pipeline/scene.py`
- Test: `pipeline/tests/test_scene.py`

**Interfaces:**
- Consumes: `config.*`, `web/data/candle.obj`, `web/data/wick.obj`
- Produces:
  - `camera_origin(view: dict) -> np.ndarray(3)` — orbite autour de `CAMERA.target`, az autour de Y depuis +Z, el au-dessus de l'horizontale.
  - `render(view: dict, flame_offset: np.ndarray(3), res: int, spp: int) -> dict` avec clés `full (H,W,3)`, `spec (H,W,3)`, `pos (H,W,3)`, `nrm (H,W,3)`, `mask (H,W) bool` (pixels cire), tout en float32 linéaire.

Décisions :
- `mi.set_variant("llvm_rgb")`, repli `scalar_rgb`.
- Caméra `perspective`, `fov=CAMERA.fov_y`, `fov_axis="y"`, `look_at(origin, target, up=(0,1,0))`, image carrée `res×res`.
- Cire : `roughdielectric(int_ior=WAX.ior, alpha=WAX.roughness)` + `interior` = `homogeneous(sigma_t=WAX.sigma_t, albedo=WAX.albedo)`. Intégrateur `volpath`, `max_depth=128`.
- **Passe `spec`** (isole le spéculaire) : même scène, milieu intérieur `albedo=0, sigma_t=1e5` (absorbeur) → seule la réflexion de surface subsiste. Le SSS vaut `full - spec` (non clampé).
- Flamme : `sphere(radius=FLAME.sphere_radius, center=base+offset)` + `area(radiance=FLAME.radiance)`. Mèche : `diffuse` noir 0.02. Table : `rectangle` horizontal à y=0, taille `TABLE.size`, `diffuse(TABLE.albedo)`.
- AOV : intégrateur `aov` avec `"pos:position,nn:sh_normal,si:shape_index"` ; `mask = si == index de la forme cire`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_camera_origin_distance():
    o = camera_origin(dict(az_deg=0, el_deg=0, radius=0.3))
    assert np.allclose(o, np.array(CAMERA["target"]) + [0, 0, 0.3], atol=1e-6)

def test_render_smoke():
    r = render(REF_VIEWS[0], np.zeros(3), res=32, spp=8)
    assert r["full"].shape == (32, 32, 3) and np.isfinite(r["full"]).all()
    assert r["mask"].sum() > 20
    m = r["mask"]
    assert r["full"][m].mean() > r["spec"][m].mean()      # la cire diffuse de la lumière
    assert np.allclose(np.linalg.norm(r["nrm"][m], axis=1), 1, atol=1e-3)
```

- [ ] **Step 2:** Run `pytest pipeline/tests/test_scene.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `scene.py`.
- [ ] **Step 4:** Run → PASS. Sauver un rendu 256² 128 spp tonemappé grossièrement en PNG dans `pipeline/out/preview.png` et **le regarder** : bougie qui luit en haut, table sombre. Ajuster `WAX`/`FLAME.radiance` dans `config.py` si c'est absurde (tout noir, tout blanc).
- [ ] **Step 5:** Commit `feat: mitsuba scene with spec-only pass and AOVs`.

---

### Task 3: Features et génération des données

**Files:**
- Create: `pipeline/features.py`, `pipeline/gen_data.py`
- Test: `pipeline/tests/test_features.py`

**Interfaces:**
- Consumes: `render`, `camera_origin`, `config`
- Produces:
  - `features(pos (N,3), nrm (N,3), cam (3,), flame_offset (3,)) -> float32 (N,17)`, ordre **figé** (le GLSL de la Task 7 le reproduit) :
    `pos(3) | nrm(3) | view=normalize(cam-pos)(3) | flame_offset(3) | L=normalize(flame-pos)(3) | dist=|flame-pos|(1) | dot(nrm,L)(1)` avec `flame = FLAME.base + flame_offset`.
  - `chord_to_flame(pos (N,3), flame (3,), radius, height) -> (N,)` : longueur du segment pos→flame à l'intérieur du cylindre `r<radius, 0<y<height` (cuvette ignorée), points de départ sur la surface compris.
  - `python -m pipeline.gen_data --views 200 --res 128 --spp 256 --out pipeline/out/train.npz` : vues aléatoires (az ∈ [0,360), el ∈ `CAMERA.elevation_deg`, rayon ∈ `CAMERA.radius`, offset uniforme dans ±`flicker_xz`/±`flicker_y`/±`flicker_xz`), seed fixe. Sauve `X raw (N,17)`, `Y = full-spec (N,3)` sur les pixels `mask`, `view_id (N,)`, plus `pos, nrm, cam, offset` par échantillon (pour le fit classique). Écrit au fil de l'eau (reprise possible : `--start`).

- [ ] **Step 1: Write the failing tests:**

```python
def test_feature_layout():
    pos = np.array([[0.035, 0.05, 0.0]]); nrm = np.array([[1.0, 0, 0]])
    x = features(pos, nrm, cam=np.array([0.3, 0.05, 0]), flame_offset=np.zeros(3))
    assert x.shape == (1, 17)
    assert np.allclose(x[0, 6:9], [1, 0, 0])                # view
    L = np.array(FLAME["base"]) - pos[0]; d = np.linalg.norm(L)
    assert np.allclose(x[0, 12:15], L / d) and np.isclose(x[0, 15], d)
    assert np.isclose(x[0, 16], (L / d)[0])

def test_chord_side_point_to_axis_flame():
    # point sur le flanc à y=0.05, flamme sur l'axe à y=0.05 : corde = rayon
    c = chord_to_flame(np.array([[0.035, 0.05, 0]]), np.array([0, 0.05, 0]), 0.035, 0.10)
    assert np.isclose(c[0], 0.035, atol=1e-6)

def test_chord_outside_is_zero():
    c = chord_to_flame(np.array([[0.2, 0.0, 0]]), np.array([0, 0.5, 0]), 0.035, 0.10)
    assert c[0] == 0
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implémenter. **Step 4:** Run → PASS.
- [ ] **Step 5:** Lancer un mini-run `--views 4 --res 32 --spp 16` et vérifier que le npz a N>0 et pas de NaN.
- [ ] **Step 6:** Commit `feat: feature encoding and training data generation`.
- [ ] **Step 7:** Lancer le vrai run **en arrière-plan** (`--views 200 --res 128 --spp 256`). Mesurer le temps de la première vue ; si > 3 min/vue, baisser `--spp` à 128 avant de continuer. Les Tasks 4–5 peuvent se développer sur le mini-run pendant ce temps.

---

### Task 4: Entraînement et export des MLP

**Files:**
- Create: `pipeline/train.py`
- Test: `pipeline/tests/test_train.py`

**Interfaces:**
- Consumes: `pipeline/out/train.npz`, `features`
- Produces:
  - `class MLP(nn.Module)(in_dim=17, width, depth=2)` : `Linear→relu(x)²` × depth, puis `Linear(width,3)` et sortie `exp(o)`. Entrées standardisées par `in_mean/in_std` (calculés sur le train).
  - `train(npz_path, width, epochs=...) -> (model, in_mean, in_std, val_rel_err)` ; split par `view_id` (10 % des vues en validation). Perte relative : `((pred - y)**2 / (pred.detach() + 0.01)**2).mean()` (pas de transformation non linéaire de la cible). Adam.
  - `export(model, in_mean, in_std, path)` → JSON `{"width": w, "in_mean": [17], "in_std": [17], "layers": [{"W": [[in]...out], "b": [out]}, ...]}` (W en `[out][in]`, 3 couches).
  - `forward_numpy(weights: dict, x_raw (N,17)) -> (N,3)`.
  - CLI `python -m pipeline.train --data pipeline/out/train.npz` → `web/data/weights_{8,16,32}.json` + `web/data/selftest.json` = 64 échantillons de validation `{"pos","nrm","cam","flame_offset","expected"}` (expected = `forward_numpy` du réseau 16).

- [ ] **Step 1: Write the failing tests:**

```python
def test_forward_numpy_matches_torch(tmp_path):
    m = MLP(17, 8); mean, std = np.zeros(17), np.ones(17)
    export(m, mean, std, tmp_path / "w.json")
    x = np.random.default_rng(0).normal(size=(10, 17)).astype(np.float32)
    w = json.loads((tmp_path / "w.json").read_text())
    with torch.no_grad(): ref = m(torch.from_numpy(x)).numpy()
    assert np.allclose(forward_numpy(w, x), ref, rtol=1e-5)

def test_overfits_toy(tmp_path):  # npz synthétique : y = 0.5 + 0.1*x[:, :3] ; val_rel_err < 0.05
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implémenter. **Step 4:** Run → PASS.
- [ ] **Step 5:** Une fois `train.npz` complet : lancer le CLI, noter `val_rel_err` pour 8/16/32 dans le message de commit. Expected : erreur décroissante avec la largeur.
- [ ] **Step 6:** Commit `feat: MLP training and JSON export`.

---

### Task 5: Approximation classique ajustée

**Files:**
- Create: `pipeline/classic.py`
- Test: `pipeline/tests/test_classic.py`

**Interfaces:**
- Consumes: `chord_to_flame`, `train.npz`
- Produces:
  - `classic_sss(pos, nrm, cam, flame, p: dict) -> (N,3)` avec, `L=normalize(flame-pos)`, `d=|flame-pos|`, `c=chord_to_flame(...)` :
    `sss = (1/d²) * tint * (a * max((dot(nrm,L)+w)/(1+w), 0) + b * exp(-k*c))`
    paramètres `p = {w, k, a, b, tint[3]}`.
  - `fit_classic(npz_path) -> dict` : Adam sur la même perte relative que la Task 4, paramètres contraints positifs (softplus). CLI écrit `web/data/classic.json`.

L'idée : la technique « wrap + épaisseur » standard, mais **réglée au mieux** sur les mêmes données que le réseau, pour que la comparaison soit honnête.

- [ ] **Step 1: Write the failing test:** `test_classic_positive_and_decays_with_chord` — même point/flamme, `c` plus grand ⇒ sortie plus petite quand `a=0` ; sortie ≥ 0.
- [ ] **Step 2–4:** FAIL → implémenter → PASS ; fit sur le vrai npz, imprimer l'erreur relative de validation (à comparer à la Task 4).
- [ ] **Step 5:** Commit `feat: fitted classic translucency baseline`.

---

### Task 6: Références path tracées et évaluation

**Files:**
- Create: `pipeline/tonemap.py`, `pipeline/render_refs.py`, `pipeline/evaluate.py`
- Test: `pipeline/tests/test_tonemap.py`

**Interfaces:**
- Consumes: `render`, `forward_numpy`, `classic_sss`, `REF_VIEWS`, `EXPOSURE`
- Produces:
  - `aces_filmic(rgb_linear, exposure) -> float [0,1]` puis `to_srgb8(x) -> uint8` : port **verbatim** de `ACESFilmicToneMapping` et de la conversion sRGB de la version three.js épinglée (`src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js`), avec la version citée en commentaire.
  - `python -m pipeline.render_refs --res 1024 --spp 512` → `web/data/refs/view{i}.png`, `web/data/refs.json` (`[{az_deg, el_deg, radius}]`), données brutes `pipeline/out/ref{i}.npz`. Flamme à offset 0, intensité 1.
  - `python -m pipeline.evaluate` → `docs/results.md` : par vue et par méthode (classique, MLP 8/16/32), RMSE relative sur `mask` pour (a) le terme SSS seul, (b) l'image `sss_pred + spec` contre `full`.

- [ ] **Step 1: Write the failing tests:** `aces_filmic(0)==0` ; monotone sur `np.linspace(0,50,200)` ; `aces_filmic(1e4)` > 0.99 ; `to_srgb8` de 0 et 1 → 0 et 255.
- [ ] **Step 2–4:** FAIL → implémenter → PASS.
- [ ] **Step 5:** Lancer `render_refs` en arrière-plan (long). Si le bruit est visible sur les PNG, augmenter `--spp` sur une seule vue pour estimer le coût avant de tout relancer.
- [ ] **Step 6:** Lancer `evaluate`, relire `docs/results.md` : le MLP 16 doit battre le classique sur (a). Sinon, revenir à la Task 4 (epochs, largeur) avant de continuer.
- [ ] **Step 7:** Commit `feat: path-traced references and evaluation`.

---

### Task 7: Rendu neuronal dans three.js + auto-test de parité

**Files:**
- Create: `web/package.json`, `web/index.html`, `web/src/main.js`, `web/src/materials.js`, `web/src/glsl.js`, `web/src/mlp.js`, `web/src/flame.js`, `web/selftest.html`, `web/src/selftest.js`
- Test: `web/test/mlp.test.js`, `web/test/flame.test.js`

**Interfaces:**
- Consumes: `web/data/{config.json,candle.obj,wick.obj,weights_*.json,selftest.json}`
- Produces:
  - `mlpGlsl(weights) -> string` : définit `vec3 neuralSSS(float x[17])` ; poids en `const float` arrays (`const float W0[17*w] = float[](...)`) + boucles, standardisation incluse, sortie `exp`. Une matière compilée par largeur, toutes au chargement.
  - `glsl.js` exporte `FEATURES_GLSL` (`void features(vec3 pos, vec3 nrm, vec3 cam, vec3 flameOffset, vec3 flameBase, out float x[17])`, même ordre que la Task 3), `CHORD_GLSL` (`float chordToFlame(vec3 pos, vec3 flame, float radius, float height)`), `GGX_GLSL` (spéculaire GGX, `F0 = ((ior-1)/(ior+1))²`, `alpha = WAX.roughness`).
  - `flameOffset(t: number, cfg) -> {x, y, z, intensity}` : somme de sinus déterministe, bornée par `flicker_xz`, `flicker_y`, `intensity_range`.
  - `makeWaxMaterial({mode: "neural"|"classic", glsl, config, classic?}) -> THREE.ShaderMaterial` ; uniforms `uFlameOffset`, `uFlameIntensity`. Neural : `sss = neuralSSS(x) * uFlameIntensity`. Sortie `sss + spec` puis `#include <tonemapping_fragment>` / `<colorspace_fragment>`.
  - `makeTableMaterial(config)` : Lambert × `1/d²` × visibilité `chordToFlame(...) == 0`. (Écart connu et accepté : la lueur transmise par la cire sur la table n'est pas modélisée.)
  - Intensité ponctuelle de la flamme côté three : `I = radiance · π · sphere_radius²`.
  - Renderer : `toneMapping = ACESFilmicToneMapping`, `toneMappingExposure = config.exposure`, canvas carré, DPR `min(devicePixelRatio, 2)`. Flamme visuelle : sprite additif au-dessus de la mèche.

- [ ] **Step 1: Write the failing tests:**

```js
// mlp.test.js
test("const array sizes match width", () => {
  const w = JSON.parse(readFileSync("data/weights_16.json"));
  const src = mlpGlsl(w);
  assert.match(src, /float W0\[272\]/);   // 17*16
  assert.match(src, /float W1\[256\]/);   // 16*16
  assert.match(src, /float W2\[48\]/);    // 16*3
  assert.match(src, /vec3 neuralSSS\(float x\[17\]\)/);
});
// flame.test.js — Review Focus #2
test("flicker stays in training bounds", () => {
  for (let t = 0; t < 1000; t += 0.013) {
    const f = flameOffset(t, cfg);
    assert.ok(Math.abs(f.x) <= cfg.flicker_xz && Math.abs(f.z) <= cfg.flicker_xz);
    assert.ok(Math.abs(f.y) <= cfg.flicker_y);
    assert.ok(f.intensity >= cfg.intensity_range[0] && f.intensity <= cfg.intensity_range[1]);
  }
});
```

- [ ] **Step 2:** `cd web && node --test` → FAIL. **Step 3:** Implémenter. **Step 4:** `node --test` → PASS.
- [ ] **Step 5: Auto-test GLSL** (`selftest.html`) : rend une texture 64×1 RGBA32F où chaque pixel évalue `features` + `neuralSSS` sur l'entrée i de `selftest.json`, relit avec `readPixels`, compare à `expected` (tolérance relative 1e-3). Affiche `PASS n/64`, `FAIL` (avec le pire écart) ou `UNSUPPORTED` si `EXT_color_buffer_float` est absent (Review Focus #1).
  Run : `python -m http.server -d web 8000`, ouvrir `http://localhost:8000/selftest.html`. Expected : `PASS 64/64`.
- [ ] **Step 6:** Ouvrir `index.html` : bougie en rendu neuronal, orbite, flamme qui vacille, pas d'erreur console.
- [ ] **Step 7:** Commit `feat: neural wax shader in three.js with GLSL parity selftest`.

---

### Task 8: Mode classique, rideau, références

**Files:**
- Create: `web/src/refs.js`
- Modify: `web/src/main.js`, `web/src/materials.js`, `web/index.html`
- Test: `web/test/refs.test.js`

**Interfaces:**
- Consumes: `makeWaxMaterial`, `web/data/{classic.json,refs.json,refs/*.png}`
- Produces:
  - Classique : `makeWaxMaterial({mode:"classic", classic})` implémente exactement la formule de la Task 5 en GLSL.
  - `nearestRefView(camDir: [x,y,z], refs) -> index` (direction depuis la cible, distance angulaire minimale).
  - `orbitLimits(config) -> {minPolarAngle, maxPolarAngle, minDistance, maxDistance}` dérivés de `CAMERA.elevation_deg` / `CAMERA.radius` (Review Focus #3).
  - Rideau : position `split ∈ [0,1]`, poignée draggable (pointer events, tactile). Chaque côté a un menu `classic | neural | ref` (défaut : gauche classic, droite neural). Rendu par `setScissor` de chaque moitié avec la matière du côté.
  - Côté `ref` : la caméra s'anime (~400 ms) vers `nearestRefView`, la flamme se fige (offset 0, intensité 1), la moitié affiche `refs/view{i}.png` en quad plein écran, sprite de flamme par-dessus. Au premier `controls.start` suivant, ce côté repasse en `neural`.

- [ ] **Step 1: Write the failing tests:**

```js
test("nearest ref view", () => {
  // direction exacte de REF_VIEWS[2] -> 2
});
test("orbit limits match training domain", () => {
  const l = orbitLimits(cfg);
  assert.ok(Math.abs(l.maxPolarAngle - (Math.PI/2 + 5*Math.PI/180)) < 1e-9);  // el min -5°
  assert.ok(Math.abs(l.minPolarAngle - (Math.PI/2 - 60*Math.PI/180)) < 1e-9); // el max 60°
  assert.equal(l.minDistance, 0.2); assert.equal(l.maxDistance, 0.5);
});
```

- [ ] **Step 2–4:** FAIL → implémenter → PASS.
- [ ] **Step 5: Vérif d'alignement** : mettre les deux côtés en `neural` | `ref` sur chaque vue : silhouette, table et flamme alignées au pixel près le long du rideau. Si décalage : comparer FOV/axe/caméra avec la Task 2 avant toute autre retouche.
- [ ] **Step 6: Vérif rotation** (Review Focus #5) : dans les devtools en mode appareil, passer portrait ↔ paysage avec un côté en `ref` : canvas toujours carré, alignement conservé.
- [ ] **Step 7:** Commit `feat: classic mode, comparison curtain and reference views`.

---

### Task 9: HUD, réglages, benchmark

**Files:**
- Create: `web/src/hud.js`, `web/src/bench.js`
- Modify: `web/src/main.js`, `web/index.html`
- Test: `web/test/hud.test.js`

**Interfaces:**
- Produces:
  - `formatHud({leftLabel, leftMs, rightLabel, rightMs, frameMs, device}) -> string` : si `leftMs`/`rightMs` sont `null` (pas d'extension timer), affiche `frame X.X ms` à la place ; jamais `NaN`.
  - `gpuTimer(gl) -> {begin(), end(), poll() -> ms|null} | null` via `EXT_disjoint_timer_query_webgl2` (null si absent).
  - `bench(renderer, renderFn, {seconds=3, size=1024, reps=20}) -> Promise<number>` : rend `renderFn` `reps` fois par frame dans un `WebGLRenderTarget(size,size)`, synchronise par `readPixels` 1×1, renvoie ms par passe.
  - Bouton « Mesurer » : bench successif de `classic`, `neural 8/16/32` en plein canvas, résultat affiché en tableau superposé + `console.table` (copier-coller pour l'article). Device : `navigator.userAgentData?.model ?? userAgent` abrégé.
  - Réglages : largeur 8/16/32 (défaut 16, change la matière neural instantanément, matières précompilées) ; flamme vacille/fixe.

- [ ] **Step 1: Write the failing tests:**

```js
test("hud without timer shows frame time", () => {
  const s = formatHud({leftLabel:"Classique", leftMs:null, rightLabel:"Neural", rightMs:null, frameMs:16.7, device:"Pixel 6a"});
  assert.match(s, /frame 16\.7 ms/); assert.doesNotMatch(s, /NaN|null|undefined/);
});
test("hud with timer shows both sides", () => { /* "Classique 0.9 ms · Neural 1.4 ms" */ });
```

- [ ] **Step 2–4:** FAIL → implémenter → PASS.
- [ ] **Step 5:** « Mesurer » sur le PC : le neural 32 est plus lent que le 16, lui-même plus lent que le 8 ; les nombres sont stables à ±10 % sur deux lancements.
- [ ] **Step 6:** Commit `feat: HUD, network size switch, benchmark mode`.

---

### Task 10: Publication et note de résultats

**Files:**
- Create: `README.md`, `.github/workflows/pages.yml`
- Modify: `docs/results.md`

**Interfaces:**
- Produces : workflow GitHub Pages officiel (`actions/upload-pages-artifact` sur `web/`, `actions/deploy-pages`) ; README = commandes du pipeline dans l'ordre (`assets → gen_data → train → classic → render_refs → evaluate`), durées constatées, comment servir en local.

- [ ] **Step 1:** Écrire README + workflow. Vérifier que `web/` servi seul fonctionne (chemins relatifs partout, aucun `../`).
- [ ] **Step 2:** Compléter `docs/results.md` (déjà rempli par `evaluate.py`) avec : temps du pipeline, poids des fichiers (`weights_*.json`, total `web/data`), résultats « Mesurer » sur le PC, et un emplacement vide pour les chiffres du Pixel 6a.
- [ ] **Step 3:** Commit `docs: readme, pages workflow, results notes`. **Ne pas pousser** : la création du dépôt GitHub et le push sont faits par l'auteur.
- [ ] **Step 4 (auteur) :** pousser, ouvrir la page sur le Pixel 6a, lancer `selftest.html` puis « Mesurer », reporter les chiffres dans `docs/results.md`.
