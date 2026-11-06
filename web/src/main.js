import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { flameUniforms, makeTableMaterial, makeWaxMaterial } from "./materials.js";
import { flameOffset } from "./flame.js";
import { leaveRef, nearestRefView, orbitLimits, viewDir, withPose } from "./refs.js";
import { formatHud } from "./hud.js";
import { bench, gpuTimer } from "./bench.js";

const $ = (id) => document.getElementById(id);
const json = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const LABELS = { classic: "Classique", neural: "Neural", ref: "Path tracé" };

try {
  await main();
} catch (e) {
  document.body.insertAdjacentHTML("afterbegin", `<div id="err">Erreur : ${e.message}</div>`);
  throw e;
}

async function main() {
  const cfg = await json("data/config.json");
  const [classic, refs, ...weights] = await Promise.all([
    json("data/classic.json"), json("data/refs.json"), ...cfg.widths.map((w) => json(`data/weights_${w}.json`)),
  ]);

  // --- renderer, camera, controls -------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas: $("c"), antialias: true });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = cfg.exposure;
  renderer.autoClear = false;
  renderer.setClearColor(0x000000);
  const gl = renderer.getContext();

  const camera = new THREE.PerspectiveCamera(cfg.camera.fov_y, 1, 0.01, 10);
  const target = new THREE.Vector3(...cfg.camera.target);
  const placeAt = (v) => camera.position.copy(target).addScaledVector(new THREE.Vector3(...viewDir(v)), v.radius);
  placeAt(refs[0]);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(target);
  Object.assign(controls, orbitLimits(cfg), { enablePan: false, enableDamping: false });
  controls.update();

  // --- scene ----------------------------------------------------------------------------------
  const shared = flameUniforms(cfg);
  const mats = { classic: makeWaxMaterial({ mode: "classic", classic, cfg, shared }) };
  cfg.widths.forEach((w, i) => { mats[`n${w}`] = makeWaxMaterial({ mode: "neural", weights: weights[i], cfg, shared }); });
  const loader = new OBJLoader();
  const [candleObj, wickObj] = await Promise.all([loader.loadAsync("data/candle.obj"), loader.loadAsync("data/wick.obj")]);
  const candle = candleObj.children[0];
  wickObj.children[0].material = new THREE.MeshBasicMaterial({ color: 0x050403 });
  const s = cfg.table.size;
  const table = new THREE.Mesh(new THREE.PlaneGeometry(s, s).rotateX(-Math.PI / 2), makeTableMaterial(cfg, shared));
  const scene = new THREE.Scene();
  scene.add(candleObj, wickObj, table);

  const flameScene = new THREE.Scene();
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: flameTexture(), blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, transparent: true,
  }));
  sprite.scale.set(0.012, 0.026, 1);
  flameScene.add(sprite);

  const refScene = new THREE.Scene();
  const refMat = new THREE.MeshBasicMaterial({ toneMapped: false, depthWrite: false, depthTest: false });
  refScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), refMat));
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const texLoader = new THREE.TextureLoader();
  const refTex = refs.map((_, i) => {
    const t = texLoader.load(`data/refs/view${i}.png`);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  });

  // --- UI state -------------------------------------------------------------------------------
  const state = { split: 0.5, sides: { left: "classic", right: "neural" }, width: 16, flicker: true, refIdx: null };
  let size = 0, tween = null, benching = false;

  for (const side of ["left", "right"]) {
    const sel = $(side);
    sel.innerHTML = Object.entries(LABELS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
    sel.value = state.sides[side];
    sel.onchange = () => setSide(side, sel.value);
  }
  function setSide(side, mode) {
    state.sides[side] = mode;
    $(side).value = mode;
    if (mode === "ref") {
      const dir = camera.position.clone().sub(target).normalize().toArray();
      state.refIdx = nearestRefView(dir, refs);
      refMat.map = refTex[state.refIdx];
      refMat.needsUpdate = true;
      const v = refs[state.refIdx];
      tween = { from: camera.position.clone().sub(target), to: new THREE.Vector3(...viewDir(v)).multiplyScalar(v.radius), t0: performance.now() };
    } else if (!Object.values(state.sides).includes("ref")) {
      state.refIdx = null;
    }
  }
  // deep links for the article, e.g. ?left=neural&right=ref&split=0.5
  const q = new URLSearchParams(location.search);
  for (const side of ["left", "right"]) if (q.get(side) in LABELS) setSide(side, q.get(side));
  if (q.has("split")) state.split = THREE.MathUtils.clamp(+q.get("split") || 0.5, 0, 1);
  // A user gesture cancels the snap tween; the first actual camera move (not a mere tap)
  // leaves the reference pose, so "Path tracé" never stays over a moving view.
  let orbiting = false;
  controls.addEventListener("start", () => { tween = null; orbiting = true; });
  controls.addEventListener("end", () => { orbiting = false; });
  controls.addEventListener("change", () => {
    if (!orbiting) return;
    const next = leaveRef(state.sides);
    for (const side of ["left", "right"]) if (next[side] !== state.sides[side]) setSide(side, next[side]);
  });

  $("widths").innerHTML = cfg.widths.map((w) => `<button data-w="${w}">${w}</button>`).join("");
  const markWidth = () => $("widths").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.w === state.width));
  $("widths").onclick = (e) => { if (e.target.dataset.w) { state.width = +e.target.dataset.w; markWidth(); } };
  markWidth();
  const markFlicker = () => { $("flicker").classList.toggle("on", state.flicker); $("still").classList.toggle("on", !state.flicker); };
  $("flicker").onclick = () => { state.flicker = true; markFlicker(); };
  $("still").onclick = () => { state.flicker = false; markFlicker(); };

  // curtain handle
  const handle = $("handle");
  let dragging = false;
  handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); e.stopPropagation(); });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = $("stage").getBoundingClientRect();
    state.split = THREE.MathUtils.clamp((e.clientX - r.left) / r.width, 0, 1);
    handle.style.left = `${state.split * 100}%`;
  });
  handle.addEventListener("pointerup", () => { dragging = false; });
  handle.style.left = `${state.split * 100}%`;

  function resize() {
    size = Math.max(240, Math.floor(Math.min(innerWidth, innerHeight - $("controls").offsetHeight)));
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(size, size);
    $("stage").style.width = `${size}px`;
  }
  addEventListener("resize", resize);
  resize();

  // --- HUD ------------------------------------------------------------------------------------
  const timers = { left: gpuTimer(gl), right: gpuTimer(gl) };
  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  let device = (dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "").replace(/^ANGLE \(|\)$/g, "").slice(0, 48);
  navigator.userAgentData?.getHighEntropyValues?.(["model"]).then((v) => { if (v.model) device = v.model; }).catch(() => {});
  const label = (mode) => (mode === "neural" ? `Neural ${state.width}` : LABELS[mode]);

  // --- render loop ----------------------------------------------------------------------------
  function renderSide(side, x, w) {
    const mode = state.sides[side];
    renderer.setScissor(x, 0, w, size);
    renderer.clear();
    timers[side]?.begin();
    if (mode === "ref") {
      renderer.render(refScene, ortho);
    } else {
      candle.material = mode === "classic" ? mats.classic : mats[`n${state.width}`];
      renderer.render(scene, camera);
    }
    timers[side]?.end();
    renderer.render(flameScene, camera);
  }

  let last = performance.now(), frameMs = NaN;
  renderer.setAnimationLoop((now) => {
    if (benching) return;
    frameMs = Number.isFinite(frameMs) ? 0.9 * frameMs + 0.1 * (now - last) : now - last;
    last = now;
    if (tween) {
      const k = Math.min((now - tween.t0) / 400, 1), e = k * k * (3 - 2 * k);
      const dir = tween.from.clone().normalize().lerp(tween.to.clone().normalize(), e).normalize();
      camera.position.copy(target).addScaledVector(dir, THREE.MathUtils.lerp(tween.from.length(), tween.to.length(), e));
      controls.update();
      if (k === 1) tween = null;
    }
    const f = state.flicker && state.refIdx === null ? flameOffset(now / 1000, cfg.flame) : { x: 0, y: 0, z: 0, intensity: 1 };
    shared.uFlameOffset.value.set(f.x, f.y, f.z);
    shared.uFlameIntensity.value = f.intensity;
    sprite.position.set(cfg.flame.base[0] + f.x, cfg.flame.base[1] + f.y + 0.008, cfg.flame.base[2] + f.z);
    sprite.material.opacity = 0.85 + 0.15 * (f.intensity - 1) / 0.2;

    renderer.setScissorTest(true);
    const xs = Math.round(state.split * size);
    renderSide("left", 0, xs);
    renderSide("right", xs, size - xs);
    renderer.setScissorTest(false);

    $("hud").textContent = formatHud({
      leftLabel: label(state.sides.left), leftMs: timers.left?.poll() ?? null,
      rightLabel: label(state.sides.right), rightMs: timers.right?.poll() ?? null,
      frameMs, device,
    });
  });

  // --- benchmark ------------------------------------------------------------------------------
  $("measure").onclick = async () => {
    const btn = $("measure"), out = $("bench");
    btn.disabled = true;
    benching = true;
    shared.uFlameOffset.value.set(0, 0, 0);
    shared.uFlameIntensity.value = 1;
    const rows = [];
    const techniques = [["Classique", mats.classic], ...cfg.widths.map((w) => [`Neural ${w}`, mats[`n${w}`]])];
    // fixed pose (reference view 0) so numbers compare across devices whatever the user's zoom
    const v0 = refs[0];
    const pose = target.clone().addScaledVector(new THREE.Vector3(...viewDir(v0)), v0.radius);
    await withPose(camera, pose, target, async () => {
      for (const [name, mat] of techniques) {
        out.textContent = `mesure : ${name}…`;
        candle.material = mat;
        const ms = await bench(renderer, () => { renderer.clear(); renderer.render(scene, camera); });
        rows.push({ technique: name, "ms / image 1024²": +ms.toFixed(3) });
      }
    });
    out.textContent = `${device}\n` + rows.map((r) => `${r.technique.padEnd(10)} ${r["ms / image 1024²"].toFixed(2)} ms`).join("\n");
    console.table(rows);
    benching = false;
    btn.disabled = false;
  };
}

function flameTexture() {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 128;
  const g = c.getContext("2d");
  g.translate(32, 80);
  g.scale(1, 2);
  const grd = g.createRadialGradient(0, 4, 1, 0, 0, 30);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,224,138,0.95)");
  grd.addColorStop(0.55, "rgba(255,154,42,0.6)");
  grd.addColorStop(1, "rgba(255,120,20,0)");
  g.fillStyle = grd;
  g.fillRect(-32, -40, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
