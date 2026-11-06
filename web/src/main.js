import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";
import { EXRLoader } from "three/addons/loaders/EXRLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { Pass } from "three/addons/postprocessing/Pass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";

// Sensor saturation before bloom: the bulb's radiance (thousands) would otherwise wash the whole
// frame white. Applied to the composed frame, so both curtain sides get the same treatment.
const ClampShader = {
  uniforms: { tDiffuse: { value: null }, uMax: { value: 10.0 } },
  vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "uniform sampler2D tDiffuse; uniform float uMax; varying vec2 vUv; void main() { vec4 c = texture2D(tDiffuse, vUv); gl_FragColor = vec4(min(c.rgb, vec3(uMax)), c.a); }",
};
import { makeGround, makeJadeMaterial, makeLamp, setJadeAttributes } from "./materials.js";
import { backlight, clampLamp, lampPosition } from "./lamp.js";
import { leaveRef, nearestRefView, orbitLimits, viewDir, withPose } from "./refs.js";
import { effectiveWidth, pickPipeline, sideMode } from "./policy.js";
import { formatHud } from "./hud.js";
import { bench, benchFrame, gpuTimer } from "./bench.js";

const $ = (id) => document.getElementById(id);
const json = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const bytes = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.arrayBuffer(); })
  .then((b) => new Uint8Array(b));
const LABELS = { classic: "Classique", neural: "Neural", ref: "Path tracé" };
const RAD = 180 / Math.PI;

try {
  await main();
} catch (e) {
  document.body.insertAdjacentHTML("afterbegin", `<div id="err">Erreur : ${e.message}</div>`);
  throw e;
}

async function main() {
  const cfg = await json("data/config.json");
  const [envSH, classic, geometry, thickness] = await Promise.all([
    json("data/env_sh.json"), json("data/classic.json"), new PLYLoader().loadAsync("data/dragon.ply"),
    bytes("data/thickness.bin"),
  ]);
  const hdr = await new HDRLoader().loadAsync(`data/${envSH.file}`);
  hdr.mapping = THREE.EquirectangularReflectionMapping;

  // --- renderer, pipeline, camera ---------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas: $("c"), antialias: true });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = cfg.exposure;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.autoClear = false;
  const gl = renderer.getContext();
  const pipeline = pickPipeline({ halfFloatRT: renderer.extensions.has("EXT_color_buffer_half_float") || renderer.extensions.has("EXT_color_buffer_float") });

  const envMap = new THREE.PMREMGenerator(renderer).fromEquirectangular(hdr).texture;
  const camera = new THREE.PerspectiveCamera(cfg.camera.fov_y, 1, 0.01, 20);
  const target = new THREE.Vector3(...cfg.camera.target);
  const posOf = (v) => target.clone().addScaledVector(new THREE.Vector3(...viewDir(v)), v.radius);
  camera.position.copy(posOf(cfg.ref_poses[0]));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(target);
  Object.assign(controls, orbitLimits(cfg), { enablePan: false, enableDamping: false });
  controls.update();

  // --- scene ------------------------------------------------------------------------------------
  const shared = {
    uLamp: { value: new THREE.Vector3() },
    uSH: { value: envSH.sh.map((c) => new THREE.Vector3(...c.map((x) => x * cfg.env.scale))) },
  };
  const scene = new THREE.Scene();
  const setBackground = (s) => { s.background = hdr; s.backgroundBlurriness = 0.4; s.backgroundIntensity = cfg.env.scale; };
  setBackground(scene);
  const dragon = new THREE.Mesh(geometry);
  dragon.castShadow = true;
  const ground = makeGround(cfg, envMap);
  const { light, bulb } = makeLamp(cfg);
  scene.add(dragon, ground, light, light.target, bulb);

  const mats = { classic: makeJadeMaterial({ mode: "classic", classic, cfg, shared, envMap }) };
  const latents = {};
  const loaded = new Set();
  async function loadWidth(w) {
    if (loaded.has(w)) return;
    const [weights, lat] = await Promise.all([json(`data/weights_${w}.json`), bytes(`data/latents_${w}.bin`)]);
    mats[`n${w}`] = makeJadeMaterial({ mode: "neural", weights, cfg, shared, envMap });
    latents[w] = lat;
    loaded.add(w);
  }

  // reference side: blurred room behind the Mitsuba image (alpha = geometry)
  const bgScene = new THREE.Scene();
  setBackground(bgScene);
  const refMat = new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false, toneMapped: false });
  const refScene = new THREE.Scene();
  refScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), refMat));
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const refTex = {};
  const loadRef = (i) => (refTex[i] ??= new EXRLoader().loadAsync(`data/refs/view${i}.exr`).then((t) => { refTex[i] = t; return t; }));

  // --- state ------------------------------------------------------------------------------------
  const state = { split: 0.5, sides: { left: "classic", right: "neural" }, width: 32, wantWidth: 32,
                  lamp: { az_deg: cfg.ref_poses[0].lamp_az_deg, el_deg: cfg.ref_poses[0].lamp_el_deg },
                  refIdx: null, move: "camera" };
  let size = 0, tween = null, benching = false;
  await loadWidth(32);
  setJadeAttributes(geometry, latents[32], thickness);

  function setLamp(l) {
    state.lamp = clampLamp(l, cfg);
    const p = lampPosition(state.lamp, cfg);
    shared.uLamp.value.set(...p);
    light.position.set(...p);
    bulb.position.set(...p);
  }
  setLamp(state.lamp);

  function setWidth(w) {
    state.wantWidth = w;
    loadWidth(w).then(() => {
      if (state.wantWidth !== w) return;
      state.width = effectiveWidth(w, loaded, state.width);
      setJadeAttributes(geometry, latents[state.width], thickness);
      markWidth();
    });
  }

  const camAz = () => { const d = camera.position.clone().sub(target); return Math.atan2(d.x, d.z) * RAD; };
  const leaveReference = () => {
    const next = leaveRef(state.sides);
    for (const side of ["left", "right"]) if (next[side] !== state.sides[side]) setSide(side, next[side]);
  };

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
      const i = nearestRefView(dir, cfg.ref_poses), pose = cfg.ref_poses[i];
      state.refIdx = i;
      loadRef(i);
      setLamp({ az_deg: pose.lamp_az_deg, el_deg: pose.lamp_el_deg });
      tween = { from: camera.position.clone().sub(target), to: posOf(pose).sub(target), t0: performance.now() };
    } else if (!Object.values(state.sides).includes("ref")) {
      state.refIdx = null;
    }
  }
  const q = new URLSearchParams(location.search);
  const qp = Number.parseInt(q.get("pose"), 10);  // deep link: start at a reference pose (camera + lamp)
  if (qp >= 0 && qp < cfg.ref_poses.length) {
    const pose = cfg.ref_poses[qp];
    camera.position.copy(posOf(pose));
    controls.update();
    setLamp({ az_deg: pose.lamp_az_deg, el_deg: pose.lamp_el_deg });
  }
  for (const side of ["left", "right"]) if (Object.hasOwn(LABELS, q.get(side) ?? "")) setSide(side, q.get(side));
  if (q.has("split") && Number.isFinite(+q.get("split"))) state.split = THREE.MathUtils.clamp(+q.get("split"), 0, 1);
  if (q.has("lamp")) { const [az, el] = q.get("lamp").split(",").map(Number); if (Number.isFinite(az) && Number.isFinite(el)) setLamp({ az_deg: az, el_deg: el }); }

  // camera gestures: a real move (not a tap) leaves the reference pose
  let orbiting = false;
  controls.addEventListener("start", () => { tween = null; orbiting = true; });
  controls.addEventListener("end", () => { orbiting = false; });
  controls.addEventListener("change", () => { if (orbiting) leaveReference(); });

  // lamp gestures
  const canvas = renderer.domElement;
  let lampDrag = null;
  canvas.addEventListener("pointerdown", (e) => {
    if (state.move !== "lamp") return;
    lampDrag = { x: e.clientX, y: e.clientY, l: { ...state.lamp } };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!lampDrag) return;
    setLamp({ az_deg: lampDrag.l.az_deg - (e.clientX - lampDrag.x) * 0.5, el_deg: lampDrag.l.el_deg + (e.clientY - lampDrag.y) * 0.3 });
    leaveReference();
  });
  const endLampDrag = () => { lampDrag = null; };
  canvas.addEventListener("pointerup", endLampDrag);
  canvas.addEventListener("pointercancel", endLampDrag);
  const markMove = () => { $("moveCam").classList.toggle("on", state.move === "camera"); $("moveLamp").classList.toggle("on", state.move === "lamp"); };
  $("moveCam").onclick = () => { state.move = "camera"; controls.enabled = true; markMove(); };
  $("moveLamp").onclick = () => { state.move = "lamp"; controls.enabled = false; markMove(); };
  $("backlight").onclick = () => { setLamp(backlight(camAz())); leaveReference(); };

  $("widths").innerHTML = cfg.widths.map((w) => `<button data-w="${w}">${w}</button>`).join("");
  const markWidth = () => $("widths").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.w === state.width));
  $("widths").onclick = (e) => { if (e.target.dataset.w) setWidth(+e.target.dataset.w); };
  markWidth();

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
  handle.addEventListener("pointercancel", () => { dragging = false; });
  handle.style.left = `${state.split * 100}%`;

  // --- rendering --------------------------------------------------------------------------------
  const timers = { left: gpuTimer(gl), right: gpuTimer(gl) };
  const shownMode = (side) => sideMode(state.sides[side], state.refIdx !== null && refTex[state.refIdx] instanceof THREE.Texture);
  const pr = () => renderer.getPixelRatio();

  function renderSide(target, side, x, w) {
    if (target) {
      target.scissor.set(x * pr(), 0, w * pr(), size * pr());
      target.scissorTest = true;
      renderer.setRenderTarget(target);
    } else {
      renderer.setRenderTarget(null);
      renderer.setScissor(x, 0, w, size);
      renderer.setScissorTest(true);
    }
    renderer.clear();
    const mode = shownMode(side);
    timers[side]?.begin();
    if (mode === "ref") {
      refMat.map = refTex[state.refIdx];
      refMat.needsUpdate = true;
      renderer.render(bgScene, camera);
      renderer.render(refScene, ortho);
    } else {
      dragon.material = mode === "classic" ? mats.classic : mats[`n${state.width}`];
      renderer.render(scene, camera);
    }
    timers[side]?.end();
  }
  function renderCurtain(target) {
    const xs = Math.round(state.split * size);
    renderSide(target, "left", 0, xs);
    renderSide(target, "right", xs, size - xs);
    if (target) target.scissorTest = false;
    renderer.setScissorTest(false);
  }

  class CurtainPass extends Pass {
    constructor() { super(); this.needsSwap = false; }
    render(r, writeBuffer, readBuffer) { renderCurtain(this.renderToScreen ? null : readBuffer); }
  }
  let composer = null, bloom = null;
  if (pipeline === "hdr") {
    composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType }));
    bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.12, 0.3, 3.0);
    composer.addPass(new CurtainPass());
    composer.addPass(new ShaderPass(ClampShader));
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }
  const drawFrame = () => (composer ? composer.render() : renderCurtain(null));

  function resize() {
    size = Math.max(240, Math.floor(Math.min(innerWidth, innerHeight - $("controls").offsetHeight)));
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(size, size);
    composer?.setPixelRatio(Math.min(devicePixelRatio, 2));
    composer?.setSize(size, size);
    $("stage").style.width = `${size}px`;
  }
  addEventListener("resize", resize);
  resize();

  const dbg = gl.getExtension("WEBGL_debug_renderer_info");
  let device = (dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : "").replace(/^ANGLE \(|\)$/g, "").slice(0, 48);
  navigator.userAgentData?.getHighEntropyValues?.(["model"]).then((v) => { if (v.model) device = v.model; }).catch(() => {});
  const label = (side) => {
    const want = state.sides[side], mode = shownMode(side);
    if (want === "ref" && mode !== "ref") return "Path tracé (chargement…)";
    return mode === "neural" ? `Neural ${state.width}` : LABELS[mode];
  };

  let last = performance.now(), frameMs = NaN;
  renderer.setAnimationLoop((now) => {
    if (benching) { last = now; return; }
    frameMs = Number.isFinite(frameMs) ? 0.9 * frameMs + 0.1 * (now - last) : now - last;
    last = now;
    if (tween) {
      const k = Math.min((now - tween.t0) / 400, 1), e = k * k * (3 - 2 * k);
      const dir = tween.from.clone().normalize().lerp(tween.to.clone().normalize(), e).normalize();
      camera.position.copy(target).addScaledVector(dir, THREE.MathUtils.lerp(tween.from.length(), tween.to.length(), e));
      controls.update();
      if (k === 1) tween = null;
    }
    drawFrame();
    $("hud").textContent = formatHud({
      leftLabel: label("left"), leftMs: timers.left?.poll() ?? null,
      rightLabel: label("right"), rightMs: timers.right?.poll() ?? null,
      frameMs, device: pipeline === "ldr" ? `${device} · sans halo` : device,
    });
  });

  // --- benchmark (fixed pose: reference pose 0, camera and lamp) --------------------------------
  $("measure").onclick = async () => {
    const btn = $("measure"), out = $("bench");
    btn.disabled = true;
    benching = true;
    const saved = { sides: { ...state.sides }, lamp: { ...state.lamp }, width: state.width };
    const p0 = cfg.ref_poses[0];
    setLamp({ az_deg: p0.lamp_az_deg, el_deg: p0.lamp_el_deg });
    const rows = [];
    try {
      await withPose(camera, posOf(p0), target, async () => {
        const techniques = [["Classique", "classic", null], ...cfg.widths.map((w) => [`Neural ${w}`, "neural", w])];
        for (const [name, mode, w] of techniques) {
          out.textContent = `mesure : ${name}…`;
          if (w) { await loadWidth(w); state.width = w; setJadeAttributes(geometry, latents[w], thickness); }
          dragon.material = mode === "classic" ? mats.classic : mats[`n${w}`];
          const shade = await bench(renderer, () => { renderer.clear(); renderer.render(scene, camera); });
          state.sides = { left: mode, right: mode };
          const frame = await benchFrame(renderer, drawFrame);
          rows.push({ technique: name, "shading ms (1024²)": +shade.toFixed(3), "image complète ms": +frame.toFixed(3) });
        }
      });
    } finally {
      state.sides = saved.sides;
      state.width = saved.width;
      setJadeAttributes(geometry, latents[state.width], thickness);
      setLamp(saved.lamp);
      benching = false;
      btn.disabled = false;
    }
    out.textContent = `${device}\n` + rows.map((r) =>
      `${r.technique.padEnd(10)} shading ${r["shading ms (1024²)"].toFixed(2)} ms · image ${r["image complète ms"].toFixed(2)} ms`).join("\n");
    console.table(rows);
  };
}
