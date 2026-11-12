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

import Stats from "three/addons/libs/stats.module.js";
import { makeGround, makeJadeMaterial, makeLamp, setJadeAttributes } from "./materials.js";
import { backlight, clampLamp, lampPosition } from "./lamp.js";
import { leaveRef, nearestRefView, orbitLimits, viewDir, withPose } from "./refs.js";
import { parseDeepLink, pickPipeline, pipelineSettings, sideMode } from "./policy.js";
import { costRows } from "./hud.js";
import { bench, benchFrame } from "./bench.js";
import { curtainHandle, markOn, setText, showError } from "./ui.js";

// Sensor saturation before bloom: the bulb's radiance (thousands) would otherwise wash the whole
// frame white. Applied to the composed frame, so both curtain sides get the same treatment.
const ClampShader = {
  uniforms: { tDiffuse: { value: null }, uMax: { value: 10.0 } },
  vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "uniform sampler2D tDiffuse; uniform float uMax; varying vec2 vUv; void main() { vec4 c = texture2D(tDiffuse, vUv); gl_FragColor = vec4(min(c.rgb, vec3(uMax)), c.a); }",
};

const $ = (id) => document.getElementById(id);
const json = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const bytes = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.arrayBuffer(); })
  .then((b) => new Uint8Array(b));
const LABELS = { classic: "Classique", neural: "Neural", ref: "Path tracé" };
const RAD = 180 / Math.PI;

try {
  await main();
} catch (e) {
  showError(e);
  throw e;
}

async function main() {
  const cfg = await json("data/config.json");
  const [envSH, classic, geometry, thickness, netNormals] = await Promise.all([
    json("data/env_sh.json"), json("data/classic.json"), new PLYLoader().loadAsync("data/dragon.ply"),
    bytes("data/thickness.bin"), bytes("data/normals_net.bin"),
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

  const settings = pipelineSettings(pipeline);
  const envMap = settings.useEnvMap ? new THREE.PMREMGenerator(renderer).fromEquirectangular(hdr).texture : null;
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
  const setBackground = (s) => { s.background = hdr; s.backgroundBlurriness = settings.backgroundBlurriness; s.backgroundIntensity = cfg.env.scale; };
  setBackground(scene);
  const dragon = new THREE.Mesh(geometry);
  dragon.castShadow = true;
  const ground = makeGround(cfg, envMap);
  const { light, bulb } = makeLamp(cfg);
  scene.add(dragon, ground, light, light.target, bulb);

  const mats = { classic: makeJadeMaterial({ mode: "classic", classic, cfg, shared, envMap }) };
  const latents = {};
  const loading = {};
  const loadWidth = (w) => (loading[w] ??= (async () => {
    const [weights, lat] = await Promise.all([json(`data/weights_${w}.json`), bytes(`data/latents_${w}.bin`)]);
    mats[`n${w}`] = makeJadeMaterial({ mode: "neural", weights, cfg, shared, envMap });
    latents[w] = lat;
  })().catch((e) => { delete loading[w]; throw e; })); // a failed download can be retried

  // reference side: blurred room behind the Mitsuba image (alpha = geometry)
  const bgScene = new THREE.Scene();
  setBackground(bgScene);
  const refMat = new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false });
  const refScene = new THREE.Scene();
  refScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), refMat));
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const refTex = {};
  const loadRef = (i) => (refTex[i] ??= new EXRLoader().loadAsync(`data/refs/view${i}.exr`)
    .then((t) => { refTex[i] = t; return t; })
    .catch((e) => { delete refTex[i]; showError(e); })); // a failed download can be retried

  // --- state ------------------------------------------------------------------------------------
  const state = { split: 0.5, sides: { left: "classic", right: "neural" }, width: 16, wantWidth: 16,  // 16: same quality as 32 at a quarter of the cost
                  lamp: { az_deg: cfg.ref_poses[0].lamp_az_deg, el_deg: cfg.ref_poses[0].lamp_el_deg },
                  refIdx: null, move: "camera" };
  let size = 0, tween = null, benching = false;
  await loadWidth(16);
  geometry.setAttribute("thick", new THREE.BufferAttribute(thickness, 1, true));
  // the network's normal: vertex normals smoothed over ~5 mm (pipeline/features.py::smooth_normals)
  geometry.setAttribute("nNet", new THREE.BufferAttribute(new Int8Array(netNormals.buffer), 3, true));
  setJadeAttributes(geometry, latents[16]);

  function setLamp(l) {
    state.lamp = clampLamp(l, cfg);
    const p = lampPosition(state.lamp, cfg);
    shared.uLamp.value.set(...p);
    light.position.set(...p);
    bulb.position.set(...p);
  }
  setLamp(state.lamp);

  function setWidth(w) {
    if (benching) return;
    state.wantWidth = w;
    loadWidth(w).then(() => {
      if (state.wantWidth !== w || benching) return; // a later click, or a measurement, took over
      state.width = w;
      setJadeAttributes(geometry, latents[w]);
      markWidth();
    }, showError);
  }

  const camAz = () => { const d = camera.position.clone().sub(target); return Math.atan2(d.x, d.z) * RAD; };
  const leaveReference = () => {
    const next = leaveRef(state.sides);
    for (const side of ["left", "right"]) if (next[side] !== state.sides[side]) setSide(side, next[side]);
  };

  for (const side of ["left", "right"]) {
    $(side).innerHTML = Object.entries(LABELS).map(([k, v]) => `<button data-mode="${k}">${v}</button>`).join("");
    $(side).onclick = (e) => { if (e.target.dataset.mode) setSide(side, e.target.dataset.mode); };
  }
  const markSides = () => {
    for (const side of ["left", "right"]) $(side).querySelectorAll("button").forEach((b) => markOn(b, b.dataset.mode === state.sides[side]));
  };
  markSides();
  function setSide(side, mode) {
    if (benching) return;
    state.sides[side] = mode;
    markSides();
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
  const link = parseDeepLink(location.search, cfg);  // order: pose, lamp, sides (a ref side re-snaps the lamp)
  if (link.pose !== null) {
    const pose = cfg.ref_poses[link.pose];
    camera.position.copy(posOf(pose));
    controls.update();
    setLamp({ az_deg: pose.lamp_az_deg, el_deg: pose.lamp_el_deg });
  }
  if (link.lamp) setLamp(link.lamp);
  for (const [side, mode] of Object.entries(link.sides)) setSide(side, mode);
  if (link.split !== null) state.split = link.split;

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
  const markMove = () => { markOn($("moveCam"), state.move === "camera"); markOn($("moveLamp"), state.move === "lamp"); };
  $("moveCam").onclick = () => { state.move = "camera"; controls.enabled = true; markMove(); };
  $("moveLamp").onclick = () => { state.move = "lamp"; controls.enabled = false; markMove(); };
  $("backlight").onclick = () => { setLamp(backlight(camAz())); leaveReference(); };

  $("widths").innerHTML = cfg.widths.map((w) => `<button data-w="${w}">${w}</button>`).join("");
  const markWidth = () => $("widths").querySelectorAll("button").forEach((b) => markOn(b, +b.dataset.w === state.width));
  $("widths").onclick = (e) => { if (e.target.dataset.w) setWidth(+e.target.dataset.w); };
  markWidth();
  markMove();

  const placeSplit = curtainHandle($("handle"), $("stage"), () => state.split, (s) => { state.split = s; });
  placeSplit();

  // --- rendering --------------------------------------------------------------------------------
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
    if (mode === "ref") {
      const t = refTex[state.refIdx];
      if (refMat.map !== t) { refMat.map = t; refMat.needsUpdate = true; }
      renderer.render(bgScene, camera);
      renderer.render(refScene, ortho);
    } else {
      dragon.material = mode === "classic" ? mats.classic : mats[`n${state.width}`];
      renderer.render(scene, camera);
    }
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
    composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType, samples: Math.min(4, renderer.capabilities.maxSamples) })); // MSAA: antialias:true only covers the screen pass
    bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.12, 0.3, 3.0);
    composer.addPass(new CurtainPass());
    composer.addPass(new ShaderPass(ClampShader));
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }
  const drawFrame = () => (composer ? composer.render() : renderCurtain(null));

  const desktop = matchMedia("(min-width: 900px)");
  if (desktop.matches) $("more").open = true;
  const stats = new Stats();
  stats.dom.classList.add("stats");
  $("stage").append(stats.dom);
  function resize() {
    // square canvas: the path-traced references are square renders with the same vertical FOV
    size = Math.max(240, Math.floor(desktop.matches ? Math.min($("view").clientWidth, innerHeight) : Math.min(innerWidth, innerHeight * 0.62)));
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

  renderer.setAnimationLoop((now) => {
    if (benching) return;
    if (tween) {
      const k = Math.min((now - tween.t0) / 400, 1), e = k * k * (3 - 2 * k);
      const dir = tween.from.clone().normalize().lerp(tween.to.clone().normalize(), e).normalize();
      camera.position.copy(target).addScaledVector(dir, THREE.MathUtils.lerp(tween.from.length(), tween.to.length(), e));
      controls.update();
      if (k === 1) tween = null;
    }
    drawFrame();
    stats.update();
    setText($("tagL"), label("left"));
    setText($("tagR"), label("right"));
  });

  const costLabels = ["Classique", ...cfg.widths.map((w) => `Neural ${w}`)];
  const renderCost = (results) => {
    $("cost").innerHTML = costRows(results, costLabels).map((r) =>
      `<span class="name">${r.label}</span><div class="bar"><i class="${r.over ? "over" : ""}" style="width:${r.pct}%"></i></div>` +
      `<span class="val">${r.value}</span>${r.detail ? `<span class="detail">${r.detail}</span>` : ""}`).join("");
  };
  renderCost(null);
  if (pipeline === "ldr") $("device").textContent = "Ce GPU n'a pas de rendu flottant : affichage sans halo.";

  // --- benchmark (fixed pose: reference pose 0, camera and lamp) --------------------------------
  $("measure").onclick = async () => {
    const btn = $("measure"), out = $("device");
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
          if (w) { await loadWidth(w); state.width = w; setJadeAttributes(geometry, latents[w]); }
          dragon.material = mode === "classic" ? mats.classic : mats[`n${w}`];
          const shade = await bench(renderer, () => { renderer.clear(); renderer.render(scene, camera); });
          state.sides = { left: mode, right: mode };
          const frame = await benchFrame(renderer, drawFrame);
          rows.push({ technique: name, frameMs: frame, shadeMs: shade });
        }
      });
    } finally {
      state.sides = saved.sides;
      state.width = saved.width;
      setJadeAttributes(geometry, latents[state.width]);
      setLamp(saved.lamp);
      markSides();
      markWidth();
      benching = false;
      btn.disabled = false;
    }
    renderCost(rows);
    out.textContent = `Mesuré sur ${device || "cet appareil"} · image ${Math.round(size * pr())} px, halo compris${pipeline === "ldr" ? " (sans halo)" : ""}.`;
    console.table(rows);
  };
}
