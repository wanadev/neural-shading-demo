// Real-time screen-space AO for the labo page: classic SSAO (fitted) vs the network trained in
// spikes/ao/ao_spike.py. Both read the same G-buffer (world positions + normals) at the same 16
// screen offsets; features mirror ss_features() in the spike, with the image y axis flipped
// (numpy rows go down, GL rows go up).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import Stats from "three/addons/libs/stats.module.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { benchFrame } from "./bench.js";
import { orbitLimits, viewDir } from "./refs.js";

const TEXW = 256;

// layers [{W:[out][in], b}] -> vec4 texture, each row padded to a multiple of 4
export function packNet(layers) {
  const vecs = [], layout = [];
  const push = (arr) => { const s = vecs.length; for (let i = 0; i < arr.length; i += 4) vecs.push([arr[i] ?? 0, arr[i + 1] ?? 0, arr[i + 2] ?? 0, arr[i + 3] ?? 0]); return s; };
  for (const l of layers) {
    const din = l.W[0].length, w = vecs.length;
    l.W.forEach((row) => push(row));
    layout.push({ w, b: push(l.b), dinV: Math.ceil(din / 4), dout: l.b.length });
  }
  const height = Math.ceil(vecs.length / TEXW), data = new Float32Array(height * TEXW * 4);
  vecs.forEach((v, i) => data.set(v, i * 4));
  return { data, width: TEXW, height, layout };
}

function netGlsl(layout) {
  const hidden = layout.slice(0, -1).map((l, k) => `
  vec4 h${k + 1}[${l.dout / 4}];
  for (int v = 0; v < ${l.dout / 4}; v++) h${k + 1}[v] = vec4(0.0);
  for (int j = 0; j < ${l.dout}; j++) {
    float s = comp(netv(${l.b} + j / 4), j);
    for (int v = 0; v < ${l.dinV}; v++) s += dot(netv(${l.w} + j * ${l.dinV} + v), h${k}[v]);
    h${k + 1}[j / 4] += lane(j) * max(s, 0.0);
  }`).join("");
  const o = layout.at(-1), last = layout.length - 1;
  return `
uniform highp sampler2D uNet;
vec4 netv(int i) { return texelFetch(uNet, ivec2(i % ${TEXW}, i / ${TEXW}), 0); }
vec4 lane(int j) { return vec4(equal(ivec4(0, 1, 2, 3), ivec4(j % 4))); }
float comp(vec4 v, int j) { return dot(v, lane(j)); }
float neuralAO(vec4 h0[${layout[0].dinV}]) {${hidden}
  float s = comp(netv(${o.b}), 0);
  for (int v = 0; v < ${o.dinV}; v++) s += dot(netv(${o.w} + v), h${last}[v]);
  return 1.0 / (1.0 + exp(-s));
}`;
}

function aoMaterial(net, mode, uniforms) {
  const K = net.K, packed = packNet(net.layers);
  const tex = new THREE.DataTexture(packed.data, packed.width, packed.height, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  const pat = net.pattern.map(([x, y]) => `vec2(${x.toFixed(7)}, ${y.toFixed(7)})`).join(", ");
  return new THREE.ShaderMaterial({
    uniforms: { ...uniforms, uNet: { value: tex }, uS: { value: net.classic.s }, uBias: { value: net.classic.bias } },
    defines: mode === "neural" ? { NEURAL: "" } : {},
    vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tPos, tNrm;
uniform vec3 uEye, uFwd;
uniform float uFocal, uR, uS, uBias;
${mode === "neural" ? netGlsl(packed.layout) : ""}
const vec2 PAT[${K}] = vec2[${K}](${pat});
void main() {
  ivec2 size = textureSize(tPos, 0), ip = ivec2(gl_FragCoord.xy);
  vec4 P = texelFetch(tPos, ip, 0);
  if (P.a < 0.5) { gl_FragColor = vec4(1.0); return; }
  vec3 n = normalize(texelFetch(tNrm, ip, 0).xyz);
  float rad = uR * uFocal / max(dot(P.xyz - uEye, uFwd), 1e-3);
  float ign = fract(52.9829189 * fract(0.06711056 * float(ip.x) + 0.00583715 * float(ip.y)));
  float ca = cos(6.28318531 * ign), sa = sin(6.28318531 * ign);
#ifdef NEURAL
  float f[${2 * K + 4}];
#else
  float acc = 0.0;
#endif
  for (int k = 0; k < ${K}; k++) {
    vec2 pk = PAT[k];
    vec2 o = vec2(ca * pk.x - sa * pk.y, sa * pk.x + ca * pk.y) * rad;
    ivec2 sp = clamp(ip + ivec2(int(round(o.x)), -int(round(o.y))), ivec2(0), size - 1);  // numpy rows go down
    vec4 Q = texelFetch(tPos, sp, 0);
    bool valid = Q.a > 0.5;
    vec3 d = Q.xyz - P.xyz;
    float dist = length(d);
#ifdef NEURAL
    f[2 * k] = valid ? clamp(dot(n, d) / uR, -4.0, 4.0) : 0.0;
    f[2 * k + 1] = valid ? clamp(dist / uR, 0.0, 4.0) : 4.0;
#else
    if (valid) acc += max(dot(n, d) / max(dist, 1e-5) - uBias, 0.0) * max(1.0 - (dist / uR) * (dist / uR), 0.0);
#endif
  }
#ifdef NEURAL
  f[${2 * K}] = n.x; f[${2 * K + 1}] = n.y; f[${2 * K + 2}] = n.z; f[${2 * K + 3}] = 0.0;
  vec4 h0[${(2 * K + 4) / 4}];
  for (int v = 0; v < ${(2 * K + 4) / 4}; v++) h0[v] = vec4(f[4 * v], f[4 * v + 1], f[4 * v + 2], f[4 * v + 3]);
  float ao = neuralAO(h0);
#else
  float ao = clamp(1.0 - uS * acc / ${K}.0, 0.0, 1.0);
#endif
  gl_FragColor = vec4(vec3(ao), 1.0);
}`,
  });
}

export async function createAo({ canvas, net, cfg, gtaoParams }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.autoClear = false;
  const floatRT = renderer.extensions.has("EXT_color_buffer_float") ? THREE.FloatType : THREE.HalfFloatType;
  const geometry = await new PLYLoader().loadAsync("data/dragon.ply");
  const camera = new THREE.PerspectiveCamera(cfg.camera.fov_y, 1, 0.01, 20);
  const target = new THREE.Vector3(...cfg.camera.target);
  const posOf = (v) => target.clone().addScaledVector(new THREE.Vector3(...viewDir(v)), v.radius);
  camera.position.copy(posOf(net.views[0]));
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(target);
  Object.assign(controls, orbitLimits(cfg), { enablePan: false, enableDamping: false });
  controls.update();

  // G-buffer: world position (alpha = geometry) and normal, one render each
  const gmat = (what) => new THREE.ShaderMaterial({
    vertexShader: "varying vec3 vP; varying vec3 vN; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w; }",
    fragmentShader: `precision highp float; varying vec3 vP; varying vec3 vN; void main() { gl_FragColor = vec4(${what}, 1.0); }`,
  });
  const scene = new THREE.Scene();
  const dragon = new THREE.Mesh(geometry);
  const s = cfg.ground.size;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(s, s).rotateX(-Math.PI / 2));
  scene.add(dragon, ground);
  const mPos = gmat("vP"), mNrm = gmat("normalize(vN)");

  let rtPos, rtNrm, rtAo, rtBlurA, rtBlurB, aoScale = 0.5;
  const uniforms = { tPos: { value: null }, tNrm: { value: null }, uEye: { value: new THREE.Vector3() },
    uFwd: { value: new THREE.Vector3() }, uFocal: { value: 1 }, uR: { value: net.R } };
  const mats = { classic: aoMaterial(net, "classic", uniforms), neural: aoMaterial(net, "neural", uniforms) };
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
  const quadScene = new THREE.Scene(); quadScene.add(quad);
  const ortho = new THREE.Camera();

  // depth- and normal-aware separable blur: removes the rotation noise, keeps edges
  const blurMat = new THREE.ShaderMaterial({
    uniforms: { tAo: { value: null }, tPos: uniforms.tPos, tNrm: uniforms.tNrm, uDir: { value: new THREE.Vector2(1, 0) }, uR: uniforms.uR },
    vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tAo, tPos, tNrm;
uniform vec2 uDir;
uniform float uR;
void main() {
  ivec2 size = textureSize(tPos, 0), ip = ivec2(gl_FragCoord.xy);
  vec4 P = texelFetch(tPos, ip, 0);
  if (P.a < 0.5) { gl_FragColor = vec4(1.0); return; }
  vec3 n = texelFetch(tNrm, ip, 0).xyz;
  float sum = 0.0, wsum = 0.0;
  for (int i = -4; i <= 4; i++) {
    ivec2 q = clamp(ip + ivec2(uDir) * i, ivec2(0), size - 1);
    vec4 Q = texelFetch(tPos, q, 0);
    if (Q.a < 0.5) continue;
    vec3 m = texelFetch(tNrm, q, 0).xyz;
    float dz = length(Q.xyz - P.xyz) / (0.25 * uR);
    float w = exp(-float(i * i) / 8.0) * pow(max(dot(n, m), 0.0), 8.0) * exp(-dz * dz);
    sum += texelFetch(tAo, q, 0).r * w; wsum += w;
  }
  gl_FragColor = vec4(vec3(sum / max(wsum, 1e-5)), 1.0);
}`,
  });

  // final pass, full resolution: the scene itself, shaded as clay x AO (or AO alone)
  const compMat = new THREE.ShaderMaterial({
    uniforms: { tAo: { value: null }, uScreen: { value: new THREE.Vector2(1, 1) }, uApplied: { value: 1 } },
    vertexShader: "varying vec3 vN; void main() { vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0); }",
    fragmentShader: /* glsl */ `
precision highp float;
uniform sampler2D tAo;
uniform vec2 uScreen;
uniform float uApplied;
varying vec3 vN;
void main() {
  float ao = texture2D(tAo, gl_FragCoord.xy / uScreen).r;
  vec3 n = normalize(vN);
  float hemi = mix(0.45, 1.0, n.y * 0.5 + 0.5);
  float key = max(dot(n, normalize(vec3(0.4, 0.8, 0.3))), 0.0);
  vec3 clay = vec3(0.86, 0.84, 0.80) * (0.75 * hemi + 0.35 * key);
  gl_FragColor = vec4(mix(vec3(ao), clay * ao, uApplied), 1.0);
}`,
  });
  const gtTex = net.views.map((_, i) => new THREE.TextureLoader().load(`data/labo/ao/view${i}-gt.jpg`));

  // state of the art in screen space: three.js GTAO with its own Poisson denoise, tuned on the
  // same reference poses (labo/gtao.json, written by the tuning run)
  const gtao = new GTAOPass(scene, camera, 64, 64);
  gtao.output = GTAOPass.OUTPUT.Off;  // we only read gtao.gtaoMap
  gtao.setSceneClipBox(new THREE.Box3().setFromObject(scene));
  const setGtaoParams = (p) => gtao.updateGtaoMaterial({ radius: net.R, samples: 16, ...p });
  setGtaoParams(gtaoParams?.params ?? {});

  let size = 0;
  function allocate() {
    const px = Math.max(64, Math.round(size * renderer.getPixelRatio() * aoScale));
    for (const rt of [rtPos, rtNrm, rtAo, rtBlurA, rtBlurB]) rt?.dispose();
    const mk = (type, filter) => new THREE.WebGLRenderTarget(px, px, { type, minFilter: filter, magFilter: filter, depthBuffer: type !== THREE.UnsignedByteType });
    rtPos = mk(floatRT, THREE.NearestFilter); rtNrm = mk(floatRT, THREE.NearestFilter); rtAo = mk(THREE.UnsignedByteType, THREE.NearestFilter);
    rtBlurA = mk(THREE.UnsignedByteType, THREE.NearestFilter); rtBlurB = mk(THREE.UnsignedByteType, THREE.LinearFilter);
    uniforms.tPos.value = rtPos.texture; uniforms.tNrm.value = rtNrm.texture;
    uniforms.uFocal.value = (px / 2) / Math.tan(THREE.MathUtils.degToRad(cfg.camera.fov_y) / 2);
    gtao.setSize(px, px);
  }
  function resize(cssSize, dpr = Math.min(devicePixelRatio, 2)) {
    size = cssSize;
    renderer.setPixelRatio(dpr);
    renderer.setSize(size, size);
    allocate();
  }

  const state = { left: "classic", right: "neural", split: 0.5, view: 0, applied: true };
  function render() {
    const fwd = target.clone().sub(camera.position).normalize();
    uniforms.uEye.value.copy(camera.position); uniforms.uFwd.value.copy(fwd);
    renderer.setClearColor(0x000000, 0);
    for (const [rt, m] of [[rtPos, mPos], [rtNrm, mNrm]]) {
      scene.overrideMaterial = m;
      renderer.setRenderTarget(rt); renderer.clear(); renderer.render(scene, camera);
    }
    const W = rtAo.width, xs = Math.round(state.split * W);
    for (const [side, x0, w] of [["left", 0, xs], ["right", xs, W - xs]]) {
      if (state[side] === "gt" || state[side] === "gtao" || w <= 0) continue;
      rtAo.scissor.set(x0, 0, w, W); rtAo.scissorTest = true;
      renderer.setRenderTarget(rtAo);
      quad.material = mats[state[side]];
      renderer.render(quadScene, ortho);
    }
    rtAo.scissorTest = false;
    if (state.left === "gtao" || state.right === "gtao") {
      scene.overrideMaterial = null;
      gtao.render(renderer, null, null);
    }
    quad.material = blurMat;
    for (const [src, dst, dir] of [[rtAo, rtBlurA, [1, 0]], [rtBlurA, rtBlurB, [0, 1]]]) {
      blurMat.uniforms.tAo.value = src.texture; blurMat.uniforms.uDir.value.set(...dir);
      renderer.setRenderTarget(dst); renderer.render(quadScene, ortho);
    }
    renderer.setRenderTarget(null);
    renderer.setClearColor(0xffffff, 1); renderer.clear();
    const pr = renderer.getPixelRatio(), cut = Math.round(state.split * size);
    compMat.uniforms.uScreen.value.set(size * pr, size * pr);
    compMat.uniforms.uApplied.value = state.applied ? 1 : 0;
    scene.overrideMaterial = compMat;
    renderer.setScissorTest(true);
    for (const [side, x0, w] of [["left", 0, cut], ["right", cut, size - cut]]) {
      if (w <= 0) continue;
      compMat.uniforms.tAo.value = state[side] === "gt" ? gtTex.at(state.view) : state[side] === "gtao" ? gtao.gtaoMap : rtBlurB.texture;
      renderer.setScissor(x0, 0, w, size);
      renderer.render(scene, camera);
    }
    renderer.setScissorTest(false);
    scene.overrideMaterial = null;
  }

  let tween = null;
  function setPose(i, animate = true) {
    const to = posOf(net.views[i]).sub(target), from = camera.position.clone().sub(target);
    if (!animate) { camera.position.copy(target).add(to); controls.update(); return; }
    tween = { from, to, t0: performance.now() };
  }
  const stats = new Stats();
  stats.dom.classList.add("stats");
  let running = true, onMove = null;
  controls.addEventListener("start", () => { tween = null; });
  controls.addEventListener("change", () => { if (!tween) onMove?.(); });
  renderer.setAnimationLoop((now) => {
    if (!running) return;
    if (tween) {
      const k = Math.min((now - tween.t0) / 400, 1), e = k * k * (3 - 2 * k);
      const dir = tween.from.clone().normalize().lerp(tween.to.clone().normalize(), e).normalize();
      camera.position.copy(target).addScaledVector(dir, THREE.MathUtils.lerp(tween.from.length(), tween.to.length(), e));
      controls.update();
      if (k === 1) tween = null;
    }
    render();
    stats.update();
  });

  // parity check against the spike's own images: full-res neural AO at each reference pose
  async function check() {
    running = false;
    const saved = { ...state };
    resize(512, 1); aoScale = 1; allocate();
    Object.assign(state, { left: "neural", right: "neural" });
    const out = [];
    for (let i = 0; i < net.views.length; i++) {
      setPose(i, false); render();
      const px = new Uint8Array(512 * 512 * 4);
      renderer.setRenderTarget(rtAo); renderer.readRenderTargetPixels(rtAo, 0, 0, 512, 512, px); renderer.setRenderTarget(null);
      const bmp = await createImageBitmap(await (await fetch(`data/labo/ao/view${i}-neural.jpg`)).blob());
      const c = new OffscreenCanvas(512, 512).getContext("2d"); c.drawImage(bmp, 0, 0);
      const ref = c.getImageData(0, 0, 512, 512).data;
      let sum = 0, n = 0, bad = 0;
      for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) {
        const a = px[((511 - y) * 512 + x) * 4] / 255, b = ref[(y * 512 + x) * 4] / 255;  // GL rows bottom-up
        if (b > 0.995 && a > 0.995) continue;
        const e = Math.abs(a - b); sum += e; n++; if (e > 0.1) bad++;
      }
      out.push({ view: i + 1, mae: sum / Math.max(n, 1), bad: bad / Math.max(n, 1) });
    }
    Object.assign(state, saved); aoScale = 0.5; running = true;
    return out;
  }

  const copyMat = new THREE.ShaderMaterial({ uniforms: { t: { value: null } },
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: "precision highp float; uniform sampler2D t; varying vec2 vUv; void main() { gl_FragColor = vec4(vec3(texture2D(t, vUv).r), 1.0); }" });
  const gtPixels = [];
  async function gtImage(i) {
    if (!gtPixels[i]) {
      const bmp = await createImageBitmap(await (await fetch(`data/labo/ao/view${i}-gt.jpg`)).blob());
      const c = new OffscreenCanvas(512, 512).getContext("2d"); c.drawImage(bmp, 0, 0);
      gtPixels[i] = c.getImageData(0, 0, 512, 512).data;
    }
    return gtPixels[i];
  }
  async function measure(methods = ["classic", "neural", "gtao"]) {
    running = false;
    const saved = { ...state }, savedScale = aoScale;
    resize(512, 1); aoScale = 1; allocate();
    const res = methods.map(() => []);
    for (let i = 0; i < net.views.length; i++) {
      setPose(i, false);
      const gt = await gtImage(i);
      for (const [m, method] of methods.entries()) {
        Object.assign(state, { left: method, right: method });
        render();
        let rt = rtBlurB;
        if (method === "gtao") {  // half-float target: copy to 8 bits first, like the other methods
          copyMat.uniforms.t.value = gtao.gtaoMap; quad.material = copyMat;
          renderer.setRenderTarget(rtBlurA); renderer.render(quadScene, ortho); renderer.setRenderTarget(null);
          rt = rtBlurA;
        }
        const float = false;
        const px = new Uint8Array(512 * 512 * 4);
        renderer.readRenderTargetPixels(rt, 0, 0, 512, 512, px);
        let sum = 0, n = 0;
        for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) {
          const g = gt[(y * 512 + x) * 4] / 255;
          if (g >= 0.9) continue;
          const a = px[((511 - y) * 512 + x) * 4] / (float ? 1 : 255);  // GL rows bottom-up
          sum += Math.abs(a - g); n++;
        }
        res[m].push(sum / Math.max(n, 1));
      }
    }
    Object.assign(state, saved); aoScale = savedScale; running = true;
    return Object.fromEntries(methods.map((m, k) => [m, res[k]]));
  }

  // ms per full frame with both sides of the curtain on the same method (bench.js: readback forces the GPU to finish)
  async function benchMethods(methods, opts) {
    running = false;
    const saved = { ...state }, out = {};
    for (const m of methods) { Object.assign(state, { left: m, right: m }); out[m] = await benchFrame(renderer, render, opts); }
    Object.assign(state, saved); running = true;
    return out;
  }

  return {
    camera, controls, stats, state, resize, setPose, check, measure, setGtaoParams, benchMethods,
    setAoScale(v) { aoScale = v; allocate(); },
    onMove(fn) { onMove = fn; },
    pause() { running = false; }, resume() { running = true; },
    nearestPose() {
      const d = camera.position.clone().sub(target).normalize();
      let best = 0, bd = -2;
      net.views.forEach((v, i) => { const dot = d.dot(new THREE.Vector3(...viewDir(v))); if (dot > bd) { bd = dot; best = i; } });
      return best;
    },
  };
}
