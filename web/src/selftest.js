// GLSL parity check: features() + neuralSSS() on the GPU vs pipeline/train.py forward_numpy
// on 64 validation samples (data/selftest.json, width-16 network).
import * as THREE from "three";
import { FEATURES_GLSL } from "./glsl.js";
import { mlpGlsl } from "./mlp.js";

const out = document.getElementById("out");
const say = (s, cls) => { out.textContent = s; out.className = cls; };
const json = (p) => fetch(p).then((r) => r.json());

const [cfg, w16, fx] = await Promise.all([json("data/config.json"), json("data/weights_16.json"), json("data/selftest.json")]);
const renderer = new THREE.WebGLRenderer();
const caps = renderer.capabilities;
const info = `précision fragment : ${caps.precision}`;

if (!renderer.extensions.has("EXT_color_buffer_float")) {
  say(`UNSUPPORTED — EXT_color_buffer_float absent, lecture float impossible\n${info}`, "warn");
} else if (caps.precision !== "highp") {
  say(`UNSUPPORTED — pas de highp en fragment, le réseau sera imprécis\n${info}`, "warn");
} else {
  const N = fx.length;
  const data = new Float32Array(4 * 4 * N);
  fx.forEach((s, i) => [s.pos, s.nrm, s.cam, s.flame_offset].forEach((v, k) => data.set(v, (i * 4 + k) * 4)));
  const tex = new THREE.DataTexture(data, 4, N, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({
    uniforms: { uData: { value: tex }, uFlameBase: { value: new THREE.Vector3(...cfg.flame.base) } },
    vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: `
precision highp float;
precision highp sampler2D;
uniform sampler2D uData;
uniform vec3 uFlameBase;
${FEATURES_GLSL}
${mlpGlsl(w16)}
void main() {
  int i = int(gl_FragCoord.x);
  vec3 pos = texelFetch(uData, ivec2(0, i), 0).xyz;
  vec3 nrm = texelFetch(uData, ivec2(1, i), 0).xyz;
  vec3 cam = texelFetch(uData, ivec2(2, i), 0).xyz;
  vec3 off = texelFetch(uData, ivec2(3, i), 0).xyz;
  float x[17];
  features(pos, nrm, cam, off, uFlameBase, x);
  gl_FragColor = vec4(neuralSSS(x), 1.0);
}`,
    toneMapped: false,
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
  const rt = new THREE.WebGLRenderTarget(N, 1, { type: THREE.FloatType });
  renderer.setRenderTarget(rt);
  renderer.render(scene, new THREE.Camera());
  const px = new Float32Array(4 * N);
  renderer.readRenderTargetPixels(rt, 0, 0, N, 1, px);
  let ok = 0, worst = 0;
  fx.forEach((s, i) => {
    const err = Math.max(...s.expected.map((e, c) => Math.abs(px[i * 4 + c] - e) / Math.max(Math.abs(e), 1e-6)));
    worst = Math.max(worst, err);
    if (err <= 1e-3) ok++;
  });
  const msg = `${ok === N ? "PASS" : "FAIL"} ${ok}/${N} — pire écart relatif ${worst.toExponential(2)}\n${info}`;
  say(msg, ok === N ? "pass" : "fail");
}
