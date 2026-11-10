// GPU parity check for every network width, including the per-vertex latent interpolation:
// each validation sample is drawn as a tiny triangle whose 3 vertices carry the uint8 latents of
// the 3 real dragon vertices, placed so the sample's pixel centre has exactly the sample's
// barycentrics. The GPU interpolates, dequantizes and runs the network; the result is compared
// with pipeline/train.py forward_numpy (data/selftest.json).
import * as THREE from "three";
import { FEATURES_GLSL } from "./glsl.js";
import { mlpGlsl } from "./mlp.js";
import { makeNetTexture } from "./materials.js";

const out = document.getElementById("out");
const say = (s, cls) => { out.textContent = s; out.className = cls; };
const json = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p} : HTTP ${r.status}`); return r.json(); });
const bytes = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p} : HTTP ${r.status}`); return r.arrayBuffer(); })
  .then((b) => new Uint8Array(b));
// One 320x320 px tile per sample (8 per row). Triangles are ~32 px: the rasteriser snaps vertices to a
// 1/256 px grid, so sub-pixel triangles gave ~2% barycentric (hence latent) errors. With b2 >= 0.2 a
// triangle reaches at most 5 x 32 = 160 px from its centre, never another tile's centre.
const TILE = 320, COLS = 8, A = 32;

// any failure (no WebGL2, missing file, shader compile) must end in a visible verdict, never "en cours…"
try {
  const cfg = await json("data/config.json");
  const fx = await json("data/selftest.json");
  const corrupt = new URLSearchParams(location.search).has("corrupt"); // negative control
  const renderer = new THREE.WebGLRenderer();
  if (!renderer.extensions.has("EXT_color_buffer_float")) {
    say("UNSUPPORTED — EXT_color_buffer_float absent, lecture float impossible", "warn");
  } else {
    const N = fx.samples.length;
    const lines = [];
    let allOk = true;
    for (const w of cfg.widths) {
      const [weights, lat] = await Promise.all([json(`data/weights_${w}.json`), bytes(`data/latents_${w}.bin`)]);
      if (corrupt) for (let k = 0; k < lat.length; k++) lat[k] = 255 - lat[k]; // every latent wrong
      const got = renderSamples(renderer, fx.samples, lat, weights);
      let ok = 0, worst = 0;
      fx.expected[String(w)].forEach((e, i) => {
        const g = got[i];
        const err = Math.max(...e.map((v, c) => Math.abs(g[c] - v) / Math.max(Math.abs(v), 1e-6)));
        worst = Math.max(worst, Number.isNaN(err) ? Infinity : err);
        if (err <= 2e-3) ok++;
      });
      allOk &&= ok === N;
      lines.push(`réseau ${w} : ${ok === N ? "PASS" : "FAIL"} ${ok}/${N} — pire écart ${worst.toExponential(2)}`);
    }
    say(`${allOk ? "PASS" : "FAIL"}\n${lines.join("\n")}\nprécision fragment : ${renderer.capabilities.precision}`, allOk ? "pass" : "fail");
  }
} catch (e) {
  say(`ERREUR — ${e.message}\n(WebGL2 indisponible ou fichier manquant : le test ne peut pas conclure)`, "fail");
}

function renderSamples(renderer, samples, lat, weights) {
  const W = TILE * COLS, H = TILE * Math.ceil(samples.length / COLS);
  const centre = (i) => [(i % COLS) * TILE + TILE / 2, Math.floor(i / COLS) * TILE + TILE / 2]; // pixel indices
  const n = samples.length;
  const pos = new Float32Array(n * 9), la = new Uint8Array(n * 12), lb = new Uint8Array(n * 12);
  const attr = { aNrm: new Float32Array(n * 9), aPos: new Float32Array(n * 9), aCam: new Float32Array(n * 9), aLamp: new Float32Array(n * 9) };
  samples.forEach((s, i) => {
    const [px, py] = centre(i), sx = 2 / W, sy = 2 / H;
    const cx = -1 + (px + 0.5) * sx, cy = -1 + (py + 0.5) * sy; // that pixel's centre in NDC
    const [b0, b1, b2] = s.bary, a = A * sx, ay = A * sy;
    const d = [[a, 0], [0, ay], [-(b0 * a) / b2, -(b1 * ay) / b2]]; // sum b_k d_k = 0
    for (let k = 0; k < 3; k++) {
      const v = i * 3 + k;
      pos.set([cx + d[k][0], cy + d[k][1], 0], v * 3);
      la.set(lat.subarray(s.tri_verts[k] * 8, s.tri_verts[k] * 8 + 4), v * 4);
      lb.set(lat.subarray(s.tri_verts[k] * 8 + 4, s.tri_verts[k] * 8 + 8), v * 4);
      attr.aNrm.set(s.nrm, v * 3); attr.aPos.set(s.pos, v * 3); attr.aCam.set(s.cam, v * 3); attr.aLamp.set(s.lamp, v * 3);
    }
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("latA", new THREE.BufferAttribute(la, 4, true));
  g.setAttribute("latB", new THREE.BufferAttribute(lb, 4, true));
  for (const [k, v] of Object.entries(attr)) g.setAttribute(k, new THREE.BufferAttribute(v, 3));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uLatLo: { value: weights.latent_lo }, uLatHi: { value: weights.latent_hi }, uWhich: { value: 0 },
                uNet: { value: makeNetTexture(weights) } },
    vertexShader: `
attribute vec4 latA; attribute vec4 latB; attribute vec3 aNrm; attribute vec3 aPos; attribute vec3 aCam; attribute vec3 aLamp;
varying vec4 vLatA; varying vec4 vLatB; varying vec3 vNrm; varying vec3 vPos; varying vec3 vCam; varying vec3 vLamp;
void main() {
  vLatA = latA; vLatB = latB; vNrm = aNrm; vPos = aPos; vCam = aCam; vLamp = aLamp;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`,
    fragmentShader: `
precision highp float;
uniform int uWhich;
varying vec4 vLatA; varying vec4 vLatB; varying vec3 vNrm; varying vec3 vPos; varying vec3 vCam; varying vec3 vLamp;
${FEATURES_GLSL}
${mlpGlsl(weights)}
void main() {
  float x[19];
  features(vLatA, vLatB, vNrm, vPos, vCam, vLamp, x);
  vec3 e, l;
  neuralSSS(x, e, l);
  gl_FragColor = vec4(uWhich == 0 ? e : l, 1.0);
}`,
    toneMapped: false, side: THREE.DoubleSide,
  });
  const scene = new THREE.Scene();
  const mesh = new THREE.Mesh(g, mat);
  mesh.frustumCulled = false;
  scene.add(mesh);
  const rt = new THREE.WebGLRenderTarget(W, H, { type: THREE.FloatType });
  const px = new Float32Array(4);
  const read = (which) => {  // only each sample's centre pixel: a full readback is ~50 MB on a phone
    mat.uniforms.uWhich.value = which;
    renderer.setRenderTarget(rt);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, new THREE.Camera());
    return samples.map((_, i) => { const [x, y] = centre(i); renderer.readRenderTargetPixels(rt, x, y, 1, 1, px); return [px[0], px[1], px[2]]; });
  };
  const e = read(0), l = read(1);
  rt.dispose(); g.dispose(); mat.uniforms.uNet.value.dispose(); mat.dispose();
  return samples.map((_, i) => [...e[i], ...l[i]]);
}
