// Real-time glass for the labo page. Two live methods, same Fresnel, same environment:
//   classic : one refraction at the entry surface, then an environment lookup
//   traced  : the exact dominant path through the mesh (refraction in, up to 8 total internal
//             reflections, refraction out), ray-cast in the fragment shader against a BVH
//             (three-mesh-bvh). GLSL port of trace_exit() in spikes/glass/glass_spike.py.
// The Mitsuba reference and the (failed) network are precomputed images at the reference poses.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";
import Stats from "three/addons/libs/stats.module.js";
import { BVHShaderGLSL, FloatVertexAttributeTexture, MeshBVH, MeshBVHUniformStruct } from "three-mesh-bvh";
import { orbitLimits, viewDir } from "./refs.js";

const IOR = 1.5, ENV_BOOST = 4;  // same as the spike

function glassMaterial(mode, envMap, envScale, bvhUniform, normalTex) {
  return new THREE.ShaderMaterial({
    uniforms: { envMap: { value: envMap }, envScale: { value: envScale }, bvh: { value: bvhUniform }, normalAttr: { value: normalTex } },
    defines: mode === "traced" ? { TRACED: "" } : {},
    vertexShader: "varying vec3 vP; varying vec3 vN; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w; }",
    fragmentShader: /* glsl */ `
precision highp float;
precision highp isampler2D;
precision highp usampler2D;
${BVHShaderGLSL.common_functions}
${BVHShaderGLSL.bvh_struct_definitions}
${BVHShaderGLSL.bvh_ray_functions}
uniform BVH bvh;
uniform sampler2D normalAttr;
uniform sampler2D envMap;
uniform float envScale;
varying vec3 vP;
varying vec3 vN;
vec3 env(vec3 d) {  // three.js equirectUv
  vec2 uv = vec2(atan(d.z, d.x) * 0.15915494 + 0.5, asin(clamp(d.y, -1.0, 1.0)) * 0.31830989 + 0.5);
  return texture2D(envMap, uv).rgb * envScale;
}
// exact dielectric Fresnel, eta = n_incident / n_transmitted
float fresnel(float cosi, float eta, out float cost, out bool tir) {
  float sin2t = eta * eta * max(0.0, 1.0 - cosi * cosi);
  tir = sin2t >= 1.0;
  cost = sqrt(max(0.0, 1.0 - sin2t));
  float rs = (eta * cosi - cost) / (eta * cosi + cost + 1e-12);
  float rp = (eta * cost - cosi) / (eta * cost + cosi + 1e-12);
  return tir ? 1.0 : 0.5 * (rs * rs + rp * rp);
}
vec3 refr(vec3 d, vec3 n, float eta, float cosi, float cost) { return normalize(eta * d + (eta * cosi - cost) * n); }
void main() {
  vec3 d = normalize(vP - cameraPosition), n = normalize(vN);
  if (dot(d, n) > 0.0) n = -n;
  float cosi = -dot(d, n), cost; bool tir;
  float F0 = fresnel(cosi, ${(1 / IOR).toFixed(8)}, cost, tir);
  vec3 col = F0 * env(reflect(d, n));
  vec3 t = refr(d, n, ${(1 / IOR).toFixed(8)}, cosi, cost);
#ifndef TRACED
  col += (1.0 - F0) * env(t);
#else
  float T = 1.0 - F0;
  vec3 p = vP - 1e-4 * n;
  for (int i = 0; i < 8; i++) {
    uvec4 faceIndices = uvec4(0u); vec3 faceNormal = vec3(0.0); vec3 bary = vec3(0.0); float side = 1.0; float dist = 0.0;
    if (!bvhIntersectFirstHit(bvh, p, t, faceIndices, faceNormal, bary, side, dist)) break;
    vec3 q = p + t * dist;
    vec3 m = normalize(textureSampleBarycoord(normalAttr, bary, faceIndices.xyz).xyz);
    if (dot(t, m) < 0.0) m = -m;  // outward, along the inside ray
    float c = dot(t, m), cost2; bool tir2;
    float F = fresnel(c, ${IOR.toFixed(8)}, cost2, tir2);
    if (!tir2) { col += T * (1.0 - F) * env(refr(t, -m, ${IOR.toFixed(8)}, c, cost2)); break; }
    t = reflect(t, m);  // total internal reflection: lossless
    p = q - 1e-4 * m;
  }
#endif
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
  });
}

export async function createGlass({ canvas, views, cfg }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.autoClear = false;
  const [geometry, envSH] = await Promise.all([new PLYLoader().loadAsync("data/dragon.ply"), fetch("data/env_sh.json").then((r) => r.json())]);
  const hdr = await new HDRLoader().loadAsync(`data/${envSH.file}`);
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  const envScale = cfg.env.scale * ENV_BOOST;
  const bvhUniform = new MeshBVHUniformStruct();
  bvhUniform.updateFrom(new MeshBVH(geometry));
  const normalTex = new FloatVertexAttributeTexture();
  normalTex.updateFrom(geometry.attributes.normal);

  const camera = new THREE.PerspectiveCamera(cfg.camera.fov_y, 1, 0.01, 20);
  const target = new THREE.Vector3(...cfg.camera.target);
  const posOf = (v) => target.clone().addScaledVector(new THREE.Vector3(...viewDir(v)), v.radius);
  camera.position.copy(posOf(views[0]));
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(target);
  Object.assign(controls, orbitLimits(cfg), { enablePan: false, enableDamping: false });
  controls.update();

  const scene = new THREE.Scene();
  scene.background = hdr;
  scene.backgroundIntensity = envScale;
  const dragon = new THREE.Mesh(geometry);
  scene.add(dragon);
  const mats = { classic: glassMaterial("classic", hdr, envScale, bvhUniform, normalTex),
                 traced: glassMaterial("traced", hdr, envScale, bvhUniform, normalTex) };
  // precomputed sides: the image, already tone-mapped, drawn as-is
  const loader = new THREE.TextureLoader();
  const imgs = {};
  const imgTex = (key, i) => (imgs[`${key}${i}`] ??= Object.assign(loader.load(`data/labo/glass/view${i}-${key}.jpg`), { colorSpace: THREE.SRGBColorSpace }));
  const imgMat = new THREE.MeshBasicMaterial({ toneMapped: false, depthTest: false, depthWrite: false });
  const imgScene = new THREE.Scene(); imgScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), imgMat));
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  let size = 0;
  function resize(cssSize, dpr = Math.min(devicePixelRatio, 2)) { size = cssSize; renderer.setPixelRatio(dpr); renderer.setSize(size, size); }

  const state = { left: "classic", right: "traced", split: 0.5, view: 0 };
  function render() {
    renderer.setScissorTest(true);
    const cut = Math.round(state.split * size);
    for (const [side, x0, w] of [["left", 0, cut], ["right", cut, size - cut]]) {
      if (w <= 0) continue;
      renderer.setScissor(x0, 0, w, size);
      renderer.clear();
      const m = state[side];
      if (mats[m]) { dragon.material = mats[m]; renderer.render(scene, camera); }
      else { imgMat.map = imgTex(m, state.view); imgMat.needsUpdate = true; renderer.render(imgScene, ortho); }
    }
    renderer.setScissorTest(false);
  }

  let tween = null, running = true, onMove = null;
  function setPose(i, animate = true) {
    const to = posOf(views[i]).sub(target), from = camera.position.clone().sub(target);
    if (!animate) { camera.position.copy(target).add(to); controls.update(); return; }
    tween = { from, to, t0: performance.now() };
  }
  controls.addEventListener("start", () => { tween = null; });
  controls.addEventListener("change", () => { if (!tween) onMove?.(); });
  const stats = new Stats();
  stats.dom.classList.add("stats");
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

  // parity: live traced vs the spike's traced image (same path, tone-mapped JPEG), at each pose
  async function check() {
    running = false;
    const saved = { ...state };
    resize(512, 1);
    const out = [];
    for (let i = 0; i < views.length; i++) {
      setPose(i, false);
      Object.assign(state, { left: "traced", right: "traced", view: i });
      render();
      const gl = renderer.getContext(), px = new Uint8Array(512 * 512 * 4);
      gl.readPixels(0, 0, 512, 512, gl.RGBA, gl.UNSIGNED_BYTE, px);
      const bmp = await createImageBitmap(await (await fetch(`data/labo/glass/view${i}-traced.jpg`)).blob());
      const c = new OffscreenCanvas(512, 512).getContext("2d"); c.drawImage(bmp, 0, 0);
      const ref = c.getImageData(0, 0, 512, 512).data;
      let sum = 0, n = 0;
      for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) for (let k = 0; k < 3; k++) {
        sum += Math.abs(px[((511 - y) * 512 + x) * 4 + k] - ref[(y * 512 + x) * 4 + k]) / 255; n++;
      }
      out.push({ view: i + 1, mae: sum / n });
    }
    Object.assign(state, saved); running = true;
    return out;
  }

  return {
    stats, state, resize, setPose, check,
    pause() { running = false; }, resume() { running = true; },
    onMove(fn) { onMove = fn; },
    nearestPose() {
      const d = camera.position.clone().sub(target).normalize();
      let best = 0, bd = -2;
      views.forEach((v, i) => { const dot = d.dot(new THREE.Vector3(...viewDir(v))); if (dot > bd) { bd = dot; best = i; } });
      return best;
    },
  };
}
