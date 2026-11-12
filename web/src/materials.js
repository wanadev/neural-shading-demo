import * as THREE from "three";
import { CLASSIC_GLSL, FEATURES_GLSL, SH_GLSL } from "./glsl.js";
import { mlpGlsl, packWeights } from "./mlp.js";
import { shadowCameraRange } from "./lamp.js";

export function makeNetTexture(weights) {
  const { data, width, height } = packWeights(weights);
  const t = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

// Jade = three.js MeshPhysicalMaterial with a black diffuse (so three computes the GGX coat:
// prefiltered room reflections + lamp highlight, Fresnel from the jade IOR) plus our SSS term
// injected as emissive radiance. Mitsuba alpha = three roughness^2.
export function makeJadeMaterial({ mode, weights, classic, cfg, shared, envMap }) {
  const neural = mode === "neural";
  const m = new THREE.MeshPhysicalMaterial({
    color: 0x000000, metalness: 0, roughness: Math.sqrt(cfg.jade.roughness), ior: cfg.jade.ior,
    specularIntensity: 1, envMap, envMapIntensity: cfg.env.scale,
  });
  const own = neural
    ? { uLatLo: { value: weights.latent_lo }, uLatHi: { value: weights.latent_hi }, uNet: { value: makeNetTexture(weights) } }
    : {
        cW: { value: classic.w }, cK: { value: classic.k }, cP: { value: classic.p }, cDist: { value: classic.dist },
        cTintD: { value: new THREE.Vector3(...classic.tint_d) }, cTintT: { value: new THREE.Vector3(...classic.tint_t) },
        cTintE: { value: new THREE.Vector3(...classic.tint_e) }, cTintEB: { value: new THREE.Vector3(...classic.tint_eb) },
        uThickMax: { value: cfg.thickness_max },
      };
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, shared, own);
    s.vertexShader = s.vertexShader
      .replace("#include <common>", `#include <common>
attribute vec4 latA;
attribute vec4 latB;
attribute float thick;
attribute vec3 nNet;
varying vec4 vLatA;
varying vec4 vLatB;
varying float vThick;
varying vec3 vWPos;
varying vec3 vWNrm;
varying vec3 vNNet;`)
      .replace("#include <begin_vertex>", `#include <begin_vertex>
vLatA = latA; vLatB = latB; vThick = thick;
vWPos = (modelMatrix * vec4(position, 1.0)).xyz;
vWNrm = normalize(mat3(modelMatrix) * normal);
vNNet = mat3(modelMatrix) * nNet;`);
    s.fragmentShader = s.fragmentShader
      .replace("#include <common>", `#include <common>
uniform vec3 uLamp;
uniform float uThickMax;
varying vec4 vLatA;
varying vec4 vLatB;
varying float vThick;
varying vec3 vWPos;
varying vec3 vWNrm;
varying vec3 vNNet;
${SH_GLSL}
${neural ? FEATURES_GLSL + mlpGlsl(weights) : CLASSIC_GLSL}`)
      .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
{
  vec3 n = normalize(vWNrm);
  vec3 envSSS, lampSSS;
${neural
    ? `  float x[19];
  features(vLatA, vLatB, normalize(vNNet), vWPos, cameraPosition, uLamp, x);  // smoothed normal (normals_net.bin)
  neuralSSS(x, envSSS, lampSSS);`
    : `  classicSSS(n, vWPos, cameraPosition, uLamp, vThick * uThickMax, envSSS, lampSSS);`}
  totalEmissiveRadiance += envSSS + lampSSS;
}`);
  };
  m.customProgramCacheKey = () => `jade-${mode}-${weights?.width ?? "classic"}`;
  return m;
}

// Per-vertex latents for the jade: 8 uint8 values as two normalized vec4 attributes. Switching
// width rewrites the same buffers in place instead of allocating new GL buffers each time.
export function setJadeAttributes(geometry, latents) {
  const n = geometry.attributes.position.count;
  for (const [name, offset] of [["latA", 0], ["latB", 4]]) {
    let attr = geometry.getAttribute(name);
    if (!attr) geometry.setAttribute(name, (attr = new THREE.BufferAttribute(new Uint8Array(n * 4), 4, true)));
    for (let i = 0; i < n; i++) attr.array.set(latents.subarray(i * 8 + offset, i * 8 + offset + 4), i * 4);
    attr.needsUpdate = true;
  }
}

export function makeGround(cfg, envMap) {
  const s = cfg.ground.size;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(s, s).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(...cfg.ground.albedo, THREE.LinearSRGBColorSpace),
      roughness: cfg.ground.roughness, metalness: 0, envMap, envMapIntensity: cfg.env.scale,
    }),
  );
  mesh.receiveShadow = true;
  return mesh;
}

// Lamp = sphere area light of radiance Le, radius r  ==  point intensity Le * pi * r^2 (per channel).
// A SpotLight aimed at the dragon carries it (one shadow-map pass, for the ground only).
export function makeLamp(cfg) {
  const r = cfg.lamp.sphere_radius;
  const I = cfg.lamp.radiance.map((c) => c * Math.PI * r * r);
  const peak = Math.max(...I);
  const light = new THREE.SpotLight(new THREE.Color().setRGB(...I.map((c) => c / peak), THREE.LinearSRGBColorSpace),
    peak, 0, Math.PI / 2.4, 0.4, 2);
  light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  light.shadow.bias = -0.0005;
  Object.assign(light.shadow.camera, shadowCameraRange(cfg)); // default near = 0.5 m clipped the whole dragon
  light.shadow.camera.updateProjectionMatrix();
  light.target.position.set(...cfg.lamp.center);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16),
    new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(...cfg.lamp.radiance, THREE.LinearSRGBColorSpace) }));
  return { light, bulb };
}
