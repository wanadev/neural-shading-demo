import * as THREE from "three";
import { CHORD_GLSL, COAT_GLSL, FEATURES_GLSL } from "./glsl.js";
import { mlpGlsl } from "./mlp.js";

const VERT = /* glsl */ `
varying vec3 vPos;
varying vec3 vNrm;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vPos = wp.xyz;
  vNrm = mat3(modelMatrix) * normal;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

// Uniforms shared by every material, updated once per frame by main.js.
export function flameUniforms(cfg) {
  const r = cfg.flame.sphere_radius;
  // sphere area light of radiance Le == point light of intensity Le * pi * r^2
  const I = new THREE.Vector3(...cfg.flame.radiance).multiplyScalar(Math.PI * r * r);
  return {
    uFlameBase: { value: new THREE.Vector3(...cfg.flame.base) },
    uFlameOffset: { value: new THREE.Vector3() },
    uFlameIntensity: { value: 1 },
    uFlameI: { value: I },
  };
}

// mode: "neural" (needs weights) | "classic" (needs classic params)
export function makeWaxMaterial({ mode, weights, classic, cfg, shared }) {
  const neural = mode === "neural";
  return new THREE.ShaderMaterial({
    uniforms: {
      ...shared,
      uIor: { value: cfg.wax.ior },
      uAlpha: { value: cfg.wax.roughness },
      uR: { value: cfg.candle.radius },
      uH: { value: cfg.candle.height },
      cW: { value: classic?.w ?? 0 }, cK: { value: classic?.k ?? 0 },
      cA: { value: classic?.a ?? 0 }, cB: { value: classic?.b ?? 0 },
      cTint: { value: new THREE.Vector3(...(classic?.tint ?? [0, 0, 0])) },
    },
    vertexShader: VERT,
    fragmentShader: /* glsl */ `
precision highp float;
uniform vec3 uFlameBase, uFlameOffset, uFlameI;
uniform float uFlameIntensity, uIor, uAlpha, uR, uH;
uniform float cW, cK, cA, cB;
uniform vec3 cTint;
varying vec3 vPos;
varying vec3 vNrm;
${FEATURES_GLSL}
${CHORD_GLSL}
${COAT_GLSL}
${neural ? mlpGlsl(weights) : ""}
void main() {
  vec3 n = normalize(vNrm);
  vec3 flame = uFlameBase + uFlameOffset;
  vec3 toL = flame - vPos;
#ifdef NEURAL
  float x[17];
  features(vPos, n, cameraPosition, uFlameOffset, uFlameBase, x);
  vec3 sss = neuralSSS(x);
#else
  float d = length(toL);
  float wrap = max((dot(n, toL / d) + cW) / (1.0 + cW), 0.0);
  vec3 sss = cTint / (d * d) * (cA * wrap + cB * exp(-cK * chordToFlame(vPos, flame, uR, uH)));
#endif
  vec3 spec = coatSpec(n, normalize(cameraPosition - vPos), toL, uIor, uAlpha) * uFlameI;
  gl_FragColor = vec4((sss + spec) * uFlameIntensity, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
    defines: neural ? { NEURAL: "" } : {},
  });
}

// Dark table: Lambert from the flame point light, shadowed by the candle (chord > 0).
// Known gap vs Mitsuba: light transmitted through the wax onto the table is not modelled.
export function makeTableMaterial(cfg, shared) {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...shared,
      uAlbedo: { value: new THREE.Vector3(...cfg.table.albedo) },
      uR: { value: cfg.candle.radius },
      uH: { value: cfg.candle.height },
    },
    vertexShader: VERT,
    fragmentShader: /* glsl */ `
precision highp float;
uniform vec3 uFlameBase, uFlameOffset, uFlameI, uAlbedo;
uniform float uFlameIntensity, uR, uH;
varying vec3 vPos;
varying vec3 vNrm;
${CHORD_GLSL}
void main() {
  vec3 flame = uFlameBase + uFlameOffset;
  vec3 toL = flame - vPos;
  float d2 = dot(toL, toL);
  float vis = chordToFlame(vPos, flame, uR, uH) > 0.0 ? 0.0 : 1.0;
  vec3 c = uAlbedo / 3.14159265 * uFlameI * uFlameIntensity * max(toL.y, 0.0) * inversesqrt(d2) / d2 * vis;
  gl_FragColor = vec4(c, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
  });
}
