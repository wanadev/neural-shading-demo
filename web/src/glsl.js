// Shared GLSL snippets. FEATURES_GLSL mirrors pipeline/features.py::raw_features (after the 8 latents),
// SH_GLSL mirrors pipeline/env.py::eval_sh_irradiance, CLASSIC_GLSL mirrors pipeline/classic.py::classic_sss.

export const FEATURES_GLSL = /* glsl */ `
uniform float uLatLo[8];
uniform float uLatHi[8];
void features(vec4 latA, vec4 latB, vec3 nrm, vec3 pos, vec3 cam, vec3 lamp, out float x[19]) {
  float l[8] = float[8](latA.x, latA.y, latA.z, latA.w, latB.x, latB.y, latB.z, latB.w);
  for (int i = 0; i < 8; i++) x[i] = uLatLo[i] + (uLatHi[i] - uLatLo[i]) * l[i];
  vec3 view = normalize(cam - pos);
  vec3 toL = lamp - pos;
  float d = length(toL);
  vec3 L = toL / d;
  x[8] = nrm.x; x[9] = nrm.y; x[10] = nrm.z;
  x[11] = view.x; x[12] = view.y; x[13] = view.z;
  x[14] = L.x; x[15] = L.y; x[16] = L.z;
  x[17] = d;
  x[18] = dot(nrm, L);
}
`;

export const SH_GLSL = /* glsl */ `
uniform vec3 uSH[9];
vec3 shIrradiance(vec3 n) {
  return uSH[0] * 0.282095
       + uSH[1] * 0.488603 * n.y + uSH[2] * 0.488603 * n.z + uSH[3] * 0.488603 * n.x
       + uSH[4] * 1.092548 * n.x * n.y + uSH[5] * 1.092548 * n.y * n.z
       + uSH[6] * 0.315392 * (3.0 * n.z * n.z - 1.0)
       + uSH[7] * 1.092548 * n.x * n.z + uSH[8] * 0.546274 * (n.x * n.x - n.y * n.y);
}
`;

// "Local thickness" translucency as used in games (DICE/Frostbite style), fitted on the same data.
export const CLASSIC_GLSL = /* glsl */ `
uniform float cW, cK, cP, cDist;
uniform vec3 cTintD, cTintT, cTintE, cTintEB;
void classicSSS(vec3 n, vec3 pos, vec3 cam, vec3 lamp, float th, out vec3 envSSS, out vec3 lampSSS) {
  vec3 toL = lamp - pos;
  float d = length(toL);
  vec3 L = toL / d;
  vec3 V = normalize(cam - pos);
  float wrap = max((dot(n, L) + cW) / (1.0 + cW), 0.0);
  float tr = exp(-cK * th) * pow(max(dot(V, -normalize(L + n * cDist)), 0.0), cP);
  lampSSS = (cTintD * wrap + cTintT * tr) / (d * d);
  envSSS = cTintE * shIrradiance(n) + cTintEB * shIrradiance(-n) * exp(-cK * th);
}
`;
