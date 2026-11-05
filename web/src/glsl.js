// Shared GLSL snippets. FEATURES_GLSL mirrors pipeline/features.py::features, CHORD_GLSL mirrors
// chord_to_flame, COAT_GLSL mirrors Mitsuba's GGX rough-dielectric reflection (pipeline/scene.py).

export const FEATURES_GLSL = /* glsl */ `
void features(vec3 pos, vec3 nrm, vec3 cam, vec3 flameOffset, vec3 flameBase, out float x[17]) {
  vec3 view = normalize(cam - pos);
  vec3 toL = flameBase + flameOffset - pos;
  float d = length(toL);
  vec3 L = toL / d;
  x[0] = pos.x; x[1] = pos.y; x[2] = pos.z;
  x[3] = nrm.x; x[4] = nrm.y; x[5] = nrm.z;
  x[6] = view.x; x[7] = view.y; x[8] = view.z;
  x[9] = flameOffset.x; x[10] = flameOffset.y; x[11] = flameOffset.z;
  x[12] = L.x; x[13] = L.y; x[14] = L.z;
  x[15] = d;
  x[16] = dot(nrm, L);
}
`;

export const CHORD_GLSL = /* glsl */ `
float chordToFlame(vec3 P, vec3 F, float R, float H) {
  vec3 D = F - P;
  float t0 = 0.0, t1 = 1.0;
  float a = D.x * D.x + D.z * D.z;
  float b = 2.0 * (P.x * D.x + P.z * D.z);
  float c = P.x * P.x + P.z * P.z - R * R;
  if (a < 1e-14) {
    if (c >= 0.0) return 0.0;
  } else {
    float disc = b * b - 4.0 * a * c;
    if (disc <= 0.0) return 0.0;
    float s = sqrt(disc);
    t0 = max(t0, (-b - s) / (2.0 * a));
    t1 = min(t1, (-b + s) / (2.0 * a));
  }
  if (abs(D.y) < 1e-14) {
    if (P.y <= 0.0 || P.y >= H) return 0.0;
  } else {
    float y0 = -P.y / D.y, y1 = (H - P.y) / D.y;
    t0 = max(t0, min(y0, y1));
    t1 = min(t1, max(y0, y1));
  }
  return max(t1 - t0, 0.0) * length(D);
}
`;

// GGX + Smith G + Schlick Fresnel, point light: returns BRDF * cos / d^2 (multiply by intensity).
export const COAT_GLSL = /* glsl */ `
float coatSpec(vec3 n, vec3 v, vec3 toL, float ior, float alpha) {
  float d2 = dot(toL, toL);
  vec3 l = toL * inversesqrt(d2);
  vec3 h = normalize(l + v);
  float nl = max(dot(n, l), 0.0), nv = max(dot(n, v), 1e-4);
  float nh = max(dot(n, h), 0.0), vh = max(dot(v, h), 0.0);
  float a2 = alpha * alpha;
  float dd = nh * nh * (a2 - 1.0) + 1.0;
  float D = a2 / (3.14159265 * dd * dd);
  float g1 = 2.0 * nl / (nl + sqrt(a2 + (1.0 - a2) * nl * nl));
  float g2 = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
  float f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
  float F = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
  return D * g1 * g2 * F / (4.0 * nv) / d2;
}
`;
