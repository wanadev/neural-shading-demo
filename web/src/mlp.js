// Network = weights JSON (pipeline/train.py export) packed into an RGBA32F texture + a small generic
// GLSL function `void neuralSSS(float x[19], out vec3 envSSS, out vec3 lampSSS)`, x = latent(8) | raw(11).
// Weights used to be baked as GLSL const arrays: at width 64 (~6k constants) Chrome's D3D11 shader
// compiler crashed the GPU process. A texture keeps the shader tiny and the same on every GPU.
export const TEXW = 256; // vec4 per texture row

// Layout, in vec4 units: in_mean | in_std | out cap | per layer: W rows (each padded to vec4) | b.
export function packWeights(w) {
  const vecs = [];
  const push = (arr) => {
    const start = vecs.length;
    for (let i = 0; i < arr.length; i += 4) vecs.push([arr[i] ?? 0, arr[i + 1] ?? 0, arr[i + 2] ?? 0, arr[i + 3] ?? 0]);
    return start;
  };
  const layout = {
    inMean: push(w.in_mean), inStd: push(w.in_std),
    outCap: push(w.out_log_cap ?? w.layers.at(-1).b.map(() => 80)), layers: [],
  };
  for (const l of w.layers) {
    const din = l.W[0].length, start = vecs.length;
    l.W.forEach((row) => push(row));
    layout.layers.push({ w: start, b: push(l.b), din, dinV: Math.ceil(din / 4), dout: l.b.length });
  }
  const height = Math.ceil(vecs.length / TEXW);
  const data = new Float32Array(height * TEXW * 4);
  vecs.forEach((v, i) => data.set(v, i * 4));
  return { data, width: TEXW, height, layout };
}

// GLSL helpers shared by every network: netv(i) = i-th vec4 of the weight texture, lane(j) = one-hot
// vec4 for component j, comp(v, j) = v[j] without dynamic vector indexing (slow or broken on some GPUs).
export const NET_GLSL = `
uniform highp sampler2D uNet;
vec4 netv(int i) { return texelFetch(uNet, ivec2(i % ${TEXW}, i / ${TEXW}), 0); }
vec4 lane(int j) { return vec4(equal(ivec4(0, 1, 2, 3), ivec4(j % 4))); }
float comp(vec4 v, int j) { return dot(v, lane(j)); }
`;

const ACTIVATIONS = { relu2: "max(s, 0.0) * max(s, 0.0)", relu: "max(s, 0.0)", linear: "s" };

// One dense layer: vec4 h{k} (input) -> vec4 h{k+1}, packed 4 values per vec4.
export function layerGlsl(l, k, act) {
  const nOut = Math.ceil(l.dout / 4);
  return `
  vec4 h${k + 1}[${nOut}];
  for (int v = 0; v < ${nOut}; v++) h${k + 1}[v] = vec4(0.0);
  for (int j = 0; j < ${l.dout}; j++) {
    float s = comp(netv(${l.b} + j / 4), j);
    for (int v = 0; v < ${l.dinV}; v++) s += dot(netv(${l.w} + j * ${l.dinV} + v), h${k}[v]);
    h${k + 1}[j / 4] += lane(j) * (${ACTIVATIONS[act]});
  }`;
}

export function mlpGlsl(w) {
  const { layout } = packWeights(w);
  const nRaw = w.in_mean.length, nIn = layout.layers[0].din, nLat = nIn - nRaw, inV = layout.layers[0].dinV;
  const L = layout.layers, out = L.at(-1);
  const hidden = L.slice(0, -1).map((l, k) => layerGlsl(l, k, "relu2")).join("");
  const last = L.length - 1;
  return `${NET_GLSL}
void neuralSSS(float x[${nIn}], out vec3 envSSS, out vec3 lampSSS) {
  vec4 h0[${inV}];
  for (int v = 0; v < ${inV}; v++) h0[v] = vec4(0.0);
  for (int i = 0; i < ${nLat}; i++) h0[i / 4] += lane(i) * x[i];
  for (int i = 0; i < ${nRaw}; i++) {
    int k = ${nLat} + i;
    h0[k / 4] += lane(k) * ((x[k] - comp(netv(${layout.inMean} + i / 4), i)) / comp(netv(${layout.inStd} + i / 4), i));
  }${hidden}
  float o[${out.dout}];
  for (int j = 0; j < ${out.dout}; j++) {
    float s = comp(netv(${out.b} + j / 4), j);
    for (int v = 0; v < ${out.dinV}; v++) s += dot(netv(${out.w} + j * ${out.dinV} + v), h${last}[v]);
    o[j] = exp(min(s, comp(netv(${layout.outCap} + j / 4), j)));
  }
  envSSS = vec3(o[0], o[1], o[2]);
  lampSSS = vec3(o[3], o[4], o[5]);
}
`;
}
