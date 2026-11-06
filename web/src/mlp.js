// Weights JSON (pipeline/train.py export) -> GLSL ES 3.0
// `void neuralSSS(float x[19], out vec3 envSSS, out vec3 lampSSS)`, x = latent(8) | raw(11).
// Weights are baked as const arrays: no uniform-count limits on mobile, one program per width.

export const glslFloat = (v) => {
  const s = Number(v).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

const arr = (name, vals) => `const float ${name}[${vals.length}] = float[${vals.length}](${vals.map(glslFloat).join(",")});`;

export function mlpGlsl(w) {
  const L = w.layers, nRaw = w.in_mean.length, nIn = L[0].W[0].length, nLat = nIn - nRaw;
  const decl = [arr("IN_MEAN", w.in_mean), arr("IN_STD", w.in_std)];
  L.forEach((l, k) => decl.push(arr(`W${k}`, l.W.flat()), arr(`B${k}`, l.b)));
  const dims = [nIn, ...L.map((l) => l.b.length)];
  const body = [
    `float h0[${nIn}];`,
    `for (int i = 0; i < ${nLat}; i++) h0[i] = x[i];`,
    `for (int i = 0; i < ${nRaw}; i++) h0[${nLat} + i] = (x[${nLat} + i] - IN_MEAN[i]) / IN_STD[i];`,
  ];
  for (let k = 0; k < L.length - 1; k++) {
    const [din, dout] = [dims[k], dims[k + 1]];
    body.push(
      `float h${k + 1}[${dout}];`,
      `for (int j = 0; j < ${dout}; j++) {`,
      `  float s = B${k}[j];`,
      `  for (int i = 0; i < ${din}; i++) s += W${k}[j * ${din} + i] * h${k}[i];`,
      `  float r = max(s, 0.0); h${k + 1}[j] = r * r;`,
      `}`,
    );
  }
  const k = L.length - 1, din = dims[k], nOut = dims[k + 1];
  body.push(
    `float o[${nOut}];`,
    `for (int j = 0; j < ${nOut}; j++) {`,
    `  float s = B${k}[j];`,
    `  for (int i = 0; i < ${din}; i++) s += W${k}[j * ${din} + i] * h${k}[i];`,
    `  o[j] = exp(s);`,
    `}`,
    `envSSS = vec3(o[0], o[1], o[2]);`,
    `lampSSS = vec3(o[3], o[4], o[5]);`,
  );
  return `${decl.join("\n")}\nvoid neuralSSS(float x[${nIn}], out vec3 envSSS, out vec3 lampSSS) {\n  ${body.join("\n  ")}\n}\n`;
}
