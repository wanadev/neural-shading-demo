// Weights JSON (pipeline/train.py export) -> GLSL ES 3.0 function `vec3 neuralSSS(float x[17])`.
// Weights are baked as const arrays: no uniform-count limits on mobile, one program per width.

export const glslFloat = (v) => {
  const s = Number(v).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

const arr = (name, vals) => `const float ${name}[${vals.length}] = float[${vals.length}](${vals.map(glslFloat).join(",")});`;

export function mlpGlsl(w) {
  const n = w.in_mean.length, L = w.layers;
  const decl = [arr("IN_MEAN", w.in_mean), arr("IN_STD", w.in_std)];
  L.forEach((l, k) => decl.push(arr(`W${k}`, l.W.flat()), arr(`B${k}`, l.b)));
  const dims = [n, ...L.map((l) => l.b.length)];
  const body = [`float h0[${n}];`, `for (int i = 0; i < ${n}; i++) h0[i] = (x[i] - IN_MEAN[i]) / IN_STD[i];`];
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
  const k = L.length - 1, din = dims[k];
  body.push(
    `vec3 o = vec3(B${k}[0], B${k}[1], B${k}[2]);`,
    `for (int i = 0; i < ${din}; i++) o += vec3(W${k}[i], W${k}[${din} + i], W${k}[${2 * din} + i]) * h${k}[i];`,
    `return exp(o);`,
  );
  return `${decl.join("\n")}\nvec3 neuralSSS(float x[${n}]) {\n  ${body.join("\n  ")}\n}\n`;
}
