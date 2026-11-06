import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mlpGlsl, packWeights } from "../src/mlp.js";

const load = (w) => JSON.parse(readFileSync(new URL(`../data/weights_${w}.json`, import.meta.url)));

// Weights live in an RGBA32F texture: baking them as GLSL const arrays made the 64-wide shader
// (~6k constants) crash Chrome's D3D11 shader compiler (GPU process exit, context lost).
test("shader reads weights from a texture, no baked const arrays", () => {
  const src = mlpGlsl(load(64));
  assert.match(src, /uniform highp sampler2D uNet;/);
  assert.doesNotMatch(src, /const float/);
  assert.match(src, /void neuralSSS\(float x\[19\], out vec3 envSSS, out vec3 lampSSS\)/);
  assert.ok(src.length < 4000, `generated GLSL should stay small, got ${src.length}`);
});

test("packWeights puts every parameter where the layout says, rows padded to vec4", () => {
  const w = load(32);
  const { data, layout, width } = packWeights(w);
  assert.equal(data.length % (4 * width), 0);
  const at = (vec4Index, comp) => data[vec4Index * 4 + comp];
  // first layer: 32 rows of 19 inputs padded to 20 (5 vec4)
  const L0 = layout.layers[0];
  assert.equal(L0.dinV, 5);
  for (const [j, i] of [[0, 0], [0, 18], [31, 7], [5, 12]]) {
    const v = L0.w + j * L0.dinV + Math.floor(i / 4);
    assert.equal(at(v, i % 4), Math.fround(w.layers[0].W[j][i]));
  }
  assert.equal(at(L0.w + 0 * L0.dinV + 4, 3), 0); // padding (input 19) is zero
  const L2 = layout.layers[2];
  assert.equal(at(L2.b + 1, 1), Math.fround(w.layers[2].b[5])); // bias index 5 -> vec4 1, comp 1
  assert.equal(at(layout.inMean + 2, 2), Math.fround(w.in_mean[10]));
  assert.equal(at(layout.outCap + 1, 1), Math.fround(w.out_log_cap[5]));
});

test("same code shape for every width", () => {
  for (const w of [16, 32, 64]) assert.match(mlpGlsl(load(w)), new RegExp(`vec4 h1\\[${w / 4}\\]`));
});
