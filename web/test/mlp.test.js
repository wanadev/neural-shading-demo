import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mlpGlsl, glslFloat } from "../src/mlp.js";

const w = JSON.parse(readFileSync(new URL("../data/weights_32.json", import.meta.url)));

test("const array sizes match width, 19 inputs, 6 outputs", () => {
  const src = mlpGlsl(w);
  assert.match(src, /float W0\[608\]/); // 19*32
  assert.match(src, /float W1\[1024\]/); // 32*32
  assert.match(src, /float W2\[192\]/); // 32*6
  assert.match(src, /float IN_MEAN\[11\]/);
  assert.match(src, /void neuralSSS\(float x\[19\], out vec3 envSSS, out vec3 lampSSS\)/);
});

test("standardization only touches the 11 raw inputs, not the latents", () => {
  const src = mlpGlsl(w);
  assert.match(src, /for \(int i = 0; i < 8; i\+\+\) h0\[i\] = x\[i\];/);
  assert.match(src, /for \(int i = 0; i < 11; i\+\+\) h0\[8 \+ i\] = \(x\[8 \+ i\] - IN_MEAN\[i\]\) \/ IN_STD\[i\];/);
});

test("every weight is emitted, row-major", () => {
  const body = mlpGlsl(w).match(/float W0\[608\] = float\[608\]\(([^)]*)\)/)[1].split(",");
  assert.equal(body.length, 608);
  assert.equal(Number(body[1]), Number(glslFloat(w.layers[0].W[0][1])));
});

test("glslFloat always yields a GLSL float literal", () => {
  for (const v of [1, -2, 0, 1e-7, -3.5e12, 0.1]) assert.match(glslFloat(v), /^-?\d+(\.\d+)?(e[+-]?\d+)?$/);
  for (const v of [1, -2, 0]) assert.ok(glslFloat(v).includes("."));
});
