import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mlpGlsl, glslFloat } from "../src/mlp.js";

const w = JSON.parse(readFileSync(new URL("../data/weights_16.json", import.meta.url)));

test("const array sizes match width", () => {
  const src = mlpGlsl(w);
  assert.match(src, /float W0\[272\]/); // 17*16
  assert.match(src, /float W1\[256\]/); // 16*16
  assert.match(src, /float W2\[48\]/);  // 16*3
  assert.match(src, /vec3 neuralSSS\(float x\[17\]\)/);
});

test("every weight is emitted, row-major", () => {
  const src = mlpGlsl(w);
  const body = src.match(/float W0\[272\] = float\[272\]\(([^)]*)\)/)[1].split(",");
  assert.equal(body.length, 272);
  assert.equal(Number(body[1]), Number(glslFloat(w.layers[0].W[0][1])));
});

test("glslFloat always yields a GLSL float literal", () => {
  for (const v of [1, -2, 0, 1e-7, -3.5e12, 0.1]) assert.match(glslFloat(v), /^-?\d+(\.\d+)?(e[+-]?\d+)?$/);
  for (const v of [1, -2, 0]) assert.ok(glslFloat(v).includes("."));
});
