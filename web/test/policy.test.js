import { test } from "node:test";
import assert from "node:assert/strict";
import { effectiveWidth, pickPipeline, sideMode } from "../src/policy.js";

test("ldr fallback without half-float targets", () => { // Review Focus #5
  assert.equal(pickPipeline({ halfFloatRT: false }), "ldr");
  assert.equal(pickPipeline({ halfFloatRT: true }), "hdr");
});

test("ref side stays neural until its image is loaded", () => { // Review Focus #4
  assert.equal(sideMode("ref", false), "neural");
  assert.equal(sideMode("ref", true), "ref");
  assert.equal(sideMode("classic", false), "classic");
});

test("width switch waits for its latents", () => { // Review Focus #3
  assert.equal(effectiveWidth(64, new Set([32]), 32), 32);
  assert.equal(effectiveWidth(64, new Set([32, 64]), 32), 64);
});
