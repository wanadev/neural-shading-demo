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

import { pipelineSettings, parseDeepLink } from "../src/policy.js";

test("ldr fallback avoids every half-float render target (PMREM) and tone-maps the reference", () => {
  const ldr = pipelineSettings("ldr"), hdr = pipelineSettings("hdr");
  assert.equal(ldr.useEnvMap, false);
  assert.equal(ldr.backgroundBlurriness, 0);
  assert.equal(hdr.useEnvMap, true);
  assert.ok(hdr.backgroundBlurriness > 0);
  assert.equal(ldr.refToneMapped, true);
});

test("deep links: lamp is applied before the sides so a ref side keeps its own lamp", () => {
  const cfg = { ref_poses: [{}, {}, {}, {}] };
  const d = parseDeepLink("?left=ref&right=toString&lamp=10,20&split=0&pose=9", cfg);
  assert.deepEqual(d.order, ["pose", "lamp", "sides"]);
  assert.deepEqual(d.sides, { left: "ref" });      // prototype keys rejected
  assert.deepEqual(d.lamp, { az_deg: 10, el_deg: 20 });
  assert.equal(d.split, 0);                         // 0 is a valid split
  assert.equal(d.pose, null);                       // out of range
});
