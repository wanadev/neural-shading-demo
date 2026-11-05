import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { flameOffset } from "../src/flame.js";

const cfg = JSON.parse(readFileSync(new URL("../data/config.json", import.meta.url))).flame;

test("flicker stays in training bounds", () => {
  for (let t = 0; t < 1000; t += 0.013) {
    const f = flameOffset(t, cfg);
    assert.ok(Math.abs(f.x) <= cfg.flicker_xz && Math.abs(f.z) <= cfg.flicker_xz);
    assert.ok(Math.abs(f.y) <= cfg.flicker_y);
    assert.ok(f.intensity >= cfg.intensity_range[0] && f.intensity <= cfg.intensity_range[1]);
  }
});

test("flicker actually moves", () => {
  const a = flameOffset(0, cfg), b = flameOffset(0.37, cfg);
  assert.notEqual(a.x, b.x);
  assert.notEqual(a.intensity, b.intensity);
});
