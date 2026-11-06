import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { backlight, clampLamp, lampPosition } from "../src/lamp.js";

const cfg = JSON.parse(readFileSync(new URL("../data/config.json", import.meta.url)));

test("lampPosition matches python convention", () => {
  const p = lampPosition({ az_deg: 90, el_deg: 0 }, cfg);
  const c = cfg.lamp.center, r = cfg.lamp.radius;
  [c[0] + r, c[1], c[2]].forEach((v, i) => assert.ok(Math.abs(p[i] - v) < 1e-12));
});

test("clampLamp keeps the lamp in the training domain", () => { // Review Focus #1
  assert.equal(clampLamp({ az_deg: 370, el_deg: 95 }, cfg).el_deg, cfg.lamp.elevation_deg[1]);
  assert.equal(clampLamp({ az_deg: -30, el_deg: -40 }, cfg).el_deg, cfg.lamp.elevation_deg[0]);
  assert.equal(clampLamp({ az_deg: 370, el_deg: 0 }, cfg).az_deg, 10);
  assert.equal(clampLamp({ az_deg: -30, el_deg: 0 }, cfg).az_deg, 330);
});

test("backlight puts the lamp opposite the camera", () => {
  assert.deepEqual(backlight(30), { az_deg: 210, el_deg: 20 });
  assert.equal(backlight(300).az_deg, 120);
});
