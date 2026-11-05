import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { nearestRefView, orbitLimits, viewDir } from "../src/refs.js";

const cfg = JSON.parse(readFileSync(new URL("../data/config.json", import.meta.url)));

test("nearest ref view", () => {
  cfg.ref_views.forEach((v, i) => {
    assert.equal(nearestRefView(viewDir(v), cfg.ref_views), i);
  });
  // slightly off view 2 still maps to 2
  const v2 = { ...cfg.ref_views[2], az_deg: cfg.ref_views[2].az_deg + 10 };
  assert.equal(nearestRefView(viewDir(v2), cfg.ref_views), 2);
});

test("viewDir az=0 el=0 points to +Z", () => {
  const d = viewDir({ az_deg: 0, el_deg: 0 });
  assert.ok(Math.abs(d[0]) < 1e-12 && Math.abs(d[1]) < 1e-12 && Math.abs(d[2] - 1) < 1e-12);
});

test("orbit limits match training domain", () => {
  const l = orbitLimits(cfg);
  assert.ok(Math.abs(l.maxPolarAngle - (Math.PI / 2 + 5 * Math.PI / 180)) < 1e-9);
  assert.ok(Math.abs(l.minPolarAngle - (Math.PI / 2 - 60 * Math.PI / 180)) < 1e-9);
  assert.equal(l.minDistance, 0.2);
  assert.equal(l.maxDistance, 0.5);
});
