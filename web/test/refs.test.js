import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { nearestRefView, orbitLimits, viewDir } from "../src/refs.js";

const cfg = JSON.parse(readFileSync(new URL("../data/config.json", import.meta.url)));

test("nearest ref view", () => {
  cfg.ref_poses.forEach((v, i) => {
    assert.equal(nearestRefView(viewDir(v), cfg.ref_poses), i);
  });
  // slightly off view 2 still maps to 2
  const v2 = { ...cfg.ref_poses[2], az_deg: cfg.ref_poses[2].az_deg + 10 };
  assert.equal(nearestRefView(viewDir(v2), cfg.ref_poses), 2);
});

test("viewDir az=0 el=0 points to +Z", () => {
  const d = viewDir({ az_deg: 0, el_deg: 0 });
  assert.ok(Math.abs(d[0]) < 1e-12 && Math.abs(d[1]) < 1e-12 && Math.abs(d[2] - 1) < 1e-12);
});

test("orbit limits match training domain", () => {
  const l = orbitLimits(cfg);
  assert.ok(Math.abs(l.maxPolarAngle - (Math.PI / 2 + 3 * Math.PI / 180)) < 1e-9);
  assert.ok(Math.abs(l.minPolarAngle - (Math.PI / 2 - 60 * Math.PI / 180)) < 1e-9);
  assert.equal(l.minDistance, 0.25);
  assert.equal(l.maxDistance, 0.6);
});

import { leaveRef, withPose } from "../src/refs.js";

test("leaving the reference pose turns ref sides into neural only", () => {
  assert.deepEqual(leaveRef({ left: "classic", right: "ref" }), { left: "classic", right: "neural" });
  assert.deepEqual(leaveRef({ left: "ref", right: "ref" }), { left: "neural", right: "neural" });
});

test("withPose benches at a fixed pose and restores the user's camera", async () => {
  const at = [];
  const cam = {
    position: { v: [9, 9, 9], clone() { return { v: [...this.v] }; }, copy(p) { this.v = [...(p.v ?? p)]; } },
    lookAt(t) { at.push([...this.position.v]); },
  };
  let seen;
  const r = await withPose(cam, [1, 2, 3], [0, 0, 0], async () => { seen = [...cam.position.v]; return 42; });
  assert.equal(r, 42);
  assert.deepEqual(seen, [1, 2, 3]);
  assert.deepEqual(cam.position.v, [9, 9, 9]);
});

test("withPose restores the camera even if the benchmark throws", async () => {
  const cam = { position: { v: [9, 9, 9], clone() { return { v: [...this.v] }; }, copy(p) { this.v = [...(p.v ?? p)]; } }, lookAt() {} };
  await assert.rejects(withPose(cam, [1, 2, 3], [0, 0, 0], async () => { throw new Error("x"); }));
  assert.deepEqual(cam.position.v, [9, 9, 9]);
});
