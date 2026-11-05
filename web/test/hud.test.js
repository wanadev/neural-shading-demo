import { test } from "node:test";
import assert from "node:assert/strict";
import { formatHud } from "../src/hud.js";

test("hud without timer shows frame time", () => {
  const s = formatHud({ leftLabel: "Classique", leftMs: null, rightLabel: "Neural", rightMs: null, frameMs: 16.7, device: "Pixel 6a" });
  assert.match(s, /frame 16\.7 ms/);
  assert.match(s, /Pixel 6a/);
  assert.doesNotMatch(s, /NaN|null|undefined/);
});

test("hud with timer shows both sides", () => {
  const s = formatHud({ leftLabel: "Classique", leftMs: 0.91, rightLabel: "Neural", rightMs: 1.4, frameMs: 16.7, device: "PC" });
  assert.match(s, /Classique 0\.91 ms · Neural 1\.40 ms/);
});

test("hud with missing frame time never prints NaN", () => {
  const s = formatHud({ leftLabel: "A", leftMs: null, rightLabel: "B", rightMs: null, frameMs: NaN, device: "" });
  assert.doesNotMatch(s, /NaN|null|undefined/);
});
