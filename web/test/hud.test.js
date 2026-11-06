import { test } from "node:test";
import assert from "node:assert/strict";
import { costRows, fmtMs } from "../src/hud.js";

test("French formatting of milliseconds", () => {
  assert.equal(fmtMs(3.4567), "3,46 ms");
  assert.equal(fmtMs(12.04), "12,0 ms");
  assert.equal(fmtMs(NaN), "—");
});

test("cost rows: bar against the 60 fps budget, fps, delta vs classic", () => {
  const rows = costRows([
    { technique: "Classique", frameMs: 2.0, shadeMs: 0.8 },
    { technique: "Neural 32", frameMs: 3.5, shadeMs: 2.1 },
    { technique: "Neural 64", frameMs: 20, shadeMs: 15 },
  ]);
  assert.equal(rows[0].value, "2,00 ms");
  assert.equal(rows[0].detail, "≈ 500 i/s");
  assert.ok(Math.abs(rows[0].pct - (2 / 16.67) * 100) < 0.5);
  assert.equal(rows[1].detail, "≈ 286 i/s · +1,50 ms vs classique");
  assert.equal(rows[2].pct, 100);   // bar capped
  assert.equal(rows[2].over, true); // over the 60 fps budget
  assert.equal(rows[1].over, false);
});

test("placeholder rows before any measurement", () => {
  const rows = costRows(null, ["Classique", "Neural 16"]);
  assert.deepEqual(rows.map((r) => [r.label, r.value, r.pct]), [["Classique", "—", 0], ["Neural 16", "—", 0]]);
});
