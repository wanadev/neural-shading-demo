// "Coût par image" panel: GPU milliseconds per full frame, against the 60 fps budget.
const BUDGET_MS = 1000 / 60;
const fr = (v, d) => v.toFixed(d).replace(".", ",");

export const fmtMs = (v) => (Number.isFinite(v) ? `${fr(v, v >= 10 ? 1 : 2)} ms` : "—");

// results: [{technique, frameMs, shadeMs}] or null (not measured yet: placeholders from `labels`).
export function costRows(results, labels = []) {
  if (!results) return labels.map((label) => ({ label, value: "—", detail: "", pct: 0, over: false }));
  const base = results.find((r) => r.technique === "Classique")?.frameMs;
  return results.map((r) => {
    const delta = r.technique !== "Classique" && Number.isFinite(base) ? ` · ${r.frameMs >= base ? "+" : "−"}${fmtMs(Math.abs(r.frameMs - base))} vs classique` : "";
    return {
      label: r.technique,
      value: fmtMs(r.frameMs),
      detail: `≈ ${Math.round(1000 / r.frameMs)} i/s${delta}`,
      pct: Math.min(100, (r.frameMs / BUDGET_MS) * 100),
      over: r.frameMs > BUDGET_MS,
    };
  });
}
