const ms = (v) => `${v.toFixed(2)} ms`;

// Live HUD line. Per-side GPU times when the timer extension exists, else frame time.
export function formatHud({ leftLabel, leftMs, rightLabel, rightMs, frameMs, device }) {
  const timing = leftMs != null && rightMs != null
    ? `${leftLabel} ${ms(leftMs)} · ${rightLabel} ${ms(rightMs)}`
    : `frame ${Number.isFinite(frameMs) ? frameMs.toFixed(1) : "–"} ms`;
  return device ? `${timing} · ${device}` : timing;
}
