// Small DOM helpers shared by the demo and the labo page.

// Error banner at the top of the page. textContent, not HTML: messages may echo URLs.
export function showError(e) {
  const div = document.getElementById("err") ?? document.body.insertAdjacentElement("afterbegin", document.createElement("div"));
  div.id = "err";
  div.textContent = `Erreur : ${e?.message ?? e}`;
}

// Toggle button state, visible (.on) and announced (aria-pressed).
export function markOn(button, on) {
  button.classList.toggle("on", on);
  button.setAttribute("aria-pressed", String(on));
}

// Per-frame labels: only touch the DOM when the text changes.
export function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

// Comparison curtain: drag the handle, or focus it and use the arrow keys.
// Returns place(), which moves the handle to the current split.
export function curtainHandle(handle, stage, getSplit, setSplit) {
  Object.assign(handle, { tabIndex: 0, role: "slider", ariaLabel: "Rideau de comparaison", ariaValueMin: "0", ariaValueMax: "100" });
  const place = () => {
    const pct = getSplit() * 100;
    handle.style.left = `${pct}%`;
    stage.style.setProperty("--split", `${pct}%`);
    handle.setAttribute("aria-valuenow", String(Math.round(pct)));
  };
  const move = (s) => { setSplit(Math.min(1, Math.max(0, s))); place(); };
  let dragging = false;
  handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); e.stopPropagation(); });
  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = stage.getBoundingClientRect();
    move((e.clientX - r.left) / r.width);
  });
  handle.addEventListener("pointerup", () => { dragging = false; });
  handle.addEventListener("pointercancel", () => { dragging = false; });
  const KEYS = { ArrowLeft: -0.05, ArrowDown: -0.05, ArrowRight: 0.05, ArrowUp: 0.05 };
  handle.addEventListener("keydown", (e) => {
    if (e.key in KEYS) move(getSplit() + KEYS[e.key]);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(1);
    else return;
    e.preventDefault();
  });
  return place;
}
