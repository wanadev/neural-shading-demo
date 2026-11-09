// Labo page. AO runs live (labo-ao.js); the ray-traced AO and everything about glass are
// precomputed images from spikes/ao and spikes/glass.
const $ = (id) => document.getElementById(id);

const EFFECTS = {
  ao: {
    label: "Ambient occlusion", live: true,
    methods: { gt: "Ray tracing", classic: "SSAO classique", neural: "Réseau" },
    reference: "gt", left: "classic", right: "neural", view: 1,
    errNote: "Erreur moyenne dans les zones d'ombre, par rapport au ray tracing, après le flou (comme à l'écran), à cette vue de référence. Plus bas, c'est mieux.",
    note: "SSAO classique et réseau tournent en temps réel, avec le même flou bilatéral. Le ray tracing est précalculé aux 4 vues de référence : le choisir cale la caméra dessus.",
    story: "Le réseau voit exactement ce que voit un SSAO classique, le depth buffer et les normales autour du pixel, et apprend une AO calculée en ray tracing. Il fait presque deux fois moins d'erreur dans les zones d'ombre et a moins de halos. Mais à l'œil le gain reste modeste, et les deux ratent l'ombre de contact sous le dragon : en screen space, on ne voit pas ce qui est caché.",
  },
  glass: {
    label: "Verre", live: false,
    methods: { ref: "Mitsuba", traced: "Trajet exact", classic: "Réfraction simple", neural: "Réseau" },
    reference: "ref", left: "neural", right: "ref", view: 3,
    errNote: "Erreur moyenne sur le verre, par rapport au rendu Mitsuba. Plus bas, c'est mieux.",
    note: "Images précalculées en 512 px, pas du temps réel.",
    story: "Le réseau devait apprendre où la lumière ressort du dragon en verre, rebonds internes compris. Ça ne marche pas : le rendu est granuleux et pas meilleur que la réfraction simple utilisée en temps réel. Dans le verre, deux pixels voisins peuvent ressortir dans des directions opposées, et des latents interpolés en douceur ne savent pas représenter ça. Le trajet exact, lui, se rapproche nettement de Mitsuba.",
  },
};

const state = { effect: "ao", view: EFFECTS.ao.view, left: EFFECTS.ao.left, right: EFFECTS.ao.right, split: 0.5, applied: true, aoScale: 0.5 };
const data = {};
let ao = null;

const seg = (el, items, current, onPick) => {
  el.innerHTML = items.map(([k, v]) => `<button data-k="${k}" class="${k === current ? "on" : ""}">${v}</button>`).join("");
  el.onclick = (e) => { if (e.target.dataset.k !== undefined) onPick(e.target.dataset.k); };
};
const fmt = (v) => v.toFixed(3).replace(".", ",");

function pickSide(side, k) {
  state[side] = k;
  if (EFFECTS[state.effect].live && k === "gt" && ao) {  // the ray-traced AO only exists at the reference poses
    state.view = ao.nearestPose();
    ao.setPose(state.view);
  }
  render();
}

function render() {
  const fx = EFFECTS[state.effect], d = data[state.effect];
  seg($("effects"), Object.entries(EFFECTS).map(([k, v]) => [k, v.label]), state.effect, (k) => {
    Object.assign(state, { effect: k, view: EFFECTS[k].view, left: EFFECTS[k].left, right: EFFECTS[k].right });
    if (EFFECTS[k].live) ao?.setPose(state.view, false);
    render();
  });
  seg($("views"), d.views.map((_, i) => [String(i), String(i + 1)]), String(state.view), (k) => {
    state.view = +k;
    if (fx.live) ao?.setPose(state.view);
    render();
  });
  const methods = Object.entries(fx.methods);
  seg($("left"), methods, state.left, (k) => pickSide("left", k));
  seg($("right"), methods, state.right, (k) => pickSide("right", k));
  const src = (k) => `data/labo/${state.effect}/view${state.view}-${k}.jpg`;
  for (const [side, img] of [["left", $("imgL")], ["right", $("imgR")]]) {
    img.hidden = fx.live;  // live AO draws everything, ray-traced reference included, in the canvas
    if (!fx.live) img.src = src(state[side]);
  }
  $("aoCanvas").hidden = !fx.live;
  $("aoOptions").hidden = !fx.live;
  seg($("display"), [["1", "AO appliquée"], ["0", "AO seule"]], state.applied ? "1" : "0", (k) => { state.applied = k === "1"; render(); });
  seg($("aores"), [["0.5", "demi"], ["1", "pleine"]], String(state.aoScale), (k) => { state.aoScale = +k; ao?.setAoScale(state.aoScale); render(); });
  if (ao) ao.stats.dom.hidden = !fx.live;
  if (ao) Object.assign(ao.state, { left: state.left, right: state.right, view: state.view, applied: state.applied });
  $("tagL").textContent = fx.methods[state.left];
  $("tagR").textContent = fx.methods[state.right];
  const errs = d.views[state.view].errors;
  const rows = Object.entries(errs).filter(([k]) => k !== fx.reference);
  const best = Math.min(...rows.map(([, v]) => v));
  $("errs").innerHTML = rows.map(([k, v]) =>
    `<span>${fx.methods[k]}</span><span class="val ${v === best ? "best" : ""}">${fmt(v)}</span>`).join("");
  $("errNote").textContent = fx.errNote;
  $("liveNote").textContent = fx.note;
  $("story").textContent = fx.story;
}

function placeSplit() {
  $("handle").style.left = `${state.split * 100}%`;
  $("stage").style.setProperty("--split", `${state.split * 100}%`);
  if (ao) ao.state.split = state.split;
}

const handle = $("handle");
let dragging = false;
handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); e.stopPropagation(); });
handle.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const r = $("stage").getBoundingClientRect();
  state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  placeSplit();
});
handle.addEventListener("pointerup", () => { dragging = false; });
handle.addEventListener("pointercancel", () => { dragging = false; });

try {
  const json = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p} : HTTP ${r.status}`); return r.json(); });
  const [aoData, glass, net, cfg] = await Promise.all(
    ["data/labo/ao.json", "data/labo/glass.json", "data/labo/ao_net.json", "data/config.json"].map(json));
  Object.assign(data, { ao: aoData, glass });
  const { createAo } = await import("./labo-ao.js");
  ao = await createAo({ canvas: $("aoCanvas"), net, cfg });
  $("stage").append(ao.stats.dom);
  const fit = () => ao.resize($("stage").clientWidth);
  addEventListener("resize", fit);
  fit();
  ao.setPose(state.view, false);
  ao.onMove(() => {  // a real camera move leaves the reference pose: ray-traced sides fall back to the network
    if (!EFFECTS[state.effect].live) return;
    let changed = false;
    for (const side of ["left", "right"]) if (state[side] === "gt") { state[side] = "neural"; changed = true; }
    if (changed) render();
  });
  const q = new URLSearchParams(location.search);  // deep links: ?effect=ao&left=gt&right=neural&display=0
  if (Object.hasOwn(EFFECTS, q.get("effect") ?? "")) Object.assign(state, { effect: q.get("effect"), view: EFFECTS[q.get("effect")].view,
    left: EFFECTS[q.get("effect")].left, right: EFFECTS[q.get("effect")].right });
  for (const side of ["left", "right"]) if (Object.hasOwn(EFFECTS[state.effect].methods, q.get(side) ?? "")) state[side] = q.get(side);
  if (q.get("display") === "0") state.applied = false;
  if (EFFECTS[state.effect].live) ao.setPose(state.view, false);
  placeSplit();
  render();
  if (q.has("aocheck")) {
    $("story").textContent = "AO CHECK en cours";
    const res = await ao.check().catch((e) => { $("story").textContent = "AO CHECK ERREUR " + e.message; throw e; });
    $("story").textContent = "AO CHECK " + res.map((r) => `vue ${r.view}: écart moyen ${(r.mae * 100).toFixed(2)} %, pixels > 10 % : ${(r.bad * 100).toFixed(2)} %`).join(" | ");
  }
} catch (e) {
  document.body.insertAdjacentHTML("afterbegin", `<div id="err">Erreur : ${e.message}</div>`);
  throw e;
}
