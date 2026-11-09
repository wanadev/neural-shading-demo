// Labo page: two live engines (labo-ao.js, labo-glass.js). Ground truths, the Mitsuba reference and
// the failed glass network are precomputed images at 4 reference poses: picking one snaps the camera
// there, and any camera move falls back to a live method.
const $ = (id) => document.getElementById(id);

const EFFECTS = {
  ao: {
    label: "Ambient occlusion", canvas: "aoCanvas",
    methods: { gt: "Ray tracing", classic: "SSAO simple", gtao: "GTAO", neural: "Réseau" },
    reference: "gt", precomputed: ["gt"], fallback: "neural", left: "gtao", right: "neural", view: 1,
    errNote: "Erreur moyenne dans les zones d'ombre, par rapport au ray tracing, après débruitage (comme à l'écran), à cette vue de référence. Plus bas, c'est mieux.",
    note: "SSAO simple, GTAO (three.js) et réseau tournent en temps réel. Le ray tracing est précalculé aux 4 vues de référence : le choisir cale la caméra dessus.",
    story: "Le réseau voit exactement ce que voit un SSAO, le depth buffer et les normales autour du pixel, et apprend une AO calculée en ray tracing. Face au GTAO de three.js, réglé lui aussi au mieux sur les mêmes vues, il reste plus proche du ray tracing. Mais tous ratent une partie de l'ombre de contact sous le dragon : en screen space, on ne voit pas ce qui est caché.",
  },
  glass: {
    label: "Verre", canvas: "glassCanvas",
    methods: { ref: "Mitsuba", traced: "Trajet exact", classic: "Réfraction", neural: "Réseau" },
    reference: "ref", precomputed: ["ref", "neural"], fallback: "traced", left: "classic", right: "traced", view: 3,
    errNote: "Erreur moyenne sur le verre, par rapport au rendu Mitsuba, à cette vue de référence. Plus bas, c'est mieux.",
    note: "Réfraction et trajet exact tournent en temps réel. Le trajet exact lance de vrais rayons à travers le mesh dans le shader (three-mesh-bvh). Mitsuba et le réseau sont précalculés aux 4 vues de référence.",
    story: "Le réseau devait apprendre où la lumière ressort du dragon en verre. Il échoue : dans le verre, deux pixels voisins peuvent ressortir dans des directions opposées, et des latents interpolés en douceur ne savent pas représenter ça. Ici, le bon outil n'est pas un réseau mais du ray tracing directement dans le shader, qui tient en temps réel.",
  },
};

const state = { effect: "ao", view: EFFECTS.ao.view, left: EFFECTS.ao.left, right: EFFECTS.ao.right, split: 0.5, applied: true, aoScale: 0.5 };
const data = {}, engines = {};
let cfg = null, net = null, glassViews = null, gtaoParams = null;

const seg = (el, items, current, onPick) => {
  el.innerHTML = items.map(([k, v]) => `<button data-k="${k}" class="${k === current ? "on" : ""}">${v}</button>`).join("");
  el.onclick = (e) => { if (e.target.dataset.k !== undefined) onPick(e.target.dataset.k); };
};
const fmt = (v) => v.toFixed(3).replace(".", ",");
const engine = () => engines[state.effect];

async function ensureEngine(effect) {
  if (engines[effect]) return engines[effect];
  const canvas = $(EFFECTS[effect].canvas);
  const e = effect === "ao"
    ? await (await import("./labo-ao.js")).createAo({ canvas, net, cfg, gtaoParams })
    : await (await import("./labo-glass.js")).createGlass({ canvas, views: glassViews, cfg });
  $("stage").append(e.stats.dom);
  e.onMove(() => {  // a real camera move leaves the reference pose
    if (state.effect !== effect) return;
    const fx = EFFECTS[effect];
    let changed = false;
    for (const side of ["left", "right"]) if (fx.precomputed.includes(state[side])) { state[side] = fx.fallback; changed = true; }
    if (changed) render();
  });
  e.resize($("stage").clientWidth);
  e.state.split = state.split;
  e.setPose(state.view, false);
  engines[effect] = e;
  return e;
}

async function setEffect(k) {
  Object.assign(state, { effect: k, view: EFFECTS[k].view, left: EFFECTS[k].left, right: EFFECTS[k].right });
  render();  // hides the other canvas right away
  const e = await ensureEngine(k);
  e.setPose(state.view, false);
  render();
}

function pickSide(side, k) {
  state[side] = k;
  if (EFFECTS[state.effect].precomputed.includes(k) && engine()) {  // only exists at the reference poses
    state.view = engine().nearestPose();
    engine().setPose(state.view);
  }
  render();
}

function render() {
  const fx = EFFECTS[state.effect], d = data[state.effect];
  seg($("effects"), Object.entries(EFFECTS).map(([k, v]) => [k, v.label]), state.effect, setEffect);
  seg($("views"), d.views.map((_, i) => [String(i), String(i + 1)]), String(state.view), (k) => {
    state.view = +k;
    engine()?.setPose(state.view);
    render();
  });
  const methods = Object.entries(fx.methods);
  seg($("left"), methods, state.left, (k) => pickSide("left", k));
  seg($("right"), methods, state.right, (k) => pickSide("right", k));
  for (const [k, v] of Object.entries(EFFECTS)) {
    const on = k === state.effect;
    $(v.canvas).hidden = !on;
    if (engines[k]) { engines[k].stats.dom.hidden = !on; on ? engines[k].resume() : engines[k].pause(); }
  }
  $("aoOptions").hidden = state.effect !== "ao";
  seg($("display"), [["1", "AO appliquée"], ["0", "AO seule"]], state.applied ? "1" : "0", (k) => { state.applied = k === "1"; render(); });
  seg($("aores"), [["0.5", "demi"], ["1", "pleine"]], String(state.aoScale), (k) => { state.aoScale = +k; engines.ao?.setAoScale(state.aoScale); render(); });
  if (engine()) Object.assign(engine().state, { left: state.left, right: state.right, view: state.view, applied: state.applied });
  $("tagL").textContent = fx.methods[state.left];
  $("tagR").textContent = fx.methods[state.right];
  const rows = Object.entries(d.views[state.view].errors).filter(([k]) => k !== fx.reference && fx.methods[k]);
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
  for (const e of Object.values(engines)) e.state.split = state.split;
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
  let aoData, glass;
  [aoData, glass, net, cfg, gtaoParams, glassViews] = await Promise.all([
    json("data/labo/ao.json"), json("data/labo/glass.json"), json("data/labo/ao_net.json"), json("data/config.json"),
    json("data/labo/gtao.json").catch(() => null), json("data/labo/glass_views.json")]);
  if (gtaoParams) aoData.views.forEach((v, i) => { v.errors.gtao = gtaoParams.errors[i]; });
  Object.assign(data, { ao: aoData, glass });

  const q = new URLSearchParams(location.search);  // deep links: ?effect=glass&left=classic&right=traced&display=0
  const effect = Object.hasOwn(EFFECTS, q.get("effect") ?? "") ? q.get("effect") : "ao";
  Object.assign(state, { effect, view: EFFECTS[effect].view, left: EFFECTS[effect].left, right: EFFECTS[effect].right });
  for (const side of ["left", "right"]) if (Object.hasOwn(EFFECTS[effect].methods, q.get(side) ?? "")) state[side] = q.get(side);
  if (q.get("display") === "0") state.applied = false;
  await ensureEngine(effect);
  if (q.has("debug")) window.labo = engines;
  addEventListener("resize", () => { for (const e of Object.values(engines)) e.resize($("stage").clientWidth); });
  placeSplit();
  render();
  for (const [flag, eff] of [["aocheck", "ao"], ["glasscheck", "glass"]]) {
    if (!q.has(flag)) continue;
    const e = await ensureEngine(eff);
    const tag = flag.toUpperCase();
    $("story").textContent = `${tag} en cours`;
    const res = await e.check().catch((err) => { $("story").textContent = `${tag} ERREUR ${err.message}`; throw err; });
    $("story").textContent = `${tag} ` + res.map((r) => `vue ${r.view}: écart moyen ${(r.mae * 100).toFixed(2)} %` +
      (r.bad !== undefined ? `, pixels > 10 % : ${(r.bad * 100).toFixed(2)} %` : "")).join(" | ");
    e.resize($("stage").clientWidth);
  }
} catch (e) {
  document.body.insertAdjacentHTML("afterbegin", `<div id="err">Erreur : ${e.message}</div>`);
  throw e;
}
