// Labo page: precomputed comparison images for the two side experiments (spikes/ao, spikes/glass).
const $ = (id) => document.getElementById(id);

const EFFECTS = {
  ao: {
    label: "Ambient occlusion",
    methods: { gt: "Ray tracing", classic: "SSAO classique", neural: "Réseau" },
    reference: "gt", left: "classic", right: "neural", view: 1,
    errNote: "Erreur moyenne dans les zones d'ombre, par rapport au ray tracing. Plus bas, c'est mieux.",
    story: "Le réseau voit exactement ce que voit un SSAO classique, le depth buffer et les normales autour du pixel, et apprend une AO calculée en ray tracing. Il fait presque deux fois moins d'erreur dans les zones d'ombre et a moins de halos. Mais à l'œil le gain reste modeste, et les deux ratent l'ombre de contact sous le dragon : en screen space, on ne voit pas ce qui est caché.",
  },
  glass: {
    label: "Verre",
    methods: { ref: "Mitsuba", traced: "Trajet exact", classic: "Réfraction simple", neural: "Réseau" },
    reference: "ref", left: "neural", right: "ref", view: 3,
    errNote: "Erreur moyenne sur le verre, par rapport au rendu Mitsuba. Plus bas, c'est mieux.",
    story: "Le réseau devait apprendre où la lumière ressort du dragon en verre, rebonds internes compris. Ça ne marche pas : le rendu est granuleux et pas meilleur que la réfraction simple utilisée en temps réel. Dans le verre, deux pixels voisins peuvent ressortir dans des directions opposées, et des latents interpolés en douceur ne savent pas représenter ça. Le trajet exact, lui, se rapproche nettement de Mitsuba.",
  },
};

const state = { effect: "ao", view: EFFECTS.ao.view, left: EFFECTS.ao.left, right: EFFECTS.ao.right, split: 0.5 };
const data = {};

const seg = (el, items, current, onPick) => {
  el.innerHTML = items.map(([k, v]) => `<button data-k="${k}" class="${k === current ? "on" : ""}">${v}</button>`).join("");
  el.onclick = (e) => { if (e.target.dataset.k !== undefined) onPick(e.target.dataset.k); };
};
const fmt = (v) => v.toFixed(3).replace(".", ",");

function render() {
  const fx = EFFECTS[state.effect], d = data[state.effect];
  const n = d.views.length;
  seg($("effects"), Object.entries(EFFECTS).map(([k, v]) => [k, v.label]), state.effect, (k) => {
    state.effect = k; state.view = EFFECTS[k].view; state.left = EFFECTS[k].left; state.right = EFFECTS[k].right; render();
  });
  seg($("views"), Array.from({ length: n }, (_, i) => [String(i), String(i + 1)]), String(state.view), (k) => { state.view = +k; render(); });
  const methods = Object.entries(fx.methods);
  seg($("left"), methods, state.left, (k) => { state.left = k; render(); });
  seg($("right"), methods, state.right, (k) => { state.right = k; render(); });
  const src = (k) => `data/labo/${state.effect}/view${state.view}-${k}.jpg`;
  $("imgL").src = src(state.left);
  $("imgR").src = src(state.right);
  $("tagL").textContent = fx.methods[state.left];
  $("tagR").textContent = fx.methods[state.right];
  const errs = d.views[state.view].errors;
  const rows = Object.entries(errs).filter(([k]) => k !== fx.reference);
  const best = Math.min(...rows.map(([, v]) => v));
  $("errs").innerHTML = rows.map(([k, v]) =>
    `<span>${fx.methods[k]}</span><span class="val ${v === best ? "best" : ""}">${fmt(v)}</span>`).join("");
  $("errNote").textContent = fx.errNote;
  $("story").textContent = fx.story;
}

function placeSplit() {
  $("handle").style.left = `${state.split * 100}%`;
  $("stage").style.setProperty("--split", `${state.split * 100}%`);
}

const handle = $("handle");
let dragging = false;
handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); });
handle.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const r = $("stage").getBoundingClientRect();
  state.split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  placeSplit();
});
handle.addEventListener("pointerup", () => { dragging = false; });
handle.addEventListener("pointercancel", () => { dragging = false; });

try {
  const [ao, glass] = await Promise.all(["ao", "glass"].map((k) => fetch(`data/labo/${k}.json`).then((r) => {
    if (!r.ok) throw new Error(`data/labo/${k}.json : HTTP ${r.status}`);
    return r.json();
  })));
  Object.assign(data, { ao, glass });
  placeSplit();
  render();
} catch (e) {
  document.body.insertAdjacentHTML("afterbegin", `<div id="err">Erreur : ${e.message}</div>`);
}
