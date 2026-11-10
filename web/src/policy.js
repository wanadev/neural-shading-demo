// Small decisions the page makes about what it can show right now.

// Bloom needs a half-float render target; without it, render straight to the screen.
export const pickPipeline = ({ halfFloatRT }) => (halfFloatRT ? "hdr" : "ldr");

// A "Path tracé" side shows neural until its (lazy, several-MB) reference image has loaded.
export const sideMode = (requested, refReady) => (requested === "ref" && !refReady ? "neural" : requested);

// What each pipeline may use. PMREM (env reflections, blurred background) renders into half-float
// targets too, so the ldr fallback must avoid it.
export const pipelineSettings = (pipeline) => (pipeline === "hdr"
  ? { useEnvMap: true, backgroundBlurriness: 0.4 }
  : { useEnvMap: false, backgroundBlurriness: 0 });

const SIDES = ["classic", "neural", "ref"];

// Deep links. The page applies them in this order: pose, then lamp, then sides — a "ref" side snaps
// the lamp to its reference pose last, so a ?lamp= can never sit under a reference image it does not match.
export function parseDeepLink(search, cfg) {
  const q = new URLSearchParams(search);
  const sides = {};
  for (const side of ["left", "right"]) if (SIDES.includes(q.get(side))) sides[side] = q.get(side);
  const pose = Number.parseInt(q.get("pose"), 10);
  const [az, el] = (q.get("lamp") ?? "").split(",").map((v) => (v === "" ? NaN : Number(v)));
  const split = q.has("split") && Number.isFinite(Number(q.get("split"))) ? Math.min(1, Math.max(0, Number(q.get("split")))) : null;
  return {
    pose: pose >= 0 && pose < cfg.ref_poses.length ? pose : null,
    lamp: Number.isFinite(az) && Number.isFinite(el) ? { az_deg: az, el_deg: el } : null,
    sides, split,
  };
}
